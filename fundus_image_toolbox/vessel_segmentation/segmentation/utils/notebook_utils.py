import torch
import numpy as np 
import cv2
from sklearn.metrics import f1_score 
from scipy.optimize import minimize
from ..utils.model_definition import FR_UNet
from scipy.stats import entropy
import torch.nn as nn
from ..models.model_utils import dice_metric


def get_ensemble(model_lists, dropout=False, device=None):
    all_models = []

    for predictor_path in model_lists:

        model = FR_UNet(num_classes=1, num_channels=3, feature_scale=2, dropout=0.1)
        checkpoint= torch.load(predictor_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        if dropout: # Dropout ensemble
            model.apply(enable_dropout) 

        all_models.append(model)

    return all_models

def dist2center(ref_point):

    """
        ref_point: tuple of 2
    """

    point = np.array(ref_point)
    # center point
    point[0] = ref_point[0] - 510/2 # accounting for slim black boarders
    point[1] = ref_point[1] - 510/2

    dist = np.sqrt(point[0] ** 2 + point[1] ** 2)

    return dist

def convolve_uncertainty(uncertainty_map, patch_size, stride=1):
    """
        ::param::    
        uncertainty_map: [np.array] of shape nxn
        patch_size: [int] defines side length of square patch
        stride: [int]
        
        ::return::
        Patch Identifier with Convolution Value
    """

    patch_convolved = []
    reference_points = [] # identifies patches
    patch_dict = {'reference': reference_points,
                  'patch uncertainty': patch_convolved}
    

    def sliding_window(arr, step_size=stride, window_size=patch_size):
        """
            Iterator which yields a binary mask for patch extraction
            alongside identifying coordinates of the reference point.
        """
        for y in range(0, arr.shape[0] - patch_size, step_size):
            for x in range(0, arr.shape[1] - patch_size, step_size):

                bool_img = np.zeros_like(arr) * False
                bool_img[y:y + window_size, x:x + window_size] = True
                
                yield (y, x, bool_img.astype(bool))

    windows = sliding_window(uncertainty_map, stride, patch_size)
    

    for y, x, window in windows:
        patch_value = uncertainty_map[window].sum()
        patch_convolved.append(patch_value)
        reference_points.append((y, x))

    return patch_dict


def estimate_dice_li(prediction, return_separately=False):

    """ 
        Estimate DICE from prediction mask.
        Estimator from Li et al. (MICCAI, 2022)

        `prediction` may contain values between 0 and 1.
    """

    mask0 = prediction <= 0.5
    mask1 = prediction > 0.5

    n0 = np.sum(mask0)
    n1 = np.sum(mask1)
    n = n0 + n1

    enumerator0 = 2 * np.sum(1 - prediction[mask0])
    enumerator1 = 2 * np.sum(prediction[mask1])

    denominator0 = n0 + np.sum(1 - prediction)
    denominator1 = n1 + np.sum(prediction)

    sDSC0 = enumerator0 / denominator0
    sDSC1 = enumerator1 / denominator1

    if return_separately:
        return enumerator1, denominator0
    else:
        return sDSC1


def dice(y_true, y_pred):

    """ Compute DICE score between predicted mask and ground truth mask """


    # Ground truth 
    # y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)


    # Prediction 
    # y_pred = y_pred.cpu().numpy()
    # y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    return f1_score(y_true, y_pred, average='binary')


def clahe_equalized(image):  
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8))
    lab[...,0] = clahe.apply(lab[...,0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr 

def calibration_curve( 
     y_true, 
     y_prob, 
     *, 
     normalize="deprecated", 
     n_bins=5, 
     strategy="uniform", 
 ): 
     

    """ Modified from sklearn.calibration.calibration_curve

        y_true, y_prob: np.array (1D)   
        return: all of shape (n_bins,)

        prob_true: proportion of samples whose class is positive in each bin
        prob_pred: The mean predicted probability in each bin
        no_samples: Number of samples within each bin
        bins: boarders of bins 

    """
    

    if y_prob.min() < 0 or y_prob.max() > 1: 
        raise ValueError("y_prob has values outside [0, 1].") 

    labels = np.unique(y_true) 
    if len(labels) > 2: 
        raise ValueError( 
            f"Only binary classification is supported. Provided labels {labels}." 
        ) 
    

    if strategy == "quantile":  # Determine bin edges by distribution of data 
        quantiles = np.linspace(0, 1, n_bins + 1) 
        bins = np.percentile(y_prob, quantiles * 100) 
    elif strategy == "uniform": 
        bins = np.linspace(0.0, 1.0, n_bins + 1) 
    else: 
        raise ValueError( 
            "Invalid entry to 'strategy' input. Strategy " 
            "must be either 'quantile' or 'uniform'." 
        ) 

    binids = np.searchsorted(bins[1:-1], y_prob) 

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins)) 
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins)) 
    bin_total = np.bincount(binids, minlength=len(bins)) 

    nonzero = bin_total != 0 
    prob_true = bin_true[nonzero] / bin_total[nonzero] 
    prob_pred = bin_sums[nonzero] / bin_total[nonzero] 

    res = {'pos_labels': prob_true,
        'mean_confidence': prob_pred,
        'bin_counts': bin_total,
        'bin_upper': bins}

    return res 

# Data Process useful function
def process_image(image, mask):
    x = image/255.0
    x = np.transpose(x, (2, 0, 1))      ## (3, 512, 512)
    
    x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)

    """ Reading mask """
    y = mask/255.0  
    y = np.expand_dims(y, axis=0)            ## (1, 512, 512)
                                   ## Normalize image to [0,1] range
    y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
    y = y.astype(np.float32)
    y = torch.from_numpy(y)

    return x, y


def enable_dropout(module):
  if type(module) == nn.Dropout2d:
        module.train()


def Entropy(X, axis=-1):
    '''
      Shannon entropy/relative entropy of the given distribution
      the 1e-8 is added for numericl stability.

    '''
    return -1 * np.sum(X * np.log(X+1e-8), axis=axis)


def error_fn(y_true, prediction):
    
    y_true = y_true.cpu().numpy()
    
    return (y_true - prediction)


def measure_uncertainty(preds):
    # calculate mean
    mean_preds = np.mean(preds, axis=0)
    # calculate entropy
    entropy = Entropy(np.mean(preds, axis=0), axis=1)
    # Expected entropy of the predictive under the parameter posterior
    entropy_exp = np.mean(Entropy(preds, axis=1))
    # calculate mutual info
    # Equation 2 of https://arxiv.org/pdf/1711.08244.pdf
    mutual_info = entropy - entropy_exp
    # calculate variance
    variance = np.std(preds[:], 0)
    # calculate aleatoric uncertainty
    aleatoric = np.mean(preds*(1-preds), axis=0)
    # calculate epistemic uncertainty
    epistemic = np.mean(preds**2, axis=0) - np.mean(preds, axis=0)**2  #TODO: check if this is same with entropy


    return mean_preds, entropy, mutual_info, variance, aleatoric, epistemic


def ensemble_inference(model_list, image, y_true):

    preds = []
    for model in model_list:
        with torch.no_grad():
            pred = model(image)
            pred = torch.sigmoid(pred).cpu().numpy()
            preds.append(pred)
    
    preds = np.array(preds)

    # calculate the uncertainty metrics
    mean_prediction, entropy, mutual_info, variance, aleatoric, epistemic = measure_uncertainty(preds)

    # dice_score = dice(y_true, prediction)
    
    error = error_fn(y_true, mean_prediction)
 
    # uncertainty_map  = epistemic #aleatoric + 

    return preds, np.squeeze(mean_prediction), np.squeeze(entropy), np.squeeze(mutual_info), np.squeeze(variance), np.squeeze(error)


def inference(model, image, y_true, N=1):

    # perform N stochastic forward passes and then append the preds
    preds = []
    for n in range(N):
        with torch.no_grad():

            pred = model(image)
            # comment this to run in the logit space
            pred = torch.sigmoid(pred).cpu().numpy()
            preds.append(pred)

    preds = np.asarray(preds)

    # calculate the uncertainty metrics
    prediction, entropy, mutual_info, variance, aleatoric, epistemic = get_uncertainty_measurement(preds)

    dice_score = dice_metric(y_true, prediction)
    
    error = error_fn(y_true, prediction)

    uncertainty_map  = epistemic #aleatoric + 

    return np.squeeze(prediction), np.squeeze(entropy), np.squeeze(error), dice_score 



### TS Scaling ### 

def eval_func(T, logits_cali, acc_cali):
    """ 
        Evaluate AC - Acc.
        Will be handed to the optimizer. 

        logits_cali: np.array of shape (n, ) containing the logits for class 1
        acc_cali: float
    """

    # scale
    ts_logits = logits_cali / T
    # apply sigmoid
    ts_pyIx = 1 / (1 + np.exp(-ts_logits))
    # average
    AC = ts_pyIx.mean()
    # compute estimation error
    ACE = np.abs(AC - acc_cali)

    return ACE

def scale_temp(logits_cali, acc_cali):
    """
        Optimize T
    """

    optim = minimize(fun=eval_func, args=(logits_cali, acc_cali),
                     x0=np.array([2]), method='Nelder-Mead',
                     tol=1e-07)
    
    return optim.x[0]

# compute new confidences
def apply_temperature(T, logits_cali, dice_true, return_separately=False):

    n_imgs = np.max(np.array(dice_true).shape)
    ts_logits = logits_cali / T
    ts_pyIx = softmax(ts_logits)
    est_dice_cali = np.apply_along_axis(func1d=estimate_dice_li, axis=1,
                                        arr=ts_pyIx.reshape(n_imgs, -1),
                                        return_separately=return_separately)

    if not return_separately:
        # and correlation to true dice
        corr_dice = np.corrcoef(est_dice_cali, dice_true)[0, 1]
    else:
        corr_dice = -9

    temperature_dict = {'estimated dice': est_dice_cali, 'correlation': corr_dice,
                        'predictions': ts_pyIx, 'T': T, 'true dice': dice_true}

    return temperature_dict

### TS-Scaling for Segmentation

def eval_func_dice(T, predictions, true_dice):
    """ 
        Evaluate est. DICE - true DICE.
        Will be handed to the optimizer. 

        predictions: [list] of [np.array] (model outputs)
        true_dice: float
    """

    # scale
    ts_logits = [logits_cali / T for logits_cali in predictions]
    # apply sigmoid
    ts_pyIx = [1 / (1 + np.exp(-one_img)) for one_img in ts_logits]
  
    # compute estimation error
    est_dice = [estimate_dice_li(one_img) for one_img in ts_pyIx]
    mean_est_dice = np.mean(est_dice) # for ensembles

    dice_error = np.abs(mean_est_dice - true_dice)
    
    return dice_error

def scale_temp_dice(predictions, true_dice):
    """
        Optimize T
    """

    optim = minimize(fun=eval_func_dice, args=(predictions, true_dice),
                     x0=np.array([2]), method='Nelder-Mead',
                     tol=1e-04)
    
    return optim.x[0]


def pass_and_log(model_list, device, x_paths, y_paths):

    # init nested list structure for all variables of interest
    dice_true, dice_estimate, pred_logit_all, logits_pred1_list, y_pred1_list = [[[] for k in range(len(model_list))] for f in range(5)]
    # logits_pred1, y_pred1 for naive TS scaling 

    for j, model in enumerate(model_list):

        model = model.to(device)
        logits_pred1 = []
        y_pred1 = []

        for i, (x, y) in enumerate(zip(x_paths, y_paths)):

            # read and preprocess image
            image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
            image = clahe_equalized(image)
            image = cv2.resize(image, (512,512))

            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
            mask = cv2.resize(mask, (512,512))

            x, y = process_image(image, mask)
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():

                # forward pass 
                pred_logit = model(x.to(device)) # torch object @cuda
                pred_logit_all[j].append(pred_logit)
                pred = torch.sigmoid(pred_logit)

                pred_mask = np.array(pred.cpu() > 0.5, dtype=int).reshape(-1)
                y = np.array(y.cpu(), dtype=int).reshape(-1)
                
                # log information which is required for naive TS scaling later, i.e.
                # select all positively classified instances and store logits, gt
                
                pred1_idx = np.where(pred_mask == 1)[0]
                logits_pred1 = np.append(logits_pred1, np.array(pred_logit.cpu()).reshape(-1)[pred1_idx])
                y_pred1 = np.append(y_pred1, np.array(y.reshape(-1)[pred1_idx]))

                # compute true dice
                dice_true[j].append(f1_score(y, pred_mask, average='binary'))
                # estimate dice
                dice_estimate[j].append(estimate_dice_li(np.array(pred.cpu()).reshape(-1)))

        logits_pred1_list[j] = logits_pred1
        y_pred1_list[j] = y_pred1

    for h in range(len(pred_logit_all)):
        pred_logit_all[h] = [tens.cpu().numpy().squeeze() for tens in pred_logit_all[h]]

    pass_dict = {'true dice': dice_true, 'estimated dice': dice_estimate,
                 'logits': pred_logit_all, 'logits | y_hat=1': logits_pred1,
                 'y | y_hat=1': y_pred1}

    return pass_dict


def pass_and_log_ood(model_list, device, dataloader):

    """
        Same as pass_and_log but for datasets for which we use a class and not
        manual dataloading.
    """

    # init nested list structure for all variables of interest
    dice_true, dice_estimate, pred_logit_all, logits_pred1_list, y_pred1_list = [[[] for k in range(len(model_list))] for f in range(5)]
    # logits_pred1, y_pred1 for naive TS scaling 

    for j, model in enumerate(model_list):

        model = model.to(device)
        logits_pred1 = []
        y_pred1 = []

        for i, (x, y) in enumerate(dataloader):

            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():

                # forward pass 
                pred_logit = model(x.to(device)) # torch object @cuda
                pred_logit_all[j].append(pred_logit)
                pred = torch.sigmoid(pred_logit)

                pred_mask = np.array(pred.cpu() > 0.5, dtype=int).reshape(-1)
                y = np.array(y.cpu(), dtype=int).reshape(-1)
                
                # log information which is required for naive TS scaling later, i.e.
                # select all positively classified instances and store logits, gt
                
                pred1_idx = np.where(pred_mask == 1)[0]
                logits_pred1 = np.append(logits_pred1, np.array(pred_logit.cpu()).reshape(-1)[pred1_idx])
                y_pred1 = np.append(y_pred1, np.array(y.reshape(-1)[pred1_idx]))

                # compute true dice
                dice_true[j].append(f1_score(y, pred_mask, average='binary'))
                # estimate dice
                dice_estimate[j].append(estimate_dice_li(np.array(pred.cpu()).reshape(-1)))

        logits_pred1_list[j] = logits_pred1
        y_pred1_list[j] = y_pred1

    for h in range(len(pred_logit_all)):
        pred_logit_all[h] = [tens.cpu().numpy().squeeze() for tens in pred_logit_all[h]]

    pass_dict = {'true dice': dice_true, 'estimated dice': dice_estimate,
                 'logits': pred_logit_all, 'logits | y_hat=1': logits_pred1,
                 'y | y_hat=1': y_pred1}

    return pass_dict


def softmax(arr):

    return 1 / (1 + np.exp(-arr))

def pixel_uncertainty(arr, method='variance'):
    """
        INPUT:
        arr: [np.array] with shape (m_ensemble, 1, W, H)
        method: [str] either 'variance' or 'entropy'

        OUTPUT: np.array with shape (1, W, H)
    """
    if method == 'variance':
        uncertainty_map = arr.var(axis=0)
    elif method == 'entropy':
        uncertainty_map = entropy(arr, axis=0)
    else:
        'Method ' + method + 'not implemented. Must be variance or entropy'

    return uncertainty_map

def get_patches(gt, prediction, patch_size, stride, return_patches=False):

    """
        gt: np.array of shape (n, n) (binary)
        prediction: np.array of shape (n, n) (binary prediction)
        patch_size: int -- side length of square defining the window
        stride: int -- how many pixels to slide the window to the side
        return_patches: bool 

        return: dict of multiple lists: estimated and true dice ... 
    """ 

    patch_true_dice = []
    patch_est_dice = []
    patch_pos_pix = []
    rest_est_dice = []
    
    if return_patches:
        patches_gt = []
        patches_pred = []
        reference_points = []
    
        patch_dict = {'reference': reference_points,
                    'patches_gt': patches_gt,
                    'patches_pred': patches_pred} 

    def sliding_window(arr, step_size=stride, window_size=patch_size):
            
        for y in range(0, arr.shape[0] - patch_size, step_size):
           for x in range(0, arr.shape[1] - patch_size, step_size):

                bool_img = np.zeros_like(gt) * False
                bool_img[y:y + window_size, x:x + window_size] = True
                
                yield (y, x, bool_img.astype(bool))
            
   
    windows = sliding_window(gt, stride, patch_size)


    for y, x, window in windows:
        # determine shape of window (sometimes it's not square, when we're at the boarders)
        # compute true dice of patch
        gt_patch = gt[window].reshape(patch_size, patch_size)
        pred_patch = prediction[window].reshape(patch_size, patch_size)
        rest = prediction[~window].flatten()

        patches_gt.append(gt_patch)
        patches_pred.append(pred_patch)
        reference_points.append((y, x))

        true_dice = dice_metric(gt_patch, pred_patch)
        est_dice = estimate_dice_li(pred_patch)
        rest_dice = estimate_dice_li(rest)

        patch_true_dice.append(true_dice) 
        patch_est_dice.append(est_dice)
        rest_est_dice.append(rest_dice)
    
        # Compute fraction of positively predicted pixels
        patch_pos_pix.append(np.mean(pred_patch > 0.5))

    if return_patches:
        return {'true': patch_true_dice, 'est': patch_est_dice,
                'patches_gt': patches_gt, 'patches_pred': patches_pred,
                'reference_points': reference_points, 'patch_size': patch_size,
                'positive predictions': patch_pos_pix,
                'remaining est': rest_est_dice}
    else:
        return {'true': patch_true_dice, 'est': patch_est_dice}