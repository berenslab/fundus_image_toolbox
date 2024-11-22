import numpy as np
import torch
import torch.nn as nn

# Data Process useful function
def process_image(image, mask):
    x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
    x = x/255.0
    x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)

    """ Reading mask """
    
    y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
    y = y/255.0                                 ## Normalize image to [0,1] range
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


def dice_metric(ground_truth, predictions):
    """

    Returns Dice coefficient for a single example.

    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target, 
                     with shape [W, H, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [W, H, D].

    Returns:
      Dice coefficient overlap (`float` in [0.0, 1.0])
      between `ground_truth` and `predictions`.

    """

    # Cast to float32 type
    ground_truth = ground_truth.cpu().numpy().astype("float32")
    predictions = predictions.astype("float32")

    # Calculate intersection and union of y_true and y_predict
    intersection = np.sum(predictions * ground_truth)
    union = np.sum(predictions) + np.sum(ground_truth)

    # Calcualte dice metric
    if intersection == 0.0 and union == 0.0:
      dice = 1.0
    else:
      dice = (2. * intersection) / (union)

    return dice


def error_fn(y_true, prediction):
    
    y_true = y_true.cpu().numpy()
    
    return (y_true - prediction)


def get_uncertainty_measurement(preds):
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


def inference(model, image, y_true, N=10):
    # Make inference based on N forward pass.

    # perform N stochastic forward passes and then append the preds
    preds = []
    for n in range(N):
        with torch.no_grad():
            pred = model(image)
            pred = torch.sigmoid(pred).cpu().numpy() # comment this to run in the logit space
            preds.append(pred)
    
    preds = np.asarray(preds)

    # calculate the uncertainty metrics
    prediction, entropy, mutual_info, variance, aleatoric, epistemic = get_uncertainty_measurement(preds)

    dice_score = dice_metric(y_true, prediction)
    
    error = error_fn(y_true, prediction)
 
    # uncertainty_map  = epistemic #aleatoric + 

    return np.squeeze(prediction), np.squeeze(entropy), np.squeeze(error), dice_score, y_true 
