import random
import sys
import time

from torch.nn import functional as F
import torch
import torch.nn as nn

from ..loss import DiceBCELoss, DiceLoss
from ..loss import triplet_margin_loss_gor, triplet_margin_loss_gor_one, sos_reg

from ..common import remove_borders, sample_keypoint_desc, simple_nms, nms, \
    sample_descriptors
from ..common import get_gaussian_kernel, affine_images

from ..common import sample_keypoint_desc, nms, simple_nms


def update_value_map(value_map, points, value_map_config):
    """
    Update value maps used for recording learned keypoints from PKE,
    and getting the final learned keypoints which are combined of previous learned keypoints.
    :param value_map: previous value maps
    :param points: the learned keypoints in this epoch
    :param value_map_config:
    :return: the final learned keypoints combined of previous learning points
    """

    raw_value_map = value_map.clone()
    # used for record areas of value=0
    raw_value_map[value_map == 0] = -1
    area_set = value_map_config['area']
    area = area_set // 2

    value_increase_point = value_map_config['value_increase_point']
    value_increase_area = value_map_config['value_increase_area']
    value_decay = value_map_config['value_decay']

    h, w = value_map[0].shape
    for (x, y) in points:
        y_d = y - area // 2 if y - area // 2 > 0 else 0
        y_u = y + area // 2 if y + area // 2 < h else h
        x_l = x - area // 2 if x - area // 2 > 0 else 0
        x_r = x + area // 2 if x + area // 2 < w else w
        tmp = value_map[0, y_d:y_u, x_l:x_r]
        if value_map[0, y, x] != 0 or tmp.sum() == 0:
            value_map[0, y, x] += value_increase_point  # if there is no learned point before, then add a high value
        else:
            tmp[tmp > 0] += value_increase_area
            value_map[0, y_d:y_u, x_l:x_r] = tmp

    value_map[torch.where(
        value_map == raw_value_map)] -= value_decay  # value decay of positions that don't appear this time

    tmp = value_map.detach().clone()

    tmp = simple_nms(tmp.unsqueeze(0).float(), area_set*2)
    tmp = tmp.squeeze()

    final_points = torch.nonzero(tmp >= value_increase_point)
    final_points = torch.flip(final_points, [1]).long()  # to x, y
    return final_points


def mapping_points(grid, points, h, w):
    """ Using grid_inverse to apply affine transform on geo_points
        :return point set and its corresponding affine point set
    """

    grid_points = [(grid[s, k[:, 1].long(), k[:, 0].long()]) for s, k in
                   enumerate(points)]
    filter_points = []
    affine_points = []
    for s, k in enumerate(grid_points):  # filter bad geo_points
        idx = (k[:, 0] < 1) & (k[:, 0] > -1) & (k[:, 1] < 1) & (
                k[:, 1] > -1)
        gp = grid_points[s][idx]
        gp[:, 0] = (gp[:, 0] + 1) / 2 * (w - 1)
        gp[:, 1] = (gp[:, 1] + 1) / 2 * (h - 1)
        affine_points.append(gp)
        filter_points.append(points[s][idx])

    return filter_points, affine_points


def content_filter(descriptor_pred, affine_descriptor_pred, geo_points,
                   affine_geo_points, content_thresh=0.7, scale=8):
    """
    content-based matching in paper
    :param descriptor_pred: descriptors of input_image images
    :param affine_descriptor_pred: descriptors of affine images
    :param geo_points: 
    :param affine_geo_points:
    :param content_thresh:
    :param scale: down sampling size of descriptor_pred
    :return: content-filtered keypoints
    """

    descriptors = [sample_keypoint_desc(k[None], d[None], scale)[0].permute(1, 0)
                   for k, d in zip(geo_points, descriptor_pred)]
    aff_descriptors = [sample_keypoint_desc(k[None], d[None], scale)[0].permute(1, 0)
                       for k, d in zip(affine_geo_points, affine_descriptor_pred)]
    content_points = []
    affine_content_points = []
    dist = [torch.norm(descriptors[d][:, None] - aff_descriptors[d], dim=2, p=2)
            for d in range(len(descriptors))]
    for i in range(len(dist)):
        D = dist[i]
        if len(D) <= 1:
            content_points.append([])
            affine_content_points.append([])
            continue
        val, ind = torch.topk(D, 2, dim=1, largest=False)

        arange = torch.arange(len(D))
        # rule1 spatial correspondence
        c1 = ind[:, 0] == arange.to(ind.device)
        # rule2 pass the ratio test
        c2 = val[:, 0] < val[:, 1] * content_thresh

        check = c2 * c1
        content_points.append(geo_points[i][check])
        affine_content_points.append(affine_geo_points[i][check])
    return content_points, affine_content_points


def geometric_filter(affine_detector_pred, points, affine_points, max_num=1024, geometric_thresh=0.5):
    """
    geometric matching in paper
    :param affine_detector_pred: geo_points probability of affine image
    :param points: nms results of input_image image
    :param affine_points: nms results of affine image
    :param max_num: maximum number of learned keypoints
    :param geometric_thresh: 
    :return: geometric-filtered keypoints
    """
    geo_points = []
    affine_geo_points = []
    for s, k in enumerate(affine_points):
        sample_aff_values = affine_detector_pred[s, 0, k[:, 1].long(), k[:, 0].long()]
        check = sample_aff_values.squeeze() >= geometric_thresh
        geo_points.append(points[s][check][:max_num])
        affine_geo_points.append(k[check][:max_num])

    return geo_points, affine_geo_points


def pke_learn(detector_pred, descriptor_pred, grid_inverse, affine_detector_pred,
              affine_descriptor_pred, kernel, loss_cal, label_point_positions,
              value_map, config, PKE_learn=True):
    """
    pke process used for detector
    :param detector_pred: probability map from raw image
    :param descriptor_pred: prediction of descriptor_pred network
    :param kernel: used for gaussian heatmaps
    :param mask_kernel: used for masking initial keypoints
    :param grid_inverse: used for inverse
    :param loss_cal: loss (default is dice)
    :param label_point_positions: positions of keypoints on labels
    :param value_map: value map for recoding and selecting learned geo_points
    :param pke_learn: whether to use PKE
    :return: loss of detector, num of additional geo_points, updated value maps and enhanced labels
    """
    # used for masking initial keypoints on enhanced labels
    initial_label = F.conv2d(label_point_positions, kernel,
                             stride=1, padding=(kernel.shape[-1] - 1) // 2)
    initial_label[initial_label > 1] = 1

    if not PKE_learn:
        return loss_cal(detector_pred, initial_label.to(detector_pred)), 0, None, None, initial_label

    nms_size = config['nms_size']
    nms_thresh = config['nms_thresh']
    scale = 8

    enhanced_label = None
    geometric_thresh = config['geometric_thresh']
    content_thresh = config['content_thresh']
    with torch.no_grad():
        h, w = detector_pred.shape[2:]

        # number of learned points
        number_pts = 0
        points = nms(detector_pred, nms_thresh=nms_thresh, nms_size=nms_size,
                     detector_label=initial_label, mask=True)

        # geometric matching
        points, affine_points = mapping_points(grid_inverse, points, h, w)
        geo_points, affine_geo_points = geometric_filter(affine_detector_pred, points, affine_points,
                                                         geometric_thresh=geometric_thresh)


        # content matching
        content_points, affine_contend_points = content_filter(descriptor_pred, affine_descriptor_pred, geo_points,
                                                               affine_geo_points, content_thresh=content_thresh,
                                                               scale=scale)
        enhanced_label_pts = []
        for step in range(len(content_points)):
            # used to combine initial points and learned points
            positions = torch.where(label_point_positions[step, 0] == 1)
            if len(positions) == 2:
                positions = torch.cat((positions[1].unsqueeze(-1), positions[0].unsqueeze(-1)), -1)
            else:
                positions = positions[0]

            final_points = update_value_map(value_map[step], content_points[step], config)

            # final_points = torch.cat((final_points, positions))

            temp_label = torch.zeros([h, w]).to(detector_pred.device)

            temp_label[final_points[:, 1], final_points[:, 0]] = 0.5
            temp_label[positions[:, 1], positions[:, 0]] = 1

            enhanced_kps = nms(temp_label.unsqueeze(0).unsqueeze(0), 0.1, 10)[0]
            if len(enhanced_kps) < len(positions):
                enhanced_kps = positions
            # print(len(final_points), len(positions), len(enhanced_kps))
            number_pts += (len(enhanced_kps) - len(positions))
            # number_pts += (len(enhanced_kps) - len(positions)) if (len(enhanced_kps) - len(positions)) > 0 else 0

            temp_label[:] = 0
            temp_label[enhanced_kps[:, 1], enhanced_kps[:, 0]] = 1

            enhanced_label_pts.append(temp_label.unsqueeze(0).unsqueeze(0))

            temp_label = F.conv2d(temp_label.unsqueeze(0).unsqueeze(0), kernel, stride=1,
                                  padding=(kernel.shape[-1] - 1) // 2)  # generating gaussian heatmaps
            temp_label[temp_label > 1] = 1

            if enhanced_label is None:
                enhanced_label = temp_label
            else:
                enhanced_label = torch.cat((enhanced_label, temp_label))

    enhanced_label_pts = torch.cat(enhanced_label_pts)
    affine_pred_inverse = F.grid_sample(affine_detector_pred, grid_inverse, align_corners=True)

    loss1 = loss_cal(detector_pred, enhanced_label)  # L_geo
    loss2 = loss_cal(detector_pred, affine_pred_inverse)  # L_clf
    # pred_mask = (enhanced_label > 0) & (affine_pred_inverse != 0)
    # loss2 = loss_cal(detector_pred[pred_mask], affine_pred_inverse[pred_mask])  # L_clf

    # mask_pred = grid_inverse
    # loss2 = loss_cal(detector_pred[mask_pred], affine_pred_inverse[mask_pred])  # L_clf

    loss = loss1+loss2

    return loss, number_pts, value_map, enhanced_label_pts, enhanced_label


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class SuperRetina(nn.Module):
    def __init__(self, config=None, device='cpu', n_class=1):
        super().__init__()

        self.PKE_learn = True
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1, d2 = 64, 64, 128, 128, 256, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)

        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)

        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)

        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=4, stride=2, padding=0)
        self.convDc = torch.nn.Conv2d(d1, d2, kernel_size=1, stride=1, padding=0)

        self.trans_conv = nn.ConvTranspose2d(d1, d2, 2, stride=2)

        # Detector Head
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(c3 + c4, c3)
        self.dconv_up2 = double_conv(c2 + c3, c2)
        self.dconv_up1 = double_conv(c1 + c2, c1)

        self.conv_last = nn.Conv2d(c1, n_class, kernel_size=1)

        if config is not None:
            self.config = config

            self.nms_size = config['nms_size']
            self.nms_thresh = config['nms_thresh']
            self.scale = 8

            self.dice = DiceLoss()

            self.kernel = get_gaussian_kernel(kernlen=config['gaussian_kernel_size'],
                                              nsig=config['gaussian_sigma']).to(device)

        self.to(device)

    def network(self, x):
        x = self.relu(self.conv1a(x))
        conv1 = self.relu(self.conv1b(x))
        x = self.pool(conv1)
        x = self.relu(self.conv2a(x))
        conv2 = self.relu(self.conv2b(x))
        x = self.pool(conv2)
        x = self.relu(self.conv3a(x))
        conv3 = self.relu(self.conv3b(x))
        x = self.pool(conv3)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        cDb = self.relu(self.convDb(cDa))
        desc = self.convDc(cDb)

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        desc = self.trans_conv(desc)

        cPa = self.upsample(x)
        cPa = torch.cat([cPa, conv3], dim=1)

        cPa = self.dconv_up3(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv2], dim=1)

        cPa = self.dconv_up2(cPa)
        cPa = self.upsample(cPa)
        cPa = torch.cat([cPa, conv1], dim=1)

        cPa = self.dconv_up1(cPa)

        semi = self.conv_last(cPa)
        semi = torch.sigmoid(semi)

        return semi, desc

    def descriptor_loss(self, detector_pred, label_point_positions, descriptor_pred,
                        affine_descriptor_pred, grid_inverse, affine_detector_pred=None):
        """
        calculate descriptor loss, construct triples on raw images and affine images
        :param detector_pred: output of detector network
        :param label_point_positions: initial label points
        :param descriptor_pred: output of descriptor network
        :param affine_descriptor_pred: output of descriptor network, with affine images as input
        :param grid_inverse: used for inverse affine transformation
        :return: descriptor loss (triplet loss)
        """

        # sample keypoints on initial labels
        # label_descriptors, label_affine_descriptors, label_keypoints = \
        #     sample_descriptors(label_point_positions, descriptor_pred, affine_descriptor_pred, grid_inverse,
        #                        nms_size=self.nms_size, nms_thresh=self.nms_thresh, scale=self.scale)
        #
        # for s, kps in enumerate(label_keypoints):
        #     label_mask = torch.zeros(detector_pred[s].shape).to(detector_pred)
        #     label_mask[0, kps[:, 1].long(), kps[:, 0].long()] = 1
        #     label_mask = F.conv2d(label_mask.unsqueeze(0), self.mask_kernel, stride=1,
        #                           padding=(self.mask_kernel.shape[-1] - 1) // 2)
        #     detector_pred[s][label_mask[0] > 1e-5] = 0
        if not self.PKE_learn:
            detector_pred[:] = 0  # only learn from the initial labels
        detector_pred[label_point_positions == 1] = 10
        descriptors, affine_descriptors, keypoints = \
            sample_descriptors(detector_pred, descriptor_pred, affine_descriptor_pred, grid_inverse,
                               nms_size=self.nms_size, nms_thresh=self.nms_thresh, scale=self.scale,
                               affine_detector_pred=affine_detector_pred)

        # descriptors_tmp = []
        # affine_descriptor_tmp = []
        # for i in range(len(descriptors)):
        #     descriptors_tmp.append(torch.cat((descriptors[i], label_descriptors[i]), -1))
        #     affine_descriptor_tmp.append(torch.cat((affine_descriptors[i], label_affine_descriptors[i]), -1))
        # descriptors = descriptors_tmp
        # affine_descriptors = affine_descriptor_tmp

        positive = []
        negatives_hard = []
        negatives_random = []
        anchor = []
        D = descriptor_pred.shape[1]
        for i in range(len(affine_descriptors)):
            if affine_descriptors[i].shape[1] == 0:
                continue
            descriptor = descriptors[i]
            affine_descriptor = affine_descriptors[i]

            n = affine_descriptors[i].shape[1]
            if n > 1000:  # avoid OOM
                return torch.tensor(0., requires_grad=True).to(descriptor_pred), False

            descriptor = descriptor.view(D, -1, 1)
            affine_descriptor = affine_descriptor.view(D, 1, -1)
            ar = torch.arange(n)

            # random
            neg_index2 = []
            if n == 1:
                neg_index2.append(0)
            else:
                for j in range(n):
                    t = j
                    while t == j:
                        t = random.randint(0, n - 1)
                    neg_index2.append(t)
            neg_index2 = torch.tensor(neg_index2, dtype=torch.long).to(affine_descriptor)

            # hard
            with torch.no_grad():
                dis = torch.norm(descriptor - affine_descriptor, dim=0)
                dis[ar, ar] = dis.max() + 1
                neg_index1 = dis.argmin(axis=1)

            positive.append(affine_descriptor[:, 0, :].permute(1, 0))
            anchor.append(descriptor[:, :, 0].permute(1, 0))
            negatives_hard.append(affine_descriptor[:, 0, neg_index1.long(), ].permute(1, 0))
            negatives_random.append(affine_descriptor[:, 0, neg_index2.long(), ].permute(1, 0))

        if len(positive) == 0:
            return torch.tensor(0., requires_grad=True).to(descriptor_pred), False

        positive = torch.cat(positive)
        anchor = torch.cat(anchor)
        negatives_hard = torch.cat(negatives_hard)
        negatives_random = torch.cat(negatives_random)

        positive = F.normalize(positive, dim=-1, p=2)
        anchor = F.normalize(anchor, dim=-1, p=2)
        negatives_hard = F.normalize(negatives_hard, dim=-1, p=2)
        negatives_random = F.normalize(negatives_random, dim=-1, p=2)

        loss = triplet_margin_loss_gor(anchor, positive, negatives_hard, negatives_random, margin=0.8)

        # can also add sos reg term .
        # reg_term = sos_reg(anchor, positive, KNN=True, k=1, eps=1e-8)
        # if not torch.isnan(reg_term) and reg_term > 0:
        #     loss = loss + 0.1 * reg_term

        return loss, True

    def forward(self, x, label_point_positions=None, value_map=None, learn_index=None):
        """
        In interface phase, only need to input x
        :param x: retinal images
        :param label_point_positions: positions of keypoints on labels
        :param value_map: value maps, used to record history learned geo_points
        :param learn_index: index of input data with detector labels
        :param phase: distinguish dataset
        :return: if training, return loss, else return predictions
        """

        detector_pred, descriptor_pred = self.network(x)
        enhanced_label_pts = None
        enhanced_label = None

        if label_point_positions is not None:
            if self.PKE_learn:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = x.shape[0]
            else:
                loss_detector_num = len(learn_index[0])
                loss_descriptor_num = loss_detector_num

            number_pts = 0  # number of learned keypoints
            value_map_update = None
            loss_detector = torch.tensor(0., requires_grad=True).to(x)
            loss_descriptor = torch.tensor(0., requires_grad=True).to(x)

            with torch.no_grad():
                affine_x, grid, grid_inverse = affine_images(x, used_for='detector')
                affine_detector_pred, affine_descriptor_pred = self.network(affine_x)
            loss_cal = self.dice
            if len(learn_index[0]) != 0:
                loss_detector, number_pts, value_map_update, enhanced_label_pts, enhanced_label = \
                    pke_learn(detector_pred[learn_index], descriptor_pred[learn_index],
                              grid_inverse[learn_index], affine_detector_pred[learn_index],
                              affine_descriptor_pred[learn_index], self.kernel, loss_cal,
                              label_point_positions[learn_index], value_map[learn_index],
                              self.config, self.PKE_learn)

            #  For showing PKE process
            if enhanced_label_pts is not None:
                enhanced_label_pts_tmp = label_point_positions.clone()
                enhanced_label_pts_tmp[learn_index] = enhanced_label_pts
                enhanced_label_pts = enhanced_label_pts_tmp
            if enhanced_label is not None:
                enhanced_label_tmp = label_point_positions.clone()
                enhanced_label_tmp[learn_index] = enhanced_label
                enhanced_label = enhanced_label_tmp

            detector_pred_copy = detector_pred.clone().detach()
            # if value_map_update is not None:
            #     # optimize descriptors of recorded points
            #     detector_pred_copy[learn_index][value_map_update >=
            #                                     self.config['VALUE MAP'].getfloat('value_increase_point')] = 1
            #
            affine_x_for_desc, grid_for_desc, grid_inverse_for_desc = affine_images(x, used_for='descriptor')
            _, affine_descriptor_pred_for_desc = self.network(affine_x_for_desc)
            loss_descriptor, descriptor_train_flag = self.descriptor_loss(detector_pred_copy, label_point_positions,
                                                                          descriptor_pred,
                                                                          affine_descriptor_pred_for_desc,
                                                                          grid_inverse_for_desc)

            if self.PKE_learn and len(learn_index[0]) != 0:
                value_map[learn_index] = value_map_update
            loss = loss_detector + loss_descriptor

            return loss, number_pts, loss_detector.cpu().data.sum(), \
                   loss_descriptor.cpu().data.sum(), enhanced_label_pts, \
                   enhanced_label, detector_pred, loss_detector_num, loss_descriptor_num

        return detector_pred, descriptor_pred
