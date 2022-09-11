# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.ops import batched_nms
from mmcv.runner import BaseModule, force_fp32
from mmdet.core import reduce_mean
from ..builder import HEADS, build_loss
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
import torch.nn as nn
from functools import partial

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map_results)

def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone().detach()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1
    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), img_poly.size(1), cnn_feature.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 2, 0)
        gcn_feature[ind == i] = feature
    return gcn_feature

def interpolation(poly, time=10):
    ori_points_num = poly.size(1)
    poly_roll =torch.roll(poly, shifts=1, dims=1)
    poly_ = poly.unsqueeze(3).repeat(1, 1, 1, time)
    poly_roll = poly_roll.unsqueeze(3).repeat(1, 1, 1, time)
    step = torch.arange(0, time, dtype=torch.float32).cuda() / time
    poly_interpolation = poly_ * step + poly_roll * (1. - step)
    poly_interpolation = poly_interpolation.permute(0, 1, 3, 2).reshape(poly_interpolation.size(0), ori_points_num * time, 2)
    return poly_interpolation

class PointResampler:
    def __init__(self, mode='uniform', sample_ratio=1, align_num=None, density=10):
        assert mode in ['uniform', 'align_uniform']
        if mode == 'align_uniform':
            assert align_num is not None
        self.mode = mode
        self.align_num = align_num
        self.sample_ratio = sample_ratio
        self.density = density

    def circumference(self, rings):
        # rings torch.Tensor(..., P, 2)
        lengths = torch.sum((torch.roll(rings, 1, dims=-2) - rings) ** 2, dim=-1) ** 0.5
        return lengths.sum(dim=-2)

    def get_sampled_idxs(self, cum_lengths, circumference, sampled_nums):
        return 

    def __call__(self, polys):
        # polys, torch.Tensor(N, P, 2)
        return


@HEADS.register_module()
class BaseContourProposalHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self,
                 in_channel=256,
                 hidden_dim=256,
                 point_nums=128,
                 global_deform_stride=10.0,
                 init_stride=10.0,
                 loss_init=dict(
                     type='SmoothL1Loss',
                     beta=0.1,
                     loss_weight=0.2),
                 loss_contour=dict(
                     type='SmoothL1Loss',
                     beta=0.1,
                     loss_weight=0.1),
                 loss_contour_mask=None,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 ):
        super(BaseContourProposalHead, self).__init__(init_cfg)
        self.point_nums = point_nums
        self.init_stride = init_stride
        self.global_deform_stride = global_deform_stride
        #init component
        self.fc1 = nn.Conv2d(in_channel, hidden_dim, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_dim, point_nums * 2, kernel_size=1, stride=1, padding=0, bias=True)
        #global refine component
        self.fc3 = nn.Conv2d(in_channel, hidden_dim, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc4 = nn.Conv2d(hidden_dim, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.linear1 = torch.nn.Linear(in_features=((point_nums + 1) * 64),
                                       out_features=point_nums * 4, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Linear(in_features=point_nums * 4,
                                       out_features=point_nums * 2, bias=True)

        self.loss_contour = build_loss(loss_contour)
        self.loss_init = build_loss(loss_init)
        if loss_contour_mask is None:
            self.loss_contour_mask = None
        else:
            self.loss_contour_mask = build_loss(loss_contour_mask)

    def init_fc(self, fc):
        if isinstance(fc, nn.Conv2d):
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0)

    def init_weights(self):
        super(BaseContourProposalHead, self).init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)
        self.init_fc(self.fc1)
        self.init_fc(self.fc2)
        self.init_fc(self.fc3)
        self.init_fc(self.fc4)

    def forward(self, x, centers, img_h, img_w, inds, use_fpn_level=0):
        x = x[use_fpn_level]
        #init_contour proposal
        normed_shape_embed = self.fc1(x)
        normed_shape_embed = self.relu1(normed_shape_embed)
        normed_shape_embed = self.fc2(normed_shape_embed)
        normed_instance_shape_embed = get_gcn_feature(normed_shape_embed, centers.unsqueeze(1),
                                                      inds, img_h, img_w).squeeze(1)
        normed_instance_shape_embed = normed_instance_shape_embed.reshape(centers.size(0), self.point_nums, 2)
        instance_shape_embed = normed_instance_shape_embed * self.init_stride
        contour_proposals = instance_shape_embed + centers.unsqueeze(1).repeat(1, normed_instance_shape_embed.size(1), 1)

        #coarse contour proposal
        global_deform_feature = self.fc3(x)
        global_deform_feature = self.relu2(global_deform_feature)
        global_deform_feature = self.fc4(global_deform_feature)
        contour_proposals_with_center = torch.cat([contour_proposals, centers.unsqueeze(1)], dim=1)
        instance_global_deform_features = get_gcn_feature(global_deform_feature, contour_proposals_with_center,
                                                          inds, img_h, img_w).flatten(1)
        instance_global_deform_features = self.linear1(instance_global_deform_features)
        instance_global_deform_features = self.relu3(instance_global_deform_features)
        normed_instance_global_offset = self.linear2(instance_global_deform_features).reshape(centers.size(0),
                                                                                              self.point_nums, 2)
        instance_global_offset = normed_instance_global_offset * self.global_deform_stride
        coarse_contour = instance_global_offset + contour_proposals.detach()

        return contour_proposals, coarse_contour, normed_instance_shape_embed, normed_instance_global_offset

    def compute_contour_losses(self, normed_init_offset_pred, normed_global_offset_pred,
                               normed_init_offset_target, normed_global_offset_target):
        num_poly = torch.tensor(
            len(normed_init_offset_target), dtype=torch.float, device=normed_init_offset_pred.device)
        num_poly = max(reduce_mean(num_poly), 1.0)
        loss_init = self.loss_init(normed_init_offset_pred, normed_init_offset_target,
                                   avg_factor=num_poly * self.point_nums * 2)
        loss_coarse = self.loss_contour(normed_global_offset_pred, normed_global_offset_target,
                                        avg_factor=num_poly * self.point_nums * 2)
        return dict(loss_init_contour=loss_init,
                    loss_coarse_contour=loss_coarse)

    def compute_contour_mask_losses(self, polys, gt_masks, gt_bboxes):
        ret = dict()
        num_mask = torch.tensor(
            len(gt_bboxes), dtype=torch.float, device=gt_bboxes.device)
        num_mask = max(reduce_mean(num_mask), 1.0)
        for i, poly in enumerate(polys):
            ret.update({'proposal_loss_mask_{}'.format(i): self.loss_contour_mask(poly,
                                                                                  gt_masks,
                                                                                  gt_bboxes,
                                                                                  avg_factor=num_mask)})
        return ret

    def loss(self, normed_init_offset_pred, normed_global_offset_pred,
             normed_init_offset_target, normed_global_offset_target,
             contour_proposals, contour_coarse, gt_bboxes=None,
             gt_masks=None, is_single_component=None):
        """Compute losses of the head."""
        ret = dict()
        if is_single_component is not None:
            normed_init_offset_pred = normed_init_offset_pred[is_single_component]
            normed_global_offset_pred = normed_global_offset_pred[is_single_component]
        ret.update(self.compute_contour_losses(normed_init_offset_pred, normed_global_offset_pred,
                                               normed_init_offset_target, normed_global_offset_target))
        if self.loss_contour_mask is not None:
            ret.update(self.compute_contour_mask_losses([contour_proposals, contour_coarse],
                                                        gt_masks, gt_bboxes))
        return ret

    def get_targets(self, gt_contours, gt_centers, contour_proposals, is_single_component=None):
        if is_single_component is not None:
            gt_centers = gt_centers[is_single_component]
            contour_proposals = contour_proposals[is_single_component]
        gt_centers = gt_centers.unsqueeze(1).repeat(1, self.point_nums, 1)
        normed_init_offset_target = (gt_contours - gt_centers) / self.init_stride
        normed_global_offset_target = (gt_contours - contour_proposals) / self.global_deform_stride
        return normed_init_offset_target.detach(), normed_global_offset_target.detach()

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_contours,
                      gt_masks=None,
                      is_single_component=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_contours (Tensor): Ground truth contours,
                shape (points_nums, 2).
        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        img_h, img_w = img_metas[0]['batch_input_shape']
        inds = torch.cat([torch.full([len(gt_bboxes[i])], i) for i in range(len(gt_bboxes))], dim=0).to(x[0].device)
        gt_bboxes = torch.cat(gt_bboxes, dim=0)
        gt_contours = torch.cat(gt_contours, dim=0)
        if is_single_component is not None:
            is_single_component = torch.cat(is_single_component, dim=0)
            is_single_component = is_single_component.to(torch.bool)
        gt_centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:4]) / 2.
        contour_proposals, coarse_contour, normed_init_offset, normed_global_offset = self(x, gt_centers,
                                                                                           img_h, img_w, inds)
        normed_init_offset_target, normed_global_offset_target = self.get_targets(gt_contours, gt_centers,
                                                                                  contour_proposals,
                                                                                  is_single_component)
        losses = self.loss(normed_init_offset, normed_global_offset,
                           normed_init_offset_target, normed_global_offset_target,
                           contour_proposals, coarse_contour, gt_bboxes,
                           gt_masks, is_single_component)
        return losses, coarse_contour, inds

    def convert_single_imagebboxes2featurebboxes(self, bboxes_, img_meta):
        bboxes = bboxes_.clone()
        img_shape = img_meta['img_shape'][:2]
        ori_shape = img_meta['ori_shape'][:2]
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / ori_shape[0] * img_shape[0]
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / ori_shape[0] * img_shape[0]
        return bboxes

    def convert_imagebboxes2featurebboxes(self, bboxes, img_metas):
        return multi_apply(self.convert_single_imagebboxes2featurebboxes, bboxes, img_metas)

    def simple_test(self, feats, img_metas, pred_bboxes):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        img_h, img_w = img_metas[0]['batch_input_shape']
        inds = torch.cat([torch.full([len(pred_bboxes[i])], i) for i in range(len(pred_bboxes))], dim=0).to(feats[0].device)
        pred_bboxes = self.convert_imagebboxes2featurebboxes(pred_bboxes, img_metas)

        pred_bboxes = torch.cat(pred_bboxes, dim=0)
        pred_centers = (pred_bboxes[..., :2] + pred_bboxes[..., 2:4]) / 2.

        contour_proposals, coarse_contour, normed_init_offset, normed_global_offset = self(feats, pred_centers, img_h,
                                                                                           img_w, inds)
        return coarse_contour, inds

@HEADS.register_module()
class FPNContourProposalHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self,
                 in_channel=256,
                 hidden_dim=256,
                 start_level=0,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, 1e6)),
                 strides=[8, 16, 32, 64, 128],
                 point_nums=128,
                 use_tanh=[True, True],
                 loss_init=dict(
                     type='SmoothL1Loss',
                     beta=0.1,
                     loss_weight=0.2),
                 loss_contour=dict(
                     type='SmoothL1Loss',
                     beta=0.1,
                     loss_weight=0.1),
                 loss_contour_mask=None,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(FPNContourProposalHead, self).__init__(init_cfg)
        self.point_nums = point_nums
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.use_tanh = use_tanh
        self.start_level = start_level
        #init component
        self.init_predictor = nn.ModuleList(nn.Linear(in_channel, hidden_dim, bias=True),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, point_nums * 2, bias=False))
        #global refine component
        self.global_offset_predictor = nn.ModuleList(nn.Linear(in_channel * (point_nums + 1), hidden_dim * 2, bias=True),
                                                     nn.ReLU(inplace=True),
                                                     nn.Linear(hidden_dim * 2, hidden_dim, bias=True),
                                                     nn.ReLU(inplace=True),
                                                     nn.Linear(hidden_dim, point_nums * 2, bias=False))
        self.loss_contour = build_loss(loss_contour)
        self.loss_init = build_loss(loss_init)
        if loss_contour_mask is None:
            self.loss_contour_mask = None
        else:
            self.loss_contour_mask = build_loss(loss_contour_mask)

    def init_weights(self):
        super(BaseContourProposalHead, self).init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    def extract_features_single(self, feat, points, inds, img_h, img_w):
        points_feat = get_gcn_feature(feat, points, inds, img_h, img_w)
        return points_feat

    def extract_features(self, ms_feats, points, img_h, img_w, img_inds, fl_inds):
        num_points = points.size(0)
        ms_points = []
        ms_img_inds = []
        for i in range(len(ms_feats)):
            ms_points.append(points[fl_inds == i])
            ms_img_inds.append(img_inds[fl_inds == i])
        points_features = torch.zeros([num_points, points.size(1),
                                       ms_feats[0].size(1)]).to(ms_feats[0].device)
        ms_points_features = multi_apply(self.extract_features_single, ms_feats, ms_points,
                                         ms_img_inds, img_h=img_h, img_w=img_w)
        for i in range(len(ms_feats)):
            points_features[fl_inds == i] = ms_points_features[i]
        return points_features

    def forward(self, ms_feats, gt_centers, gt_whs, img_h, img_w, inds):
        num_instances = gt_centers.size(0)
        gt_max_lengths = torch.max(gt_whs, dim=-1, keepdim=True)[0]
        regress_ranges = torch.zeros((num_instances, len(self.regress_ranges), 2),
                                     dtype=torch.int64, device=gt_centers.device)
        for i, regress_range in enumerate(self.regress_ranges):
            regress_ranges[:, i, 0] = regress_range[0]
            regress_ranges[:, i, 1] = regress_range[1]
        ms_inds = torch.arange(0, len(ms_feats), device=gt_centers.device,
                               dtype=torch.int64).unsqueeze(0)
        ms_inds = torch.logical_and(regress_ranges[..., 0] < gt_max_lengths,
                                    regress_ranges[..., 1] >= gt_max_lengths).astype(torch.int64) * ms_inds
        ms_inds = torch.sum(ms_inds, dim=1)
        strides = torch.Tensor(self.strides, device=gt_centers.device)[ms_inds]

        centers_features = self.extract_features(ms_feats, gt_centers.unsqueeze(1),
                                                 img_h, img_w, inds, ms_inds).squeeze(1)
        shape_embed = self.init_predictor(centers_features).reshape(num_instances, self.point_nums, 2)
        if self.use_tanh[0]:
            shape_embed = shape_embed.tanh()
            contours_proposal = gt_centers.detach().unsqueeze(1) + shape_embed * gt_whs.unsqueeze(1)
        else:
            contours_proposal = gt_centers.detach().unsqueeze(1) + shape_embed * strides

        contour_proposals_with_center = torch.cat([contours_proposal, gt_centers.unsqueeze(1)], dim=1)
        contour_proposals_with_center_features = self.extract_features(ms_feats, contour_proposals_with_center,
                                                                       img_h, img_w, inds, ms_inds)
        normed_coarse_offsets = self.global_offset_predictor(contour_proposals_with_center_features.flatten(1))
        normed_coarse_offsets = normed_coarse_offsets.reshape(num_instances, self.point_nums, 2)
        if self.use_tanh[1]:
            normed_coarse_offsets = normed_coarse_offsets.tanh()
            coarse_contours = contours_proposal.detach() + normed_coarse_offsets * gt_whs.unsqueeze(1)
        else:
            coarse_contours = contours_proposal.detach() + normed_coarse_offsets * strides
        return contours_proposal, coarse_contours, shape_embed, normed_coarse_offsets, strides

    def compute_contour_losses(self, normed_init_offset_pred, normed_global_offset_pred,
                               normed_init_offset_target, normed_global_offset_target):
        num_poly = torch.tensor(
            len(normed_init_offset_target), dtype=torch.float, device=normed_init_offset_pred.device)
        num_poly = max(reduce_mean(num_poly), 1.0)
        loss_init = self.loss_init(normed_init_offset_pred, normed_init_offset_target,
                                   avg_factor=num_poly * self.point_nums * 2)
        loss_coarse = self.loss_contour(normed_global_offset_pred, normed_global_offset_target,
                                        avg_factor=num_poly * self.point_nums * 2)
        return dict(loss_init_contour=loss_init,
                    loss_coarse_contour=loss_coarse)

    def compute_contour_mask_losses(self, polys, gt_masks, gt_bboxes):
        ret = dict()
        num_mask = torch.tensor(
            len(gt_bboxes), dtype=torch.float, device=gt_bboxes.device)
        num_mask = max(reduce_mean(num_mask), 1.0)
        for i, poly in enumerate(polys):
            ret.update({'proposal_loss_mask_{}'.format(i): self.loss_contour_mask(poly,
                                                                                  gt_masks,
                                                                                  gt_bboxes,
                                                                                  avg_factor=num_mask)})
        return ret

    def loss(self, normed_init_offset_pred, normed_global_offset_pred,
             normed_init_offset_target, normed_global_offset_target,
             contour_proposals, contour_coarse, gt_bboxes=None,
             gt_masks=None, is_single_component=None):
        """Compute losses of the head."""
        ret = dict()
        if is_single_component is not None:
            normed_init_offset_pred = normed_init_offset_pred[is_single_component]
            normed_global_offset_pred = normed_global_offset_pred[is_single_component]
        ret.update(self.compute_contour_losses(normed_init_offset_pred, normed_global_offset_pred,
                                               normed_init_offset_target, normed_global_offset_target))
        if self.loss_contour_mask is not None:
            ret.update(self.compute_contour_mask_losses([contour_proposals, contour_coarse],
                                                        gt_masks, gt_bboxes))
        return ret

    def get_targets(self, gt_contours, gt_centers, gt_whs, contour_proposals, strides, is_single_component=None):
        if is_single_component is not None:
            gt_centers = gt_centers[is_single_component]
            contour_proposals = contour_proposals[is_single_component]
            gt_whs = gt_whs[is_single_component]
        gt_centers = gt_centers.unsqueeze(1).repeat(1, self.point_nums, 1)
        if self.use_tanh[0]:
            normed_init_offset_target = (gt_contours - gt_centers) / gt_whs.unsqueeze(1)
        else:
            normed_init_offset_target = (gt_contours - gt_centers) / strides
        if self.use_tanh[1]:
            normed_global_offset_target = (gt_contours - contour_proposals) / gt_whs.unsqueeze(1)
        else:
            normed_global_offset_target = (gt_contours - contour_proposals) / strides
        return normed_init_offset_target.detach(), normed_global_offset_target.detach()

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_contours,
                      gt_masks=None,
                      is_single_component=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_contours (Tensor): Ground truth contours,
                shape (points_nums, 2).
        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        x = x[self.start_level:]
        img_h, img_w = img_metas[0]['batch_input_shape']
        inds = torch.cat([torch.full([len(gt_bboxes[i])], i) for i in range(len(gt_bboxes))], dim=0).to(x[0].device)
        gt_bboxes = torch.cat(gt_bboxes, dim=0)
        gt_contours = torch.cat(gt_contours, dim=0)
        if is_single_component is not None:
            is_single_component = torch.cat(is_single_component, dim=0)
            is_single_component = is_single_component.to(torch.bool)
        gt_centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:4]) / 2.
        gt_whs = (gt_bboxes[..., 2:4] - gt_bboxes[..., :2]) / 2.
        contour_proposals, coarse_contour, normed_init_offset, normed_global_offset, strides =\
            self(x, gt_centers, gt_whs, img_h, img_w, inds)
        normed_init_offset_target, normed_global_offset_target = self.get_targets(gt_contours, gt_centers, gt_whs,
                                                                                  contour_proposals, strides,
                                                                                  is_single_component)
        losses = self.loss(normed_init_offset, normed_global_offset,
                           normed_init_offset_target, normed_global_offset_target,
                           contour_proposals, coarse_contour, gt_bboxes,
                           gt_masks, is_single_component)
        return losses, coarse_contour, inds

    def convert_single_imagebboxes2featurebboxes(self, bboxes_, img_meta):
        bboxes = bboxes_.clone()
        img_shape = img_meta['img_shape'][:2]
        ori_shape = img_meta['ori_shape'][:2]
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / ori_shape[0] * img_shape[0]
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / ori_shape[0] * img_shape[0]
        return bboxes

    def convert_imagebboxes2featurebboxes(self, bboxes, img_metas):
        return multi_apply(self.convert_single_imagebboxes2featurebboxes, bboxes, img_metas)

    def simple_test(self, feats, img_metas, pred_bboxes):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        feats = feats[self.start_level:]
        img_h, img_w = img_metas[0]['batch_input_shape']
        inds = torch.cat([torch.full([len(pred_bboxes[i])], i) for i in range(len(pred_bboxes))], dim=0).to(feats[0].device)
        pred_bboxes = self.convert_imagebboxes2featurebboxes(pred_bboxes, img_metas)

        pred_bboxes = torch.cat(pred_bboxes, dim=0)
        pred_centers = (pred_bboxes[..., :2] + pred_bboxes[..., 2:4]) / 2.
        pred_whs = (pred_bboxes[..., 2:4] - pred_bboxes[..., :2]) / 2.

        contour_proposals, coarse_contour, normed_init_offset, normed_global_offset, strides =\
            self(feats, pred_centers, pred_whs, img_h, img_w, inds)
        return coarse_contour, inds

@HEADS.register_module()
class BaseContourEvolveHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self,
                 in_channel=256,
                 point_nums=128,
                 evolve_deform_stride=4.,
                 iter_num=3,
                 loss_contour=dict(
                     type='SmoothL1Loss',
                     beta=0.25,
                     loss_weight=1.0),
                 loss_contour_mask=None,
                 loss_last_evolve=None,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 ):
        super(BaseContourEvolveHead, self).__init__(init_cfg)
        self.point_nums = point_nums
        self.evolve_deform_stride = evolve_deform_stride
        self.iter_num = iter_num
        # evolve component
        for i in range(iter_num):
            evolve_gcn = Snake(state_dim=128, feature_dim=in_channel)
            self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        self.loss_contour = build_loss(loss_contour)
        if loss_last_evolve is not None:
            self.loss_last_evolve = build_loss(loss_last_evolve)
        else:
            self.loss_last_evolve = None
        if loss_contour_mask is not None:
            self.loss_contour_mask = build_loss(loss_contour_mask)
        else:
            self.loss_contour_mask = None

    def init_weights(self):
        super(BaseContourEvolveHead, self).init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, contour_proposals, img_h, img_w, inds, use_fpn_level=0):
        x = x[use_fpn_level]
        outputs_contours = [contour_proposals]
        normed_offsets = []
        # evolve contour
        for i in range(self.iter_num):
            py_in = outputs_contours[-1]
            py_features = get_gcn_feature(x, py_in, inds, img_h, img_w).permute(0, 2, 1)
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            normed_offset = evolve_gcn(py_features).permute(0, 2, 1)
            offset = normed_offset * self.evolve_deform_stride
            py_out = py_in.detach() + offset
            outputs_contours.append(py_out)
            normed_offsets.append(normed_offset)
        assert len(outputs_contours) == len(normed_offsets) + 1
        return outputs_contours, normed_offsets

    def loss(self, normed_offsets_preds, normed_offsets_targets, is_single_component=None):
        """Compute losses of the head."""
        ret = dict()
        num_poly = torch.tensor(
            len(normed_offsets_targets[0]), dtype=torch.float, device=normed_offsets_preds[0].device)
        num_poly = max(reduce_mean(num_poly), 1.0)
        for i, (offsets_preds, offsets_targets) in enumerate(zip(normed_offsets_preds, normed_offsets_targets)):
            if is_single_component is not None:
                offsets_preds = offsets_preds[is_single_component]
            loss = self.loss_contour(offsets_preds, offsets_targets, avg_factor=num_poly * self.point_nums * 2)
            ret.update({'evolve_loss_' + str(i): loss})
        return ret

    def loss_last(self, pred_contour, normed_pred_offsets, target_contour, key_points, key_points_mask, is_single_component=None):
        ret = dict()
        num_poly = torch.tensor(
            len(pred_contour), dtype=torch.float, device=pred_contour.device)
        num_poly = max(reduce_mean(num_poly), 1.0)
        num_key_points = torch.sum(key_points_mask).to(torch.float)
        num_key_points = max(reduce_mean(num_key_points), 1.0)
        avg_factor = (num_poly * self.point_nums * 2, num_key_points * 2)
        loss = self.loss_last_evolve(pred_contour[is_single_component], normed_pred_offsets[is_single_component], target_contour,
                                     key_points, key_points_mask, avg_factor=avg_factor)
        ret.update({'evolve_loss_last': loss})
        return ret

    def compute_loss_contour_mask(self, polys, gt_masks, gt_bboxes):
        ret = dict()
        num_mask = torch.tensor(
            len(gt_bboxes), dtype=torch.float, device=gt_bboxes.device)
        num_mask = max(reduce_mean(num_mask), 1.0)
        for i, poly in enumerate(polys):
            ret.update({'evolve_loss_mask_{}'.format(i): \
                        self.loss_contour_mask(poly, gt_masks, gt_bboxes, avg_factor=num_mask)})
        return ret

    def get_targets(self, py_in, gt_contours, is_single_component=None):
        if is_single_component is not None:
            py_in = py_in[is_single_component]
        normed_offset_target = (gt_contours - py_in) / self.evolve_deform_stride
        return normed_offset_target.detach()

    def forward_train(self,
                      x,
                      img_metas,
                      contour_proposals,
                      gt_contours,
                      inds,
                      key_points=None,
                      key_points_masks=None,
                      gt_bboxes=None,
                      gt_masks=None,
                      is_single_component=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_contours (Tensor): Ground truth contours,
                shape (points_nums, 2).
        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        img_h, img_w = img_metas[0]['batch_input_shape']
        gt_contours = torch.cat(gt_contours, dim=0)
        gt_bboxes = torch.cat(gt_bboxes, dim=0)
        if is_single_component is not None:
            is_single_component = torch.cat(is_single_component, dim=0)
            is_single_component = is_single_component.to(torch.bool)
        output_contours, normed_offsets = self(x, contour_proposals, img_h, img_w, inds)
        normed_offsets_targets = []
        for i in range(len(normed_offsets)):
            normed_offset_target = self.get_targets(output_contours[i], gt_contours, is_single_component)
            normed_offsets_targets.append(normed_offset_target)
        if self.loss_last_evolve is None:
            losses = self.loss(normed_offsets, normed_offsets_targets, is_single_component)
            if self.loss_contour_mask is not None:
                losses.update(self.compute_loss_contour_mask(output_contours[1:],
                                                             gt_masks, gt_bboxes))
        else:
            key_points = torch.cat(key_points, dim=0)
            key_points_masks = torch.cat(key_points_masks, dim=0)
            losses = self.loss(normed_offsets[:-1], normed_offsets_targets[:-1], is_single_component)
            losses.update(self.loss_last(output_contours[len(normed_offsets) - 1],
                                         normed_offsets[-1], gt_contours, key_points,
                                         key_points_masks, is_single_component))
            if self.loss_contour_mask is not None:
                losses.update(self.compute_loss_contour_mask(output_contours[1:],
                                                             gt_masks, gt_bboxes))
        return losses

    def simple_test(self,
                    x,
                    img_metas,
                    contour_proposals,
                    inds,
                    ret_stage=-1):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        img_h, img_w = img_metas[0]['batch_input_shape']
        output_contours, normed_offsets = self(x, contour_proposals, img_h, img_w, inds)
        output_contour = output_contours[ret_stage]
        ret = []
        for i in range(len(img_metas)):
            ret.append(output_contour[inds == i])
        return ret

class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj * 2 + 1)

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat([input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        return self.fc(input)


class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj * 2 + 1, dilation=self.dilation)

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat(
                [input[..., -self.n_adj * self.dilation:], input, input[..., :self.n_adj * self.dilation]], dim=2)
        return self.fc(input)

_conv_factory = {
    'grid': CircConv,
    'dgrid': DilatedCircConv
}

class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1):
        super(BasicBlock, self).__init__()
        if conv_type == 'grid':
            self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj)
        else:
            self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x


class Snake(nn.Module):
    def __init__(self, state_dim, feature_dim):
        super(Snake, self).__init__()
        if state_dim != feature_dim:
            self.head = self.head = BasicBlock(feature_dim, state_dim, 'dgrid')
        else:
            self.head = None
        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        n_adj = 4
        for i in range(self.res_layer_num):
            if dilation[i] == 0:
                conv_type = 'grid'
            else:
                conv_type = 'dgrid'
            conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=n_adj, dilation=dilation[i])
            self.__setattr__('res' + str(i), conv)

        fusion_state_dim = 256

        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)

        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )

    def forward(self, x):
        if self.head is not None:
            x = self.head(x)
        states = [x]
        for i in range(self.res_layer_num):
            x = self.__getattr__('res' + str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)

        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)

        x = self.prediction(state)

        return x
