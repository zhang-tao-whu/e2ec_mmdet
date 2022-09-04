# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import warnings
import numpy as np
import torch

from mmdet.core import bbox2result
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from ..builder import DETECTORS, build_backbone, build_head, build_neck
import pycocotools.mask as maskUtils
from functools import partial
import time

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

@DETECTORS.register_module()
class ContourBasedInstanceSegmentor(SingleStageDetector):
    """Base class for contour based segmentor.

        Single-stage segmentor directly and densely predict instance contours on the
        output features of the backbone+neck.
        """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 contour_proposal_head=None,
                 contour_evolve_head=None,
                 detector_fpn_start_level=1,
                 contour_fpn_start_level=1,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(ContourBasedInstanceSegmentor, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                            test_cfg, pretrained, init_cfg)
        contour_proposal_head.update(train_cfg=train_cfg)
        contour_proposal_head.update(test_cfg=test_cfg)
        contour_evolve_head.update(train_cfg=train_cfg)
        contour_evolve_head.update(test_cfg=test_cfg)
        self.contour_proposal_head = build_head(contour_proposal_head)
        self.contour_evolve_head = build_head(contour_evolve_head)
        self.detector_fpn_start_level = detector_fpn_start_level
        self.contour_fpn_start_level = contour_fpn_start_level

    # def forward_dummy(self, img):
    #     """Used for computing network flops.
    #
    #     See `mmdetection/tools/analysis_tools/get_flops.py`
    #     """
    #     x = self.extract_feat(img)
    #     outs = self.bbox_head(x)
    #     return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_polys,
                      gt_masks=None,
                      key_points=None,
                      key_points_masks=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        gt_contours = gt_polys
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x[self.detector_fpn_start_level:], img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        losses_contour_proposal, contour_proposals, inds = \
            self.contour_proposal_head.forward_train(x[self.contour_fpn_start_level:],
                                                       img_metas, gt_bboxes, gt_contours)
        losses_contour_evolve = self.contour_evolve_head.forward_train(x[self.contour_fpn_start_level:],
                                                                       img_metas, contour_proposals,
                                                                       gt_contours, inds, key_points,
                                                                       key_points_masks)
        losses.update(losses_contour_proposal)
        losses.update(losses_contour_evolve)
        return losses

    def simple_test(self, img, img_metas, rescale=False, print_consumed_time=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        now_time = time.time()
        time_dict = {}
        feat = self.extract_feat(img)
        time_dict['backbone'] = now_time - time.time()
        now_time = time.time()
        results_list = self.bbox_head.simple_test(
            feat[self.detector_fpn_start_level:], img_metas, rescale=rescale)
        time_dict['detector'] = now_time - time.time()
        now_time = time.time()
        #results_list [(bboxes, labels), ...]
        # boxes (Tensor): Bboxes with score after nms, has shape (num_bboxes, 5). last dimension 5 arrange as (x1, y1, x2, y2, score)
        # labels (Tensor): has shape (num_bboxes, )
        bboxes_pred = [item[0] for item in results_list]
        labels_pred = [item[1] for item in results_list]
        contour_proposals, inds = self.contour_proposal_head.simple_test(feat[self.contour_fpn_start_level:], img_metas, bboxes_pred)
        time_dict['contour_proposal'] = now_time - time.time()
        now_time = time.time()
        contours_pred = self.contour_evolve_head.simple_test(feat[self.contour_fpn_start_level:], img_metas, contour_proposals, inds)
        time_dict['contour_evolve'] = now_time - time.time()
        now_time = time.time()
        mask_results = self.convert_contour2mask(contours_pred, labels_pred, bboxes_pred, img_metas)
        bboxes_pred = [item[1] for item in mask_results]
        labels_pred = [item[2] for item in mask_results]
        mask_results = [item[0] for item in mask_results]
        time_dict['post_contour2mask'] = now_time - time.time()
        now_time = time.time()
        results_list = list(zip(bboxes_pred, labels_pred))
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        whole_time = 0
        for key in time_dict.keys():
            whole_time += time_dict[key]
        time_dict['whole'] = whole_time
        for key in time_dict.keys():
            time_dict[key] /= time_dict['whole']
        if print_consumed_time:
            print(' ')
            print(time_dict)
        return list(zip(bbox_results, mask_results))

    def converge_components_single(self, contours_pred, labels_pred, bboxes_pred,
                                   bboxes_from='detection', threthold=0.9):
        assert bboxes_from in ['detection', 'contour']
        if bboxes_from == 'contour':
            min_coords = torch.min(contours_pred, dim=1)[0]
            max_coords = torch.max(contours_pred, dim=1)[0]
            bboxes_pred = torch.cat([min_coords, max_coords], dim=1)
        iof = bbox_overlaps(bboxes_pred, bboxes_pred, is_aligned=False, mode='iof')
        same_label = labels_pred.unsqueeze(1) - labels_pred.unsqueeze(0)
        same_label = (same_label == 0).to(torch.float)
        iof = iof * same_label
        npred = iof.size(0)
        # iof (n, n)
        component_rela = torch.range(npred, device=iof.device)
        iof[torch.range(npred), torch.range(npred)] = 0
        max_iof, max_inds = torch.max(iof, dim=1)
        replace = max_iof >= threthold
        component_rela[replace] = max_inds[replace]
        valid_idxs = component_rela[torch.logical_not(replace)]
        return (valid_idxs, component_rela)

    def converge_components(self, contours_pred, laels_pred, bboxes_pred, bboxes_from='detection', threthold=0.9):
        return multi_apply(self.converge_components_single, contours_pred, laels_pred, bboxes_pred,
                           bboxes_from=bboxes_from, threthold=threthold)

    def single_convert_contour2mask(self, contours_pred, labels_pred, bboxes_pred,
                                    img_meta, rescore=True, converge_component=True,
                                    ignore_contour2mask=False, iou_threthold=0.0):
        img_shape = img_meta['img_shape'][:2]
        ori_shape = img_meta['ori_shape'][:2]
        mask_pred = [[] for _ in range(self.bbox_head.num_classes)]
        contours_pred[..., 0] = contours_pred[..., 0] / img_shape[0] * ori_shape[0]
        contours_pred[..., 1] = contours_pred[..., 1] / img_shape[1] * ori_shape[1]
        if rescore:
            scores_pred = bboxes_pred[..., 4]
            contours_maxp = torch.max(contours_pred, dim=1)[0]
            contours_minp = torch.min(contours_pred, dim=1)[0]
            contours_bboxes = torch.cat([contours_minp, contours_maxp], dim=-1)
            bboxes = bboxes_pred[..., :4]
            ious = bbox_overlaps(bboxes, contours_bboxes, is_aligned=True)
            ious = (ious - iou_threthold) / (1 - iou_threthold)
            ious = torch.clamp(ious, 0, 1)
            scores_pred = (ious * scores_pred) ** 0.5
            bboxes_pred[..., 4] *= 0
            bboxes_pred[..., 4] += scores_pred
        contours_pred = contours_pred.detach().cpu().numpy()
        labels_pred = labels_pred.detach().cpu().numpy()
        if ignore_contour2mask:
            return (mask_pred, bboxes_pred, labels_pred)
        if converge_component:
            valid_idxs, comp_rela = self.converge_components_single(contours_pred, labels_pred,
                                                                    bboxes_pred, bboxes_from='detection',
                                                                    threthold=0.9)
        rles = []
        if not converge_component:
            for contour in contours_pred:
                contour = contour.flatten().tolist()
                rle = maskUtils.frPyObjects([contour], ori_shape[0], ori_shape[1])
                rles += rle
            masks = maskUtils.decode(rles).transpose(2, 0, 1)
        else:
            for idx in valid_idxs:
                contours = []
                for comp in contours_pred[comp_rela == idx]:
                    contours.append(comp.flatten().tolist())
                rle = maskUtils.frPyObjects(contours, ori_shape[0], ori_shape[1])
                rles += maskUtils.merge(rle)
            masks = maskUtils.decode(rles).transpose(2, 0, 1)
            labels_pred = labels_pred[valid_idxs]
            bboxes_pred = bboxes_pred[valid_idxs]
        for mask, label in zip(masks, labels_pred):
            mask_pred[int(label)].append(mask)
        return (mask_pred, bboxes_pred, labels_pred)

    def convert_contour2mask(self, contours_preds, labels_preds, bboxes_pred, img_metas,
                             rescore=True, converge_component=True, ignore_contour2mask=False):
        #masks_pred [single img masks_pred]
        #single img masks_pred [single class instances mask]
        #instance mask (h, w)
        return multi_apply(self.single_convert_contour2mask,
                           contours_preds, labels_preds, bboxes_pred,
                           img_metas, rescore=rescore, converge_component=converge_component,
                           ignore_contour2mask=ignore_contour2mask)

    # def aug_test(self, imgs, img_metas, rescale=False):
    #     """Test function with test time augmentation.
    #
    #     Args:
    #         imgs (list[Tensor]): the outer list indicates test-time
    #             augmentations and inner Tensor should have a shape NxCxHxW,
    #             which contains all images in the batch.
    #         img_metas (list[list[dict]]): the outer list indicates test-time
    #             augs (multiscale, flip, etc.) and the inner list indicates
    #             images in a batch. each dict has image information.
    #         rescale (bool, optional): Whether to rescale the results.
    #             Defaults to False.
    #
    #     Returns:
    #         list[list[np.ndarray]]: BBox results of each image and classes.
    #             The outer list corresponds to each image. The inner list
    #             corresponds to each class.
    #     """
    #     assert hasattr(self.bbox_head, 'aug_test'), \
    #         f'{self.bbox_head.__class__.__name__}' \
    #         ' does not support test-time augmentation'
    #
    #     feats = self.extract_feats(imgs)
    #     results_list = self.bbox_head.aug_test(
    #         feats, img_metas, rescale=rescale)
    #     bbox_results = [
    #         bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         for det_bboxes, det_labels in results_list
    #     ]
    #     return bbox_results

    # def onnx_export(self, img, img_metas, with_nms=True):
    #     """Test function without test time augmentation.
    #
    #     Args:
    #         img (torch.Tensor): input images.
    #         img_metas (list[dict]): List of image information.
    #
    #     Returns:
    #         tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
    #             and class labels of shape [N, num_det].
    #     """
    #     x = self.extract_feat(img)
    #     outs = self.bbox_head(x)
    #     # get origin input shape to support onnx dynamic shape
    #
    #     # get shape as tensor
    #     img_shape = torch._shape_as_tensor(img)[2:]
    #     img_metas[0]['img_shape_for_onnx'] = img_shape
    #     # get pad input shape to support onnx dynamic shape for exporting
    #     # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
    #     # for inference
    #     img_metas[0]['pad_shape_for_onnx'] = img_shape
    #
    #     if len(outs) == 2:
    #         # add dummy score_factor
    #         outs = (*outs, None)
    #     # TODO Can we change to `get_bboxes` when `onnx_export` fail
    #     det_bboxes, det_labels = self.bbox_head.onnx_export(
    #         *outs, img_metas, with_nms=with_nms)
    #
    #     return det_bboxes, det_labels

@DETECTORS.register_module()
class E2EC(ContourBasedInstanceSegmentor):
    """Implementation of `E2EC`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 contour_proposal_head=None,
                 contour_evolve_head=None,
                 detector_fpn_start_level=1,
                 contour_fpn_start_level=1,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(E2EC, self).__init__(backbone, neck, bbox_head, contour_proposal_head,
                                   contour_evolve_head, detector_fpn_start_level,
                                   contour_fpn_start_level, train_cfg,
                                   test_cfg, pretrained, init_cfg)
