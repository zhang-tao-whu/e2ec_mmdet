# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
import pycocotools.mask as maskUtils

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
                      gt_contours,
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
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        contour_proposals, losses_contour_proposal = self.contour_proposal_head.forward_train(x, img_metas,
                                                                                              gt_bboxes, gt_contours)
        losses_contour_evolve = self.contour_evolve_head.forward_train(x, img_metas, contour_proposals, gt_contours)
        losses.update(losses_contour_proposal)
        losses.update(losses_contour_evolve)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
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
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        #results_list [(bboxes, labels), ...]
        # boxes (Tensor): Bboxes with score after nms, has shape (num_bboxes, 5). last dimension 5 arrange as (x1, y1, x2, y2, score)
        # labels (Tensor): has shape (num_bboxes, )
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        bboxes_pred = [item[0] for item in results_list]
        labels_pred = [item[1] for item in results_list]
        contour_proposals = self.contour_proposal_head.forward_test(feat, img_metas, bboxes_pred)
        contours_pred = self.contour_evolve_head.forward_test(feat, img_metas, contour_proposals)
        img_shape = img_metas[0]['batch_input_shape']
        return bbox_results, self.convert_contour2mask(contours_pred, labels_pred, img_shape)

    def convert_contour2mask(self, contours_pred, labels_pred, img_shape):
        #masks_pred [single img masks_pred]
        #single img masks_pred [single class instances mask]
        #instance mask (h, w)
        mask_pred = [[] for _ in range(self.bbox_head.num_classes)]
        contours_pred = contours_pred.detach().cpu().numpy()
        labels_pred = labels_pred.detach().cpu().numpy()
        rles = []
        for contour in contours_pred:
            contour = contour.flatten().tolist()
            rle = maskUtils.frPyObjects([contour], img_shape[0], img_shape[1])
            rles += rle
        masks = maskUtils.decode(rles).transpose(2, 0, 1)
        for mask, label in zip(masks, labels_pred):
            mask_pred[int(label)].append(mask)
        masks_pred = None
        return masks_pred

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
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(E2EC, self).__init__(backbone, neck, bbox_head, contour_proposal_head,
                                   contour_evolve_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
