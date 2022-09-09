# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Function
import native_rasterizer

from ..builder import LOSSES
from .utils import weighted_loss

@LOSSES.register_module()
class DMLoss(nn.Module):
    """Dynamic Matching loss.

    Args:
        key_item_weight (float, 0 <= key_item_weight<= 1): The weight if item key points.
            Defaults to 0.5
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, loss_weight=1.0, offsets_stride=4.,
                 key_item_weight=0.5, crit_type='smoothL1',
                 ignore_bound=1000.):
        super(DMLoss, self).__init__()
        self.key_item_weight = key_item_weight
        self.loss_weight = loss_weight
        self.offsets_stride = offsets_stride
        self.crit_type = crit_type
        self.ignore_bound = ignore_bound
        assert crit_type in ['smoothL1', 'L1']
        if crit_type == 'smoothL1':
            self.crit = smooth_l1_loss
            self.beta = 1.0 / offsets_stride
        else:
            self.crit = l1_loss

    def interpolation(self, poly, time=10):
        ori_points_num = poly.size(1)
        poly_roll = torch.roll(poly, shifts=1, dims=1)
        poly_ = poly.unsqueeze(3).repeat(1, 1, 1, time)
        poly_roll = poly_roll.unsqueeze(3).repeat(1, 1, 1, time)
        step = torch.arange(0, time, dtype=torch.float32).cuda() / time
        poly_interpolation = poly_ * step + poly_roll * (1. - step)
        poly_interpolation = poly_interpolation.permute(0, 1, 3, 2).reshape(poly_interpolation.size(0),
                                                                            ori_points_num * time, 2)
        return poly_interpolation

    def compute_distance(self, pred_poly, gt_poly):
        pred_poly_expand = pred_poly.unsqueeze(1)
        gt_poly_expand = gt_poly.unsqueeze(2)
        gt_poly_expand = gt_poly_expand.expand(gt_poly_expand.size(0), gt_poly_expand.size(1),
                                               pred_poly_expand.size(2), gt_poly_expand.size(3))
        pred_poly_expand = pred_poly_expand.expand(pred_poly_expand.size(0), gt_poly_expand.size(1),
                                                   pred_poly_expand.size(2), pred_poly_expand.size(3))
        distance = torch.sum((pred_poly_expand - gt_poly_expand) ** 2, dim=3)
        return distance

    def get_pred_targets_item1(self, preds, offsets, targets):
        targets = self.interpolation(targets)
        distances = self.compute_distance(preds, targets)
        matched_dis, index_gt = torch.min(distances, dim=1)
        valid = matched_dis <= self.ignore_bound ** 2
        index_0 = torch.arange(index_gt.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_gt.size(0), index_gt.size(1))
        targets = targets[index_0, index_gt, :]
        offsets_target = (targets - preds) / self.offsets_stride
        return offsets[valid], offsets_target.detach()[valid]

    def get_pred_targets_item2(self, preds, offsets, key_points, masks):
        masks = masks.to(torch.bool)
        distances = self.compute_distance(key_points, preds)
        #(N, n_key_points, n_pred_points)
        matched_dis, index_pred = torch.min(distances, dim=1)
        valid = matched_dis <= self.ignore_bound ** 2
        index_0 = torch.arange(index_pred.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_pred.size(0), index_pred.size(1))
        preds = preds[index_0, index_pred, :][masks]
        offsets = offsets[index_0, index_pred, :][masks]
        valid = valid[masks]
        offsets_target = (key_points[masks] - preds) / self.offsets_stride
        return offsets[valid], offsets_target.detach()[valid]

    def loss(self, pred, targets, weight, avg_factor=None):
        if self.crit_type == 'smoothL1':
            return self.crit(pred, targets, weight=None, beta=self.beta, reduction='mean', avg_factor=avg_factor) *\
                   weight
        else:
            return self.crit(pred, targets, weight=None, reduction='mean', avg_factor=avg_factor) * weight


    def forward(self,
                pred_contours,
                pred_offsets,
                gt_contours,
                gt_key_points,
                gt_key_points_mask,
                avg_factor=None,
                **kwargs):
        """Forward function.
        Args:
        pred_contours torch.Tensor(N, nums_points, 2)
        gt_contours torch.Tensor(N, nums_points, 2)
        gt_key_points torch.Tensor(N, nums_points, 2)
        gt_key_points_mask torch.Tensor(N, nums_points)
        """
        pred_item1, target_item1 = self.get_pred_targets_item1(pred_contours, pred_offsets, gt_contours)
        pred_item2, target_item2 = self.get_pred_targets_item2(pred_contours, pred_offsets,
                                                               gt_key_points, gt_key_points_mask)
        if avg_factor is None:
            avg_factor1, avg_factor2 = None, None
        else:
            avg_factor1, avg_factor2 = avg_factor
        loss1 = self.loss(pred_item1, target_item1, weight=1 - self.key_item_weight, avg_factor=avg_factor1)
        loss2 = self.loss(pred_item2, target_item2, weight=self.key_item_weight, avg_factor=avg_factor2)
        return (loss1 + loss2) * self.loss_weight

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss


MODE_BOUNDARY = "boundary"
MODE_MASK = "mask"
MODE_HARD_MASK = "hard_mask"

MODE_MAPPING = {
    MODE_BOUNDARY: 0,
    MODE_MASK: 1,
    MODE_HARD_MASK: 2
}


class SoftPolygonFunction(Function):
    @staticmethod
    def forward(ctx, vertices, width, height, inv_smoothness=1.0, mode=MODE_BOUNDARY):
        ctx.width = width
        ctx.height = height
        ctx.inv_smoothness = inv_smoothness
        ctx.mode = MODE_MAPPING[mode]

        vertices = vertices.clone()
        ctx.device = vertices.device
        ctx.batch_size, ctx.number_vertices = vertices.shape[:2]

        rasterized = torch.FloatTensor(ctx.batch_size, ctx.height, ctx.width).fill_(0.0).to(device=ctx.device)

        contribution_map = torch.IntTensor(
            ctx.batch_size,
            ctx.height,
            ctx.width).fill_(0).to(device=ctx.device)
        rasterized, contribution_map = native_rasterizer.forward_rasterize(vertices, rasterized, contribution_map,
                                                                           width, height, inv_smoothness, ctx.mode)
        ctx.save_for_backward(vertices, rasterized, contribution_map)

        return rasterized  # , contribution_map

    @staticmethod
    def backward(ctx, grad_output):
        vertices, rasterized, contribution_map = ctx.saved_tensors

        grad_output = grad_output.contiguous()

        # grad_vertices = torch.FloatTensor(
        #    ctx.batch_size, ctx.height, ctx.width, ctx.number_vertices, 2).fill_(0.0).to(device=ctx.device)
        grad_vertices = torch.FloatTensor(
            ctx.batch_size, ctx.number_vertices, 2).fill_(0.0).to(device=ctx.device)
        grad_vertices = native_rasterizer.backward_rasterize(
            vertices, rasterized, contribution_map, grad_output, grad_vertices, ctx.width, ctx.height,
            ctx.inv_smoothness, ctx.mode)

        return grad_vertices, None, None, None, None


class SoftPolygon(nn.Module):
    MODES = [MODE_BOUNDARY, MODE_MASK, MODE_HARD_MASK]

    def __init__(self, inv_smoothness=1.0, mode=MODE_BOUNDARY):
        super(SoftPolygon, self).__init__()

        self.inv_smoothness = inv_smoothness

        if not (mode in SoftPolygon.MODES):
            raise ValueError("invalid mode: {0}".format(mode))

        self.mode = mode

    def forward(self, vertices, width, height, p, color=False):
        return SoftPolygonFunction.apply(vertices, width, height, self.inv_smoothness, self.mode)

def dice_loss(input, target, avg_factor=None):
    n = input.size(0)
    smooth = 1.
    iflat = input.reshape(n, -1)
    tflat = target.reshape(n, -1)
    intersection = (iflat * tflat).sum(dim=-1)
    losses = 1 - ((2. * intersection + smooth) / (iflat.sum(dim=-1) + tflat.sum(dim=-1) + smooth))
    if avg_factor is None:
        return losses.sum() / n
    else:
        return losses.sum() / avg_factor

@LOSSES.register_module()
class MaskRasterizationLoss(nn.Module):
    def __init__(self, loss_weight=1.0, resolution=[64, 64], inv_smoothness=0.1):
        super().__init__()
        self.resolution = resolution
        self.register_buffer("rasterize_at",
                             torch.from_numpy(np.array(resolution).reshape(-1, 2)))
        self.inv_smoothness = inv_smoothness
        self.pred_rasterizer = SoftPolygon(inv_smoothness=self.inv_smoothness, mode="mask")
        self.offset = 0.5
        self.loss_fn = dice_loss
        self.name = "mask"
        self.loss_weight = loss_weight

    def get_union_bboxes(self, pred_polygons, targets_bboxes):
        pred_bboxes = torch.cat([torch.min(pred_polygons, dim=1)[0],
                                 torch.max(pred_polygons, dim=1)[0]], dim=-1)
        pred_bboxes[..., :2] = torch.minimum(pred_bboxes[..., :2],
                                             targets_bboxes[..., :2])
        pred_bboxes[..., 2:4] = torch.maximum(pred_bboxes[..., 2:4],
                                              targets_bboxes[..., 2:4])
        return pred_bboxes

    def crop_targets_masks(self, targets_masks, union_bboxes):
        ret = []
        sx = 0
        for targets_mask in targets_masks:
            ret.append(targets_mask.crop_and_resize(union_bboxes[sx: sx + len(targets_mask)], self.resolution,
                                                    np.arange(len(targets_mask)),
                                                    device=union_bboxes.device,
                                                    interpolation='bilinear',
                                                    binarize=True).to_tensor(dtype=torch.float32, device=union_bboxes.device))
            sx += len(targets_mask)

        return torch.cat(ret, dim=0)

    def get_normed_polygons(self, polygons, bboxes):
        bboxes = bboxes.detach()
        polygons = (polygons - bboxes[..., :2].unsqueeze(1)) /\
                   (bboxes[..., 2:4] - bboxes[..., :2]).unsqueeze(1)
        return polygons

    def forward(self, preds, targets_masks, targets_bboxes, avg_factor=None):
        # targets_masks BitMasks
        batch_size = len(preds)
        if batch_size == 0:
            return preds.sum() * 0
        union_bboxes = self.get_union_bboxes(preds, targets_bboxes)
        targets_masks = self.crop_targets_masks(targets_masks, union_bboxes)

        resolution = self.rasterize_at[0]

        # -0.5 needed to align the rasterizer with COCO.
        preds = self.get_normed_polygons(preds, union_bboxes)
        pred_masks = self.pred_rasterizer(preds * float(resolution[1].item()) - self.offset,
                                          resolution[1].item(),
                                          resolution[0].item(), 1.0).unsqueeze(1)
        # add a little noise since depending on inv_smoothness/predictions, we can exactly predict 0 or 1.0
        pred_masks = torch.clamp(pred_masks, 0.00001, 0.99999)
        return self.loss_fn(pred_masks, targets_masks, avg_factor=avg_factor) * self.loss_weight
