# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class WhuDataset(CocoDataset):
    CLASSES = ('building', )
    PALETTE = [(220, 20, 60), ]