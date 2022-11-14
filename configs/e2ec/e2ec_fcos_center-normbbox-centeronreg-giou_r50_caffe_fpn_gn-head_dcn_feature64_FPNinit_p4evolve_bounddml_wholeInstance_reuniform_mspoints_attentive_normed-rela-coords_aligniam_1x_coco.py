_base_ = 'e2ec_fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_feature64_p4init_p4evolve_dml_1x_coco.py'

model = dict(
    contour_proposal_head=dict(
        type='IamFPNContourProposalHead',
        start_level=1,
        in_channel=64,
        hidden_dim=256,
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=(16, 16), sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4]),
        align_num=4,
        point_nums=[32, 64],
        loss_contour_mask=dict(
            type='MaskRasterizationLoss',
            loss_weight=0.1,
            resolution=[64, 64],
            inv_smoothness=0.1
        ),
        loss_contour=dict(
            type='SmoothL1Loss',
            beta=0.05,
            loss_weight=0.1),
        loss_init=dict(
            type='SmoothL1Loss',
            beta=0.05,
            loss_weight=0.2),
    ),
    contour_evolve_head=dict(
        type='AttentiveContourEvolveHead',
        point_nums=[128, 128, 256],
        use_tanh=False,
        norm_type='constant',
        evolve_deform_ratio=1.0,
        attentive_expand_ratio=1.2,
        add_normed_coords=True,
        loss_contour_mask=dict(
            type='MaskRasterizationLoss',
            loss_weight=0.33,
            resolution=[64, 64],
            inv_smoothness=0.1
        ),
        loss_last_evolve=dict(ignore_bound=50.),
    ),

)

# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='Resize',
        img_scale=(736, 512),
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='AlignSampleBoundary', point_nums=128, reset_bbox=False, ignore_multi_components_instances=True,
         gt_masks2bit=True),
    dict(type='ContourDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',
         'gt_masks', 'gt_polys', 'key_points_masks', 'key_points', 'is_single_component']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(736, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
