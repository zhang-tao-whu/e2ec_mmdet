_base_ = 'e2ec_fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_feature64_p4init_p4evolve_dml_1x_coco.py'

model = dict(
    bbox_head=dict(num_classes=1,),
    contour_proposal_head=dict(
        type='FPNContourProposalHead',
        start_level=1,
        in_channel=64,
        hidden_dim=256,
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
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 1333), (1333, 896)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='AlignSampleBoundary', point_nums=128, reset_bbox=True),
    dict(type='ContourDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',
         'gt_masks', 'gt_polys', 'key_points_masks', 'key_points']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
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
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=1.0, decay_mult=1.0)}))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[45])
runner = dict(type='EpochBasedRunner', max_epochs=80)
evaluation = dict(metric=['bbox', 'segm'], interval=15)
checkpoint_config = dict(interval=5)
