import os
import glob

# 1. Inherit the base configuration
_base_ = '../configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'

# ================= 2. Core Path Configuration =================
data_root = 'E:/DJ/Scientific data/'

# ================= 3. Must explicitly define Pipeline (otherwise it cannot be called in functions) =================
# I directly copy the standard RTMDet-L Pipeline here to ensure the function can locate the variable
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

# Stage 2 Augmentation (Mosaic disabled)
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# ================= 5. Model Modification =================
model = dict(
    bbox_head=dict(num_classes=1)
)

# ================= 6. Data Loaders =================
metainfo = {
    'classes': ('person',),
    'palette': [(220, 20, 60)]
}

train_dataloader = dict(
    batch_size=3,
    num_workers=1,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
##################  Before Corruption #################
        ann_file='annotations_gt/train_merged.json',
        data_prefix=dict(img='gt/train/'),
##################  After Corruption #################
        # ann_file='annotations_corrupted/train_merged.json',
        # data_prefix=dict(img='corrupted/train/'),

        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=3,
    num_workers=1,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ##################  Before Corruption #################
        ann_file='annotations_gt/val_merged.json',
        data_prefix=dict(img='gt/val/'),
        ##################  After Corruption #################
        # ann_file='annotations_corrupted/val_merged.json',
        # data_prefix=dict(img='corrupted/val/'),

        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=3,
    num_workers=1,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ##################  Before Corruption #################
        ann_file='annotations_gt/test_merged.json',
        data_prefix=dict(img='gt/test/'),
        ##################  After Corruption #################
        # ann_file='annotations_corrupted/test_merged.json',
        # data_prefix=dict(img='corrupted/test/'),

        test_mode=True,
        pipeline=test_pipeline
    )
)
# ================= 7. Evaluation Metrics =================
val_evaluator = dict(
    type='CocoMetric',
    ##################  Before Corruption #################
    ann_file='E:/DJ/Scientific data/annotations_gt/val_merged.json',
    ##################  After Corruption #################
    # ann_file='E:/DJ/Scientific data/annotations_corrupted/val_merged.json',

    metric='bbox',
    classwise=True
)
test_evaluator = dict(
    type='CocoMetric',
    ##################  Before Corruption #################
    ann_file='E:/DJ/Scientific data/annotations_gt/test_merged.json',
    ##################  After Corruption #################
    # ann_file='E:/DJ/Scientific data/annotations_corrupted/test_merged.json',

    metric='bbox',
    classwise=True
)

# ======================================================
#            Debug / Quick Test Mode Configuration (Override Section)
# ======================================================

# 1. Set Parameters
max_epochs = 300
stage2_num_epochs = 50
base_lr = 0.00075  # <--- Use a safe small learning rate

# 2. Training Flow Control
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=2,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)]
)

# 3. Learning Rate Strategy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000), # Light warmup
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# 4. Optimizer Rewrite (The most critical step!!!)
# Must assign base_lr here and add gradient clipping (clip_grad)
optim_wrapper = dict(
    _delete_=True,        # Delete inherited old configurations and rewrite completely
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05), # Use 0.0001
    clip_grad=dict(max_norm=35, norm_type=2), # <--- Add this line to prevent NaN values
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
)

# 5. Hooks
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2) # Call the variable defined above here
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        save_best='auto',
        max_keep_ckpts=2,
        save_last=True
    )
)