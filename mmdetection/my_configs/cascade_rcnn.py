import os
import glob

# 1. Inherit official Cascade R-CNN configuration
_base_ = '../configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py'

# ================= 2. Core Path Configuration (Linux) =================
data_root = 'E:/DJ/Scientific data/'

# ================= 3. Explicitly Define Pipeline (Critical) =================
# Standard Pipeline for Cascade R-CNN (Mosaic is generally not used, different from RTMDet)
# Image size is usually kept at (1333, 800) for better accuracy
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# ================= 4. Model Modifications (Modify all 3 heads) =================
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=1),  # Stage 1
            dict(
                type='Shared2FCBBoxHead',
                num_classes=1),  # Stage 2
            dict(
                type='Shared2FCBBoxHead',
                num_classes=1)  # Stage 3
        ]
    )
)

# ================= 5. Data Loaders =================
metainfo = {
    'classes': ('person',),
    'palette': [(220, 20, 60)]
}

# --- Train ---
train_dataloader = dict(
    batch_size=4,  # Use 8 if GPU memory is sufficient, otherwise change to 4 or 2
    num_workers=2,  # Recommended to set to 4 or 8
    dataset=dict(
        _delete_=True,  # Ignore redundant configurations in the base class, redefine from scratch
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
# --- Val  ---
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
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

# --- Test  ---
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
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

# ================= 6. Evaluation Metrics =================
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
#             Training Strategy Configuration
# ======================================================

# 1. Set Parameters (Easy for Adjustment)
max_epochs = 24  # Cascade R-CNN converges slowly, recommend at least 12 epochs, preferably 24+
base_lr = 0.0001  # AdamW learning rate is usually smaller than SGD (0.02 for SGD in general)

# 2. Training Flow Control
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1
)

# 3. Learning Rate Strategy (Use Cosine for smoother decay)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),  # Warmup
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]

# 4. Optimizer (Use AdamW + Gradient Clipping to prevent NaN)
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',  # Enable Automatic Mixed Precision to save GPU memory
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),  # Critical: Prevent gradient explosion
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
)

# 5. Checkpoint Saving Strategy
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=2,
        save_last=True
    ),
    logger=dict(type='LoggerHook', interval=50)
)