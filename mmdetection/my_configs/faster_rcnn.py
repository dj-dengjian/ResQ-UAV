import os
import glob

# 1. Inherit official Faster R-CNN configuration
_base_ = '../configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# ================= 2. Core Path Configuration (Linux) =================
data_root = 'E:/DJ/Scientific data/'

# ================= 3. Explicitly Define Pipeline (Critical) =================
# Standard Faster R-CNN Pipeline (Multi-scale training is crucial for Faster R-CNN, using the standard (1333, 800) here)
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

# ================= 4. Model Modification (Single Class) =================
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1  # Modify to 1 class (person)
        )
    )
)

# ================= 5. Data Loaders =================
metainfo = {
    'classes': ('person',),
    'palette': [(220, 20, 60)]
}

# --- Train ---
train_dataloader = dict(
    batch_size=4,  # Use 8 if GPU memory is sufficient, otherwise adjust to 4 or 2
    num_workers=2,  # Recommended to set to 4 or 8
    dataset=dict(
        _delete_=True,  # Ignore redundant configurations in the base class and redefine from scratch
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
##################  Before Corruption #################
        # ann_file='annotations_gt/train_merged.json',
        # data_prefix=dict(img='gt/train/'),
##################  After Corruption #################
        ann_file='annotations_corrupted/train_merged.json',
        data_prefix=dict(img='corrupted/train/'),

        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)
# --- Val ---
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ##################  Before Corruption #################
        # ann_file='annotations_gt/val_merged.json',
        # data_prefix=dict(img='gt/val/'),
        ##################  After Corruption #################
        ann_file='annotations_corrupted/val_merged.json',
        data_prefix=dict(img='corrupted/val/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

# --- Test (Run only on GT test set) ---
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ##################  Before Corruption #################
        # ann_file='annotations_gt/test_merged.json',
        # data_prefix=dict(img='gt/test/'),
        ##################  After Corruption #################
        ann_file='annotations_corrupted/test_merged.json',
        data_prefix=dict(img='corrupted/test/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

# ================= 6. Evaluation Metrics =================
val_evaluator = dict(
    type='CocoMetric',
    ##################  Before Corruption #################
    # ann_file='E:/DJ/Scientific data/annotations_gt/val_merged.json',
    ##################  After Corruption #################
    ann_file='E:/DJ/Scientific data/annotations_corrupted/val_merged.json',
    metric='bbox',
    classwise=True
)

test_evaluator = dict(
    type='CocoMetric',
    ##################  Before Corruption #################
    # ann_file='E:/DJ/Scientific data/annotations_gt/test_merged.json',
    ##################  After Corruption #################
    ann_file='E:/DJ/Scientific data/annotations_corrupted/test_merged.json',
    metric='bbox',
    classwise=True
)

# ======================================================
#             Training Strategy Configuration
# ======================================================

# 1. Set Parameters
# Faster R-CNN defaults to 1x schedule (12 epochs), recommend 12 or 24 epochs here
max_epochs = 24
base_lr = 0.0001  # Small learning rate is recommended for AdamW

# 2. Training Flow Control
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=2
)

# 3. Learning Rate Strategy (Cosine Annealing)
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

# 4. Optimizer (AdamW + Gradient Clipping)
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',  # Enable Automatic Mixed Precision
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),  # Prevent gradient explosion
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