_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
train_dataloader = dict(
    num_workers=0,
)
val_dataloader = dict(
    num_workers=0,
)
test_dataloader = dict(
    num_workers=0,
)
data = dict(
    samples_per_gpu=2,  # 你的 batch size，保持原样或根据需要修改
    workers_per_gpu=0   # 强制改成 0，解决报错
)