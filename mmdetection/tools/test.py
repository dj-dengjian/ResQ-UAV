# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# 获取train.py所在的tools目录的上级目录（即项目根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将项目根目录加入sys.path
sys.path.append(project_root)


from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    #parser.add_argument('checkpoint', default='E:/DJ/Scientific data/mmdetection/tools/work_dirs/rtmdet/best_coco_person_precision_epoch_84.pth',help='checkpoint file')
    # parser.add_argument('config', help='test config file path')
    parser.add_argument('--work-dir',help='the directory to save the file containing evaluation metrics')
    # parser.add_argument('--work-dir', default= 'E:/DJ/Scientific data/mmdetection/tools/work_dirs/rtmdet/best_coco_person_precision_epoch_84.pth', help='the dir to save logs and models')  ###### 20216.1.2修改配置环境

    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    ###########################rtmdet模型#######################
    ############## 破坏之前 ####################
    # args.config = 'E:/DJ/Scientific data/mmdetection/my_configs/rtmdet.py'
    # args.checkpoint = 'E:/DJ/Scientific data/mmdetection/tools/work_dirs/rtmdet/best_coco_person_precision_epoch_300.pth'
    ############## 破坏之后 ####################
    # args.config = 'E:/DJ/Scientific data/mmdetection/my_configs/rtmdet.py'
    # args.checkpoint = 'E:/DJ/Scientific data/mmdetection/tools/work_dirs/rtmdet_corrupted/best_coco_person_precision_epoch_266.pth'
    ####################################################################

    ###########################faster_rcnn模型#######################
    ############## 破坏之前 ####################
    # args.config = 'E:/DJ/Scientific data/mmdetection/my_configs/faster_rcnn.py'
    # args.checkpoint = 'E:/DJ/Scientific data/mmdetection/tools/work_dirs/faster_rcnn/best_coco_person_precision_epoch_16.pth'
    # ############## 破坏之后 ####################
    # args.config = 'E:/DJ/Scientific data/mmdetection/my_configs/faster_rcnn.py'
    # args.checkpoint = 'E:/DJ/Scientific data/mmdetection/tools/work_dirs/faster_rcnn_corrupted/best_coco_person_precision_epoch_4.pth'
    ###################################################################

    ###########################cascade_rcnn模型#######################
    ############## 破坏之前 ####################
    # args.config = 'E:/DJ/Scientific data/mmdetection/my_configs/cascade_rcnn.py'
    # args.checkpoint = 'E:/DJ/Scientific data/mmdetection/tools/work_dirs/cascade_rcnn/best_coco_person_precision_epoch_6.pth'
    ############## 破坏之后 ####################
    # args.config = 'E:/DJ/Scientific data/mmdetection/my_configs/cascade_rcnn.py'
    # args.checkpoint = 'E:/DJ/Scientific data/mmdetection/tools/work_dirs/cascade_rcnn_corrupted/best_coco_person_precision_epoch_8.pth'
    ####################################################################

    ###########################dino模型#######################
    ############## 破坏之前 ####################
    args.config = 'E:/DJ/Scientific data/mmdetection/my_configs/dino.py'
    args.checkpoint = 'E:/DJ/Scientific data/mmdetection/tools/work_dirs/dino/best_coco_person_precision_epoch_14.pth'
    # ############## 破坏之后 ####################
    # args.config = 'E:/DJ/Scientific data/mmdetection/my_configs/dino.py'
    # args.checkpoint = 'E:/DJ/Scientific data/mmdetection/tools/work_dirs/dino_corrupted/best_coco_person_precision_epoch_13.pth'
    ####################################################################
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
