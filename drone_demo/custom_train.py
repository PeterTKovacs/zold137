# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch

Modified the original script by Péter Kovács
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import numpy as np

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def train(logger,cfg, custom_dict, local_rank, distributed):
    model = build_detection_model(cfg)

# maybe this is fucking with us

    dummy_checkpointer = DetectronCheckpointer(cfg, model, save_dir='.')
    _ = dummy_checkpointer.load(cfg.MODEL.WEIGHT)
    # right now, we will have 8 classes, background may cause big problems
    model.roi_heads.box.predictor.cls_score=torch.nn.Linear(1024,9)
    model.roi_heads.box.predictor.bbox_pred=torch.nn.Linear(1024,36)
    device = torch.device(cfg.MODEL.DEVICE)
    if 'reload' in custom_dict.keys():
        logger.info('reloading weigts from '+ custom_dict['reload'])
        model.load_state_dict(torch.load(custom_dict['reload']),strict=False)
 
    for name,param in model.named_parameters():
        print(name,param.size())
        clashes=[to_unfreeze in name for to_unfreeze in custom_dict['to_unfreeze']]
        if True in clashes:
            param.requires_grad = True
            logger.info('unfroze: '+ name)
        else:
            param.requires_grad = False
            
    model.cuda()

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
  #  extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(model.state_dict())

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_valid=False,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        cfg,
        model,
        custom_dict,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model


def run_test(cfg,custom_dict, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
def read_custom_dict(file):
    d={}
    
    if file=='':
        return d
    
    with open(file,'r') as f:
        for line in f:
            tmp=line.split(':')
            if 'to_unfreeze' in line:
                unf_weights=tmp[1].split(',')
                d[tmp[0].strip()]=[w.strip() for w in unf_weights]
            elif 'val_frequency' in line:
                d[tmp[0].strip()]=int(tmp[1].strip())
            else:
                d[tmp[0].strip()]=tmp[1].strip()
            
    return d
        
        

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    parser.add_argument(
        "--custom-dict",
        default="",
        metavar="FILE",
        help="path to file with customization settings: dict-like organization",
        type=str,
    )
    parser.add_argument(
        "--weights",
        default="visdrone_model_0360000.pth",
        metavar="FILE",
        help="path to .pth weigth file",
        type=str,
    )

    args = parser.parse_args()
    custom_dict=read_custom_dict(args.custom_dict)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
   # config_file = "e2e_faster_rcnn_X_101_32x8d_FPN_1x_visdrone.yaml"
    cfg.merge_from_list(["MODEL.WEIGHT", args.weights])
    #cfg.merge_from_list(args.opts)
#    cfg['MODEL']['RPN']['FG_IOU_THRESHOLD']=0.2
#    cfg['MODEL']['RPN']['BG_IOU_THRESHOLD']=0.1
#    cfg['MODEL']['RPN']['BATCH_SIZE_PER_IMAGE']=50
#    cfg['MODEL']['RPN']['POSITIVE_FRACTION']=0.5
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(logger,cfg,custom_dict, args.local_rank, args.distributed)

#    for name,param in model.named_parameters():
#        print(name,param.requires_grad)

    if not args.skip_test:
        run_test(cfg, custom_dict, model, args.distributed)


if __name__ == "__main__":
    main()
