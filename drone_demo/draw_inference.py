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
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.data.datasets import giro
import copy
from PIL import Image
from maskrcnn_benchmark.structures.image_list import to_image_list
from torchvision import transforms as trns

from single_img_inference import evalpipe, overlay_boxes, overlay_class_names
import cv2

def main():
    print('ho ho ho')
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--threshold", type=float, default=0.95)
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
    print('started')
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
    cfg.merge_from_list(["MODEL.WEIGHT", args.weights])
    cfg.freeze()

    logger = setup_logger("maskrcnn_benchmark", custom_dict['out_path'], get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    
    ### build model
    print('building model')

    model = build_detection_model(cfg)
  
    model.roi_heads.box.predictor.cls_score=torch.nn.Linear(1024,9)
    model.roi_heads.box.predictor.bbox_pred=torch.nn.Linear(1024,36)
    
    device = torch.device(cfg.MODEL.DEVICE)
    
    if 'reload' in custom_dict.keys():
        logger.info('reloading weigts from '+ custom_dict['reload'])
        model.load_state_dict(torch.load(custom_dict['reload']),strict=False)
        
    model.cuda()
    model.eval()
    
    transform=build_transforms(cfg,is_train=False)
    
    out_path=custom_dict['out_path']
    in_path=custom_dict['in_path']
    annfile=custom_dict['annfile']
    no_pred=int(custom_dict['no_pred'])
    
    dataset=giro(ann_file=annfile,root=in_path,transforms=transform)
    ind_=len(dataset)-1
    for i in range(min(len(dataset),no_pred)):
        ind_-=10
        image, boxlist, idx=dataset[ind_]
        print('inference for: '+dataset.index_to_fname[idx])
        image_cv2=cv2.imread(os.path.join(in_path,dataset.index_to_fname[idx]))
        # actually, this one is somewhat clumsy, however when testing the snippet, this proved itself to be working
        image_=trns.ToPILImage()(image)
        image_=image_.resize((1344,768))
        
        _im=np.expand_dims(np.asarray(image_,dtype=float),0) # size (1,768,1344,3)
         
        input_im=torch.Tensor([[_im[0,:,:,i] for i in range(3)]]).cuda()
        input_im_list=to_image_list(input_im)
        
        out=model(input_im_list)
        
        predboxes=out[0].resize((image_cv2.shape[1],image_cv2.shape[0]))
        boxlist=boxlist.resize((image_cv2.shape[1],image_cv2.shape[0]))
#        picture=trns.ToPILImage()(image)
        
        predboxes=predboxes.to('cpu')

        picture=evalpipe(image_cv2,predboxes,threshold=args.threshold) # 
        picture=overlay_boxes(image_cv2,boxlist,gt=True)
        print('no gt boxes: %d' % len(boxlist))
        print(boxlist.bbox)
        print('no pred boxes: %d' % len(predboxes))
        print(predboxes.bbox)
        print('saving pic to: '+os.path.join(out_path,'pred_'+dataset.index_to_fname[idx]))
        Image.fromarray(picture).save(os.path.join(out_path,'pred_'+dataset.index_to_fname[idx]))
        
        
    
    
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

if __name__ == "__main__":
    main()


