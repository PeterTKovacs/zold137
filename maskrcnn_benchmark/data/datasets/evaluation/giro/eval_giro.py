# custom evaluation for the giro dataset

# heavily based on the coco_eval.py counterpart (even copying much code!), tailored to my needs

import logging
import tempfile
import os
import torch
from collections import OrderedDict
from tqdm import tqdm
import copy

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


# keep same signature as for others

def giro_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    logger.info("Evaluating bbox proposals")
    areas = {"all": "", "small": "s", "medium": "m", "large": "l"}

    for area, suffix in areas.items():
        if area!='all':
            continue
#        print(len(predictions))
        stats = evaluate_box_proposals(predictions, dataset, area=area)
        logger.info('box sizes: '+area)
        logger.info(stats)
        if stats['num_pos']==0:
            return -1
        print('mean recall: %f'%stats['recalls'].mean())
        matches,acc1,acc2=accuracies(stats['best match labels'],stats["gt_labels"])
        print('accuracy on gt boxes:\n %f \t %f' % (acc1,acc2))
        print('number of matches %f\number of gt boxes: %f'%(matches,stats['num_pos']))
            
    return acc1

def accuracies(gt_labels,pred_labels):
    
    if len(gt_labels)>0:
        dn=float(len(gt_labels))
        aaa=gt_labels.float()==pred_labels.float()
        a1=aaa.float().sum()
        acc1=a1/dn
        matches=(pred_labels>=0.).float().sum()
        print((pred_labels>=0.).float())
        if matches>0:
            acc2=acc1*float(len(gt_labels))/float(matches)
            return (matches,acc1,acc2)
        else:
            return (matches,acc1,-1)
    else: 
        return (0,-1,-1)

# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area="all", # , limit=None : we're gonna consider all proposals, maybe not if too slow
):
    """
    Rewritten to Giro needs.
    We leverage the utilities supplied by BoxList objects and make use of the stucture of the returned predictions
    
    original help:
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    gt_labels=[]
    bm_cls_scores=[]
    bm_cls_preds=[]
    num_pos = 0

#    print(len(predictions))
    for image_id, prediction in enumerate(predictions):
#        print('Jack Robinson')
        # image_id is presumably the same at the 0,... indices we used in dataloader
        # it is supplied y the tqdm dataloader
#        print(prediction.fields())
#        print(prediction.get_field("scores"))
        img_pil, gt_boxes, _ =dataset[image_id]
        image_width, image_height=gt_boxes.size
#        img_info = dataset.get_img_info(image_id) !!! caused error because this is the raw size, not after transforms!
#        image_width = img_info["width"]
#        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height)) # BoxList object

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        
        # I guess there is no objectness field in the returned BoxList
        
#         inds = prediction.get_field("objectness").sort(descending=True)[1]
#         prediction = prediction[inds]

        # for GT boxes, we will use the standard dataset utilities
        
#        img_pil, gt_boxes, _ =dataset[image_id]
#        print(_)
#        prediction.get_field('objectness')
        gt_areas = gt_boxes.area() # standard boxlist utility, gives torch tensor

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds] # hopefully takes care of labels too, since __get_item__ does that too
    
        if len(gt_boxes) == 0:
            print('leg(gt_boxes)==0')
            continue

        if len(prediction) == 0:
            print('leg(prediction)==0')
            continue

        num_pos += len(gt_boxes)
        gt_labels.append(copy.deepcopy(gt_boxes.extra_fields['labels']))

#         if limit is not None and len(prediction) > limit: won't do first - not to spoil gt labels
#             prediction = prediction[:limit]
#        print(prediction)
#        print(gt_boxes)
        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        best_match_cls_preds=torch.ones(len(gt_boxes))*-1. # need -1: it is possible that not all GT boxes get a match
        best_match_score=torch.zeros(len(gt_boxes))
        
        for j in range(min(len(prediction), len(gt_boxes))):
            
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1 # prediction
            overlaps[:, gt_ind] = -1  # gt
            best_match_cls_preds[gt_ind]=prediction.get_field('labels')[box_ind]
            best_match_score[gt_ind]=prediction.get_field('scores')[box_ind]

        # append recorded iou coverage level for given image, GT class, pred scrore, pred class
        bm_cls_scores.append(best_match_score)
        bm_cls_preds.append(best_match_cls_preds)
        gt_overlaps.append(_gt_overlaps) 
#        print('gt_overlap appending')
    if num_pos==0:
        return {
        "ar": 0,
        "recalls": 0,
        "thresholds": 0,
        "gt_overlaps": 0,
        "gt_labels": torch.Tensor([]),
        'best match labels':torch.Tensor([]),
        'best match scores':torch.Tensor([]),
        "num_pos": num_pos,
         }


    gt_overlaps = torch.cat(gt_overlaps, dim=0) # for the whole batch
    gt_labels= torch.cat(gt_labels, dim=0)
    bm_cls_scores=torch.cat(bm_cls_scores, dim=0)
    bm_cls_preds=torch.cat(bm_cls_preds, dim=0)
    #gt_overlaps, _ = torch.sort(gt_overlaps) # do not want to loose parallel info

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos) # all gt boxes in batch
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "gt_labels": gt_labels,
        'best match labels':bm_cls_preds,
        'best match scores':bm_cls_scores,
        "num_pos": num_pos,
    }
