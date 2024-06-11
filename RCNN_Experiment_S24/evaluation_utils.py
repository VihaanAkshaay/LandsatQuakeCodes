# Imports

#Downloading torch util functions:
import os


# Imports

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torch.nn as nn
import numpy as np
import os

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from skimage.draw import polygon


import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
from torch.utils.data._utils.collate import default_collate
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from tqdm import tqdm
import torch
from tqdm import tqdm
from engine import train_one_epoch, evaluate
import matplotlib.pyplot as plt

import numpy as np
from collections import defaultdict


'''
List of utils:
bbox_iou(box1, box2) - calculate the IoU of two bounding boxes.
mask_iou(mask1, mask2) - calculate the IoU of two binary masks.
validation_evaluator(model, data_loader, device) - evaluate the model on the validation set.
evaluate_metrics(model, data_loader, device, iou_thresholds) - evaluate the model on metrics such as precision, recall and mAP.
'''

# Metrics

def bbox_iou(box1, box2):
    """
    Calculate the IoU of two bounding boxes.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    union_area = b1_area + b2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area

    return iou


def mask_iou(mask1, mask2):
    """
    Calculate the IoU of two binary masks.
    """
    # Ensure the masks are boolean
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    # Intersection and union
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()

    # Compute the IoU
    iou = intersection / union if union > 0 else 0

    return iou

# Losses:

def validation_evaluator(model, data_loader, device):
    model.train()  # Keep model in training mode but disable gradients
    loss_dict_sum = {}
    total_loss_sum = 0
    with torch.no_grad():  # Disable gradient calculations
        for images, targets in tqdm(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            total_loss_sum += total_loss.item()

            for k, v in loss_dict.items():
                if k in loss_dict_sum:
                    loss_dict_sum[k] += v.item()
                else:
                    loss_dict_sum[k] = v.item()

    # Average the losses
    for k in loss_dict_sum.keys():
        loss_dict_sum[k] /= len(data_loader)
    total_loss_avg = total_loss_sum / len(data_loader)

    loss_dict_sum['total_loss'] = total_loss_avg

    return loss_dict_sum

# eval metrics

def area_under_curve(recalls, precisions):
    # Sort recalls and precisions based on recalls
    sorted_indices = np.argsort(recalls)
    sorted_recalls = np.array(recalls)[sorted_indices]
    sorted_precisions = np.array(precisions)[sorted_indices]
    # Compute the area under the Precision-Recall curve
    return np.trapz(sorted_precisions, sorted_recalls)

def evaluate_metrics(model, data_loader, device, objectiveness_threshold=0.0, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    model.eval()
    bbox_precisions = defaultdict(list)
    bbox_recalls = defaultdict(list)
    mask_precisions = defaultdict(list)
    mask_recalls = defaultdict(list)

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                gt_boxes = targets[i]['boxes'].to(device)
                gt_masks = targets[i]['masks'].to(device)
                pred_boxes = output['boxes']
                pred_scores = output['scores']
                pred_masks = output['masks']

                high_scores_idx = pred_scores > objectiveness_threshold
                pred_boxes = pred_boxes[high_scores_idx]
                pred_masks = pred_masks[high_scores_idx]

                # Evaluate for each IoU threshold
                for iou_threshold in iou_thresholds:
                    bbox_TP = bbox_FP = bbox_FN = mask_TP = mask_FP = mask_FN = 0

                    matched_gt_bbox_indices = set()
                    matched_gt_mask_indices = set()

                    for pred_box, pred_mask in zip(pred_boxes, pred_masks):
                        max_bbox_iou = max_mask_iou = 0
                        max_bbox_iou_index = max_mask_iou_index = -1

                        for j, (gt_box, gt_mask) in enumerate(zip(gt_boxes, gt_masks)):
                            bbox_iou_val = bbox_iou(pred_box, gt_box)
                            mask_iou_val = mask_iou(pred_mask, gt_mask)

                            if bbox_iou_val > max_bbox_iou and bbox_iou_val > iou_threshold and j not in matched_gt_bbox_indices:
                                max_bbox_iou = bbox_iou_val
                                max_bbox_iou_index = j
                            
                            if mask_iou_val > max_mask_iou and mask_iou_val > iou_threshold and j not in matched_gt_mask_indices:
                                max_mask_iou = mask_iou_val
                                max_mask_iou_index = j

                        if max_bbox_iou_index != -1:
                            matched_gt_bbox_indices.add(max_bbox_iou_index)
                            bbox_TP += 1
                        else:
                            bbox_FP += 1
                        
                        if max_mask_iou_index != -1:
                            matched_gt_mask_indices.add(max_mask_iou_index)
                            mask_TP += 1
                        else:
                            mask_FP += 1

                    bbox_FN = len(gt_boxes) - len(matched_gt_bbox_indices)
                    mask_FN = len(gt_masks) - len(matched_gt_mask_indices)

                    # Bounding Box Metrics
                    if bbox_TP + bbox_FP > 0:
                        bbox_precision = bbox_TP / (bbox_TP + bbox_FP)
                        bbox_precisions[iou_threshold].append(bbox_precision)
                    if bbox_TP + bbox_FN > 0:
                        bbox_recall = bbox_TP / (bbox_TP + bbox_FN)
                        bbox_recalls[iou_threshold].append(bbox_recall)

                    # Mask Metrics
                    if mask_TP + mask_FP > 0:
                        mask_precision = mask_TP / (mask_TP + mask_FP)
                        mask_precisions[iou_threshold].append(mask_precision)
                    if mask_TP + mask_FN > 0:
                        mask_recall = mask_TP / (mask_TP + mask_FN)
                        mask_recalls[iou_threshold].append(mask_recall)

    # Calculate AP, AP50, AP75 for Bounding Boxes and Masks
    bbox_AP = {iou: area_under_curve(bbox_recalls[iou], bbox_precisions[iou]) for iou in iou_thresholds}
    bbox_AP50 = bbox_AP[0.5]
    bbox_AP75 = bbox_AP[0.75]

    mask_AP = {iou: area_under_curve(mask_recalls[iou], mask_precisions[iou]) for iou in iou_thresholds}
    mask_AP50 = mask_AP[0.5]
    mask_AP75 = mask_AP[0.75]

    return {'bbox_AP': bbox_AP, 'bbox_AP50': bbox_AP50, 'bbox_AP75': bbox_AP75, 'mask_AP': mask_AP, 'mask_AP50': mask_AP50, 'mask_AP75': mask_AP75}

# Usage Example
# results = evaluate_metrics(model, data_loader, device)
