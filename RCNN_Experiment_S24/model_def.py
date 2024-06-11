#Downloading torch util functions:
import os


# Imports

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection import MaskRCNN
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

from evaluation_utils import bbox_iou, mask_iou, validation_evaluator, evaluate_metrics

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from typing import List, Tuple


# RCNN Model (7 channels in but rest normal)

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    targets = list(targets)
    return images, targets

class SimpleTransform(GeneralizedRCNNTransform):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super().__init__(min_size, max_size, image_mean, image_std)

    def normalize(self, image):
        # Bypass normalization, just return the image as is
        return image
    
    def resize(self, image, target):
        # Bypass resizing, just return the image and target as is
        return image, target

# Custom transform class
class CustomRCNNTransform(GeneralizedRCNNTransform):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super().__init__(min_size, max_size, image_mean, image_std)

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]
    

def modify_fasterrcnn_resnet50_fpn_channels(model, num_channels, image_mean, image_std):
    # Get the first convolutional layer from the pre-trained ResNet
    old_conv = model.backbone.body.conv1

    # Create a new Conv2d layer with the desired number of input channels
    # and the same output channels, kernel size, stride, etc., as the old one
    new_conv = torch.nn.Conv2d(num_channels, old_conv.out_channels, 
                               kernel_size=old_conv.kernel_size, 
                               stride=old_conv.stride, 
                               padding=old_conv.padding, 
                               bias=old_conv.bias)

    # Initialize the new convolutional layer with random weights
    torch.nn.init.kaiming_uniform_(new_conv.weight, mode='fan_out', nonlinearity='relu')

    # Replace the first convolutional layer
    model.backbone.body.conv1 = new_conv

    # Modify the transform with new normalization parameters
    original_transform = model.transform
    new_transform = CustomRCNNTransform(
        original_transform.min_size, original_transform.max_size,
        image_mean, image_std
    )

    simple_transform = SimpleTransform(
        original_transform.min_size, original_transform.max_size,
        image_mean, image_std
    )
    model.transform = simple_transform

    return model

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )

    return model



################## Adding the custom two backbone model:

class CombinedMaskRCNN(MaskRCNN):
    def __init__(self, num_classes, pretrained=True):
        # Load two Mask R-CNN models
        model1 = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
        model2 = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')

        # Get the number of input features for the classifiers
        in_features1 = model1.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = model1.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        # Create new predictors
        combined_box_predictor = FastRCNNPredictor(in_features1, num_classes)
        combined_mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        # Initialize a backbone from one of the models
        backbone = model1.backbone

        # Initialize the MaskRCNN with the combined backbone
        super(CombinedMaskRCNN, self).__init__(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=model1.rpn.anchor_generator,
            box_roi_pool=model1.roi_heads.box_roi_pool,
            mask_roi_pool=model1.roi_heads.mask_roi_pool,
            mask_head=model1.roi_heads.mask_head,
        )

        # Replace the predictor heads with the combined predictors
        self.roi_heads.box_predictor = combined_box_predictor
        self.roi_heads.mask_predictor = combined_mask_predictor

        # Store the second model's backbone
        self.model2_backbone = model2.backbone

    def forward(self, images, targets=None):
        # Split the images into two sets of 3 channels
        images1 = [img[:3, :, :] for img in images]
        images2 = [img[3:, :, :] for img in images]

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images1:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images1, targets = self.transform(images1, targets)
        images2, _ = self.transform(images2, targets)

        # Get features from both backbones
        features1 = self.backbone(images1.tensors)
        features2 = self.model2_backbone(images2.tensors)

        # Combine the features by averaging (or any other method)
        combined_features = {key: (features1[key] + features2[key]) / 2 for key in features1}

        # Generate proposals
        proposals, proposal_losses = self.rpn(images1, combined_features, targets)

        # Use the combined features for the ROI heads
        detections, detector_losses = self.roi_heads(combined_features, proposals, images1.image_sizes, targets)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections


