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


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    targets = [{k: torch.as_tensor(v) for k, v in t.items()} for t in targets]
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

# Helper function to load and preprocess images
def load_and_preprocess_image(image_path, mean, std):
    # Load image
    image = np.load(image_path, allow_pickle=True).astype(np.float32)
    image = torch.from_numpy(image)  # Convert to torch tensor

    # Normalize the image
    for i in range(image.shape[0]):  # Assuming image has shape [C, H, W]
        image[i] = (image[i] - mean[i]) / std[i]  # Corrected to use lowercase 'i'
    return image

class MaskRCNNDataset(Dataset):
    def __init__(self, image_dir, label_dir, mean, std):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = os.listdir(image_dir)
        self.mean = mean  # Mean for normalization
        self.std = std    # Std for normalization

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = load_and_preprocess_image(image_path, self.mean, self.std)

        label_path = os.path.join(self.label_dir, self.images[idx].replace('.npy', '.txt'))
        boxes, labels, masks = self.parse_labels(label_path, image.shape[2], image.shape[1])  # Image shape: [C, H, W]

        if len(boxes) == 0:
            # Handle images with no objects
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

        target = {
            'boxes': boxes,
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': torch.as_tensor(masks, dtype=torch.uint8),
            'image_id': torch.tensor([idx])
        }

        #print("Mask shape:", masks.shape)  # Debug output
        return image, target

    def parse_labels(self, label_path, img_width, img_height):
        boxes = []
        labels = []
        masks = []
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id = 1  # Assuming class 1 is for objects

                if len(parts) > 1:  # Check if there are coordinates provided
                    poly_coords = list(map(float, parts[1:]))
                    poly_coords = np.array(poly_coords).reshape(-1, 2)
                    poly_coords[:, 0] *= img_width
                    poly_coords[:, 1] *= img_height

                    x_min = np.min(poly_coords[:, 0])
                    y_min = np.min(poly_coords[:, 1])
                    x_max = np.max(poly_coords[:, 0])
                    y_max = np.max(poly_coords[:, 1])
                    boxes.append([x_min, y_min, x_max, y_max])

                    labels.append(class_id)

                    mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    rr, cc = polygon(poly_coords[:, 1], poly_coords[:, 0], mask.shape)
                    mask[rr, cc] = 1
                    masks.append(mask)
                else:
                    # Return an empty mask and empty box if no objects are found
                    masks = np.zeros((0, img_height, img_width), dtype=np.uint8)

        return boxes, labels, np.array(masks, dtype=np.uint8) if masks else np.zeros((0, img_height, img_width), dtype=np.uint8)

