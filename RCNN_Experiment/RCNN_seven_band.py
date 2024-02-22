# This is a good reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

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



# Note: Ensure that the functions bbox_iou and mask_iou (if used) are defined in your code.



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

class MaskRCNNDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = np.load(image_path, allow_pickle=True)
        image = torch.from_numpy(image).float()

        img_height, img_width = image.shape[1], image.shape[2]

        label_path = os.path.join(self.label_dir, self.images[idx].replace('.npy', '.txt'))
        boxes, labels, masks = self.parse_labels(label_path, img_width, img_height)
        
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

        return image, target

    def parse_labels(self, label_path, img_width, img_height):
        boxes = []
        labels = []
        masks = []
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                #class_id = int(parts[0]) # which is always 0 in our annotation, but we want it to be 1 (Since RCNN background is 0)
                class_id = 1

                # Parse polygon coordinates
                poly_coords = list(map(float, parts[1:]))
                poly_coords = np.array(poly_coords).reshape(-1, 2)
                poly_coords[:, 0] *= img_width
                poly_coords[:, 1] *= img_height

                # Calculate bounding box from polygon
                x_min = np.min(poly_coords[:, 0])
                y_min = np.min(poly_coords[:, 1])
                x_max = np.max(poly_coords[:, 0])
                y_max = np.max(poly_coords[:, 1])
                boxes.append([x_min, y_min, x_max, y_max])

                labels.append(class_id)

                # Create a binary mask
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                rr, cc = polygon(poly_coords[:, 1], poly_coords[:, 0], mask.shape)
                mask[rr, cc] = 1
                masks.append(mask)

        return boxes, labels, np.array(masks)
    

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

'''

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    for images, targets in tqdm(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        losses.backward()
        optimizer.step()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    return total_loss / len(data_loader)

'''
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


'''
def evaluate_metrics(model, data_loader, device):
    model.eval()
    #ious = []
    precisions = []
    recalls = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                gt_boxes = targets[i]['boxes'].to(device)
                pred_boxes = output['boxes']
                pred_scores = output['scores']
                pred_labels = output['labels']

                # Filter out predictions with low scores (e.g., < 0.5)
                threshold = 0.25
                high_scores_idx = pred_scores > threshold
                pred_boxes = pred_boxes[high_scores_idx]
                pred_labels = pred_labels[high_scores_idx]

                # Initialize TP, FP, and FN counters
                TP = FP = FN = 0

                for pred_box in pred_boxes:
                    # Calculate IoUs with ground truth boxes
                    ious = [bbox_iou(pred_box, gt_box) for gt_box in gt_boxes]

                    # Match prediction to ground truth
                    if len(ious) > 0 and max(ious) > threshold:
                        TP += 1
                    else:
                        FP += 1

                FN = len(gt_boxes) - TP

                if TP + FP > 0:
                    precision = TP / (TP + FP)
                    precisions.append(precision)
                if TP + FN > 0:
                    recall = TP / (TP + FN)
                    recalls.append(recall)


    # Calculate average metrics
    avg_precision = sum(precisions) / len(precisions) if len(precisions) > 0 else 0
    avg_recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 0
    #avg_iou = sum(ious) / len(ious) if len(ious) > 0 else 0

    return avg_iou, avg_precision, avg_recall
'''

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

def evaluate_metrics(model, data_loader, device, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    model.eval()
    all_precisions = defaultdict(list)
    all_recalls = defaultdict(list)
    all_map = defaultdict(list)

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

                # Filter out predictions with low scores (Objectiveness score from the model, we threshold it to 0.3)
                threshold = 0.3
                high_scores_idx = pred_scores > threshold
                pred_boxes = pred_boxes[high_scores_idx]
                pred_masks = pred_masks[high_scores_idx]

                # Match predictions to ground truth
                for iou_threshold in iou_thresholds:
                    matched_gt = []
                    TP = FP = FN = 0

                    for pred_box, pred_mask in zip(pred_boxes, pred_masks):
                        max_iou = 0
                        max_iou_index = -1

                        for j, (gt_box, gt_mask) in enumerate(zip(gt_boxes, gt_masks)):
                            iou = bbox_iou(pred_box, gt_box)  # Use mask IoU for mask-based evaluation
                            if iou > max_iou and iou > iou_threshold and j not in matched_gt:
                                max_iou = iou
                                max_iou_index = j

                        if max_iou_index != -1:
                            matched_gt.append(max_iou_index)
                            TP += 1
                        else:
                            FP += 1

                    FN = len(gt_boxes) - len(matched_gt) # Model predicts no landslide, but there is one

                    if FN < 0:
                        FN = 0 # If there are more predictions than ground truth boxes

                    if TP + FP > 0:
                        precision = TP / (TP + FP)
                        all_precisions[iou_threshold].append(precision)
                    if TP + FN > 0:
                        recall = TP / (TP + FN)
                        all_recalls[iou_threshold].append(recall)

                    # Compute AP for this threshold
                    if len(pred_scores) > 0:
                    # Ensure scores are on CPU and converted to numpy for processing
                        pred_scores_np = pred_scores.cpu().numpy()

                        # Sorting using NumPy
                        sorted_indices_np = np.argsort(-pred_scores_np)

                        # Get the sorted scores in numpy format
                        sorted_scores_np = pred_scores_np[sorted_indices_np]

                        # Using NumPy's cumsum function
                        sorted_TP = np.cumsum(sorted_scores_np > iou_threshold)

                        sorted_FP = np.cumsum(sorted_scores_np < iou_threshold)
                        recall_curve = sorted_TP / (TP + FN)
                        precision_curve = sorted_TP / (sorted_TP + sorted_FP)
                        all_map[iou_threshold].append(np.trapz(precision_curve, recall_curve))

    # Calculate average metrics
    avg_precision = {iou: np.mean(all_precisions[iou]) for iou in iou_thresholds}
    avg_recall = {iou: np.mean(all_recalls[iou]) for iou in iou_thresholds}
    avg_map = {iou: np.mean(all_map[iou]) for iou in iou_thresholds}

    return avg_precision, avg_recall, avg_map



# MAIN CODE STARTS HERE:
# Setting up stuff & data

# Adjust the model for one class (landslide) plus background#
num_classes = 2  # 1 class (landslide) + 1 for background
model = get_model_instance_segmentation(num_classes)
#model = maskrcnn_resnet50_fpn(pretrained=True)

# Assuming you have a pre-trained Faster R-CNN model
#model = maskrcnn_resnet50_fpn(pretrained=True)

# Define the mean and standard deviation for each of the 7 channels
custom_means = [0.0] * 7  # Replace with your actual mean values
custom_stds = [1.0] * 7   # Replace with your actual std values

# Modify the model to accept 7-channel images and use custom normalization
model = modify_fasterrcnn_resnet50_fpn_channels(model, 7, custom_means, custom_stds)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# yolo label folders:
train_yolo_labels = 'train_yolo_labels'
val_yolo_labels = 'val_yolo_labels'
test_yolo_labels = 'test_yolo_labels'

# Target npy folders:
train_numpy_images = 'train_numpy_images'
val_numpy_images = 'val_numpy_images'
test_numpy_images = 'test_numpy_images'

# Instantiate datasets for each split
train_dataset = MaskRCNNDataset(train_numpy_images, train_yolo_labels)
val_dataset = MaskRCNNDataset(val_numpy_images, val_yolo_labels)
test_dataset = MaskRCNNDataset(test_numpy_images, test_yolo_labels)

# Create data loaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

'''
#transform = T.Compose([T.ToPILImage(), T.Resize((800, 800)), T.ToTensor()])

def split_dataset(dataset, train_size=0.7, val_size=0.15, test_size=0.15):
    # Generate indices: an array of integers from 0 to the length of the dataset
    indices = np.arange(len(dataset))
    
    # Split indices into train and temporary test+val set
    train_indices, test_val_indices = train_test_split(indices, test_size=(1 - train_size))
    
    # Split the test+val set into test and validation sets
    val_indices, test_indices = train_test_split(test_val_indices, test_size=(test_size / (test_size + val_size)))

    # Create data subsets
    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)
    test_data = Subset(dataset, test_indices)

    return train_data, val_data, test_data

# Create dataset instance
dataset = MaskRCNNDataset(images_dir, labels_dir)

# Split the dataset
train_data, val_data, test_data = split_dataset(dataset)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=collate_fn)
'''


# Training loop:

'''
num_epochs = 2  # You can adjust this
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    print('train loss type',type(train_loss))
    #val_loss = evaluate(model, val_loader, device)
    # Evaluate on Val data:
    val_loss_dict = validation_evaluator(model, val_loader, device)
    val_loss_str = ', '.join(f"{k}: {val_loss_dict[k]:.4f}" for k in val_loss_dict)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss_str}")

    #print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

    
    # Convergence check (you may want to refine this criterion)
    if epoch > 0 and abs(prev_val_loss - val_loss) < 0.01:
        print("Model has converged")
        break

    prev_val_loss = val_loss
    

# Testing the model:
avg_iou, avg_precision, avg_recall = evaluate_metrics(model, test_loader, device)
print(f"Average IoU: {avg_iou}, Precision: {avg_precision}, Recall: {avg_recall}")

'''


num_epochs = 100  # Adjust as needed
patience = 30  # Number of epochs to wait after last time validation loss improved.
best_val_loss = np.inf
epochs_since_improvement = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training phase
    metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    train_total_loss = metric_logger.meters['loss'].global_avg
    train_losses.append(train_total_loss)

    # Validation phase
    val_loss_dict = validation_evaluator(model, val_loader, device)
    val_total_loss = val_loss_dict['total_loss']
    val_losses.append(val_total_loss)

    print(f"Epoch {epoch+1}, Train Loss: {train_total_loss}, Validation Loss: {val_total_loss}")

    # Check for improvement
    if val_total_loss < best_val_loss:
        best_val_loss = val_total_loss
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1

    # Early stopping check
    if epochs_since_improvement >= patience:
        print(f"No improvement in validation loss for {patience} consecutive epochs. Stopping training.")
        break


# Testing the model on different data sets (test, train, validation)
datasets = {'Test': test_loader, 'Train': train_loader, 'Validation': val_loader}

# Evaluate on each dataset
for dataset_name, loader in datasets.items():
    avg_precision, avg_recall, avg_map = evaluate_metrics(model, loader, device)
    
    print(f'{dataset_name}:')
    for iou_threshold in avg_precision.keys():
        print(f"IoU Threshold: {iou_threshold:.2f}")
        print(f" - Average Precision: {avg_precision[iou_threshold]:.4f}")
        print(f" - Average Recall: {avg_recall[iou_threshold]:.4f}")
        print(f" - Average mAP: {avg_map[iou_threshold]:.4f}")
    print()

# Store These values for test dataset on a txt file:
avg_precision_test, avg_recall_test, avg_map_test = evaluate_metrics(model, test_loader, device)

# Save the results to a text file
with open('RCNN_test_results.txt', 'w') as file:
    for iou_threshold in avg_precision_test.keys():
        file.write(f"IoU Threshold: {iou_threshold:.2f}\n")
        file.write(f" - Average Precision: {avg_precision_test[iou_threshold]:.4f}\n")
        file.write(f" - Average Recall: {avg_recall_test[iou_threshold]:.4f}\n")
        file.write(f" - Average mAP: {avg_map_test[iou_threshold]:.4f}\n\n")

# Plotting the Losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()

# Save the figure
plt.savefig('training_validation_loss_RCNN_plot.png', format='png')

# Optionally display the plot as well
plt.show()

# Save the model after training is complete
torch.save(model.state_dict(), 'final_model_RCNN.pth')


