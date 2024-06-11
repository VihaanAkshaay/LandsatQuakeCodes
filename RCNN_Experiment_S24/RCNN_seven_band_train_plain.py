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

from evaluation_utils import bbox_iou, mask_iou, validation_evaluator, evaluate_metrics

# Importing from helper functions
import model_def
import data_utils

if __name__ == "__main__":
    # MAIN CODE STARTS HERE:
    # Setting up stuff & data

    # Adjust the model for one class (landslide) plus background#
    num_classes = 2  # 1 class (landslide) + 1 for background
    model = model_def.get_model_instance_segmentation(num_classes)
    #model = maskrcnn_resnet50_fpn(pretrained=True)

    # Assuming you have a pre-trained Faster R-CNN model
    #model = maskrcnn_resnet50_fpn(pretrained=True)

    # Define the mean and standard deviation for each of the 7 channels
    custom_means = [0.0] * 6  # Replace with your actual mean values
    custom_stds = [1.0] * 6   # Replace with your actual std values

    # Modify the model to accept 7-channel images and use custom normalization
    model = model_def.modify_fasterrcnn_resnet50_fpn_channels(model, 6, custom_means, custom_stds)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Learning rate scheduler - Reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

    # yolo label folders:
    train_yolo_labels = 'train_yolo_labels_3band'
    val_yolo_labels = 'val_yolo_labels_3band'
    test_yolo_labels = 'test_yolo_labels_3band'

    # Target npy folders:
    train_numpy_images = 'train_numpy_images_3band'
    val_numpy_images = 'val_numpy_images_3band'
    test_numpy_images = 'test_numpy_images_3band'

    # Instantiate datasets for each split
    train_dataset = data_utils.MaskRCNNDataset(train_numpy_images, train_yolo_labels, custom_means, custom_stds)
    val_dataset = data_utils.MaskRCNNDataset(val_numpy_images, val_yolo_labels, custom_means, custom_stds)
    test_dataset = data_utils.MaskRCNNDataset(test_numpy_images, test_yolo_labels, custom_means, custom_stds)

    # Create data loaders for each dataset
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_utils.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=data_utils.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=data_utils.collate_fn)

    num_epochs = 20  # Adjust as needed
    patience = 30  # Number of epochs to wait after last time validation loss improved.
    best_val_loss = np.inf
    epochs_since_improvement = 0

    train_losses = []
    val_losses = []

    # ... [previous setup code]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)  # Added weight decay for regularization

    # Learning rate scheduler - Reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

    # Early stopping parameters
    best_val_loss = np.inf
    epochs_since_improvement = 0
    delta = 0.001  # Minimum change in the monitored quantity to qualify as an improvement

    for epoch in range(num_epochs):
        # Training phase
        metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        train_total_loss = metric_logger.meters['loss'].global_avg
        train_losses.append(train_total_loss)

        # Validation phase
        val_loss_dict = validation_evaluator(model, val_loader, device)
        val_total_loss = val_loss_dict['total_loss']
        val_losses.append(val_total_loss)

        # Learning rate scheduler step
        scheduler.step(val_total_loss)

        # Print summary at the end of each epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_total_loss:.4f}, Validation Loss: {val_total_loss:.4f}")

        # Calculate the validation metrics
        results = evaluate_metrics(model, val_loader, device) 
        mean_bbox_ap = np.mean(list(results['bbox_AP'].values()))
        #mean_mask_ap = np.mean(list(results['mask_AP'].values()))

        print(f"Validation Metrics:")
        # Print the Mean BBoox AP and AP50, AP75 without decimal limit
        print(f" - Mean BBox AP: {mean_bbox_ap}")
        print(f" - BBox AP50: {results['bbox_AP50']}")
        print(f" - BBox AP75: {results['bbox_AP75']}")
        #print(f" - Mean Mask AP: {mean_mask_ap:.4f}")
        #print(f" - Mask AP50: {results['mask_AP50']:.4f}")
        #print(f" - Mask AP75: {results['mask_AP75']:.4f}")


        # Early stopping check
        if epoch <= 50:
            continue

        # Check for improvement
        improvement = best_val_loss - val_total_loss > delta
        if val_total_loss < best_val_loss and improvement:
            best_val_loss = val_total_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1


        if epochs_since_improvement >= patience:
            print(f"No significant improvement in validation loss for {patience} consecutive epochs. Stopping training.")
            break

    # ... [rest of the code]

    
    # Save the model after training is complete
    torch.save(model.state_dict(), 'final_model_RCNN.pth')

    # Testing the model on different datasets (test, train, validation)
    datasets = {'Test': test_loader, 'Train': train_loader, 'Validation': val_loader}

    print('Starting evaluation on each dataset')

    # Evaluate on each dataset
    for dataset_name, loader in datasets.items():
        results = evaluate_metrics(model, loader, device)
        
        # Calculate mean AP (mAP)
        mean_bbox_ap = np.mean(list(results['bbox_AP'].values()))
        mean_mask_ap = np.mean(list(results['mask_AP'].values()))

        print(f'{dataset_name} Dataset Evaluation:')
        print(f" - Mean BBox AP: {mean_bbox_ap:.4f}")
        print(f" - BBox AP50: {results['bbox_AP50']:.4f}")
        print(f" - BBox AP75: {results['bbox_AP75']:.4f}")
        print(f" - Mean Mask AP: {mean_mask_ap:.4f}")
        print(f" - Mask AP50: {results['mask_AP50']:.4f}")
        print(f" - Mask AP75: {results['mask_AP75']:.4f}")
        print()

        # Save the results to a text file for the Test dataset
        if dataset_name == 'Test':
            with open('RCNN_test_results.txt', 'w') as file:
                file.write(f"{dataset_name} Dataset Evaluation:\n")
                file.write(f" - Mean BBox AP: {mean_bbox_ap:.4f}\n")
                file.write(f" - BBox AP50: {results['bbox_AP50']:.4f}\n")
                file.write(f" - BBox AP75: {results['bbox_AP75']:.4f}\n")
                file.write(f" - Mean Mask AP: {mean_mask_ap:.4f}\n")
                file.write(f" - Mask AP50: {results['mask_AP50']:.4f}\n")
                file.write(f" - Mask AP75: {results['mask_AP75']:.4f}\n")



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





