# This is a good reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import torch
from torch.utils.data import DataLoader
import numpy as np
import data_utils  # Replace with your actual data_utils module
import model_def  # Replace with your actual model definition module

from engine import evaluate

# Importing from helper functions
import model_def
import data_utils


# Function to load the modified model
def load_model(model_path, num_classes, custom_means, custom_stds, device):
    # Initialize the model
    model = model_def.get_model_instance_segmentation(num_classes)
    # Modify the model to accept 7-channel images
    model = model_def.modify_fasterrcnn_resnet50_fpn_channels(model, 7, custom_means, custom_stds)
    # Load the pre-trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

# Function to evaluate the model
def evaluate_model(model, data_loader, device):
    # Assuming you have an evaluate_metrics function similar to the one you used before
    results = evaluate(model, data_loader, device)
    return results

if __name__ == "__main__":
    # Model and dataset setup
    model_path = 'final_model_RCNN.pth'

    # yolo label folders:
    train_yolo_labels = 'train_yolo_labels'
    val_yolo_labels = 'val_yolo_labels'
    test_yolo_labels = 'test_yolo_labels'

    # Target npy folders:
    train_numpy_images = 'train_numpy_images'
    val_numpy_images = 'val_numpy_images'
    test_numpy_images = 'test_numpy_images'

    num_classes = 2  # 1 class (landslide) + background
    custom_means = [0.0] * 7  # Replace with your actual mean values
    custom_stds = [1.0] * 7   # Replace with your actual std values
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the model
    model = load_model(model_path, num_classes, custom_means, custom_stds, device)

    # Setup the test dataset and loader
    test_dataset = data_utils.MaskRCNNDataset(test_numpy_images, test_yolo_labels)  # Modify as per your dataset handling
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=data_utils.collate_fn)

    # Evaluate the model
    test_results = evaluate_model(model, test_loader, device)
    print(test_results)

    # Print the evaluation results
    print("Test Dataset Evaluation:")
    # Calculate mean AP (mAP)
    mean_bbox_ap = np.mean(list(test_results['bbox_AP'].values()))
    mean_mask_ap = np.mean(list(test_results['mask_AP'].values()))

    print("Test Dataset Evaluation:")
    print(f" - Mean BBox AP: {mean_bbox_ap:.4f}")
    print(f" - BBox AP50: {test_results['bbox_AP50']:.4f}")
    print(f" - BBox AP75: {test_results['bbox_AP75']:.4f}")
    print(f" - Mean Mask AP: {mean_mask_ap:.4f}")
    print(f" - Mask AP50: {test_results['mask_AP50']:.4f}")
    print(f" - Mask AP75: {test_results['mask_AP75']:.4f}")

