import os
import numpy as np

# Add path of all images and labels:
image_full_path = '/Users/vihaan/Workspace/!Datasets/Processed_Data_S24/image_patches'
label_full_path = '/Users/vihaan/Workspace/!Datasets/Processed_Data_S24/labels'

# Output paths:
train_image_path = 'train_numpy_images_S24'
train_label_path = 'train_yolo_labels_S24'
val_image_path = 'val_numpy_images_S24'
val_label_path = 'val_yolo_labels_S24'
test_image_path = 'test_numpy_images_S24'
test_label_path = 'test_yolo_labels_S24'

# Clear all the output paths:
os.system(f'rm -rf {train_image_path}')
os.system(f'rm -rf {train_label_path}')
os.system(f'rm -rf {val_image_path}')
os.system(f'rm -rf {val_label_path}')
os.system(f'rm -rf {test_image_path}')
os.system(f'rm -rf {test_label_path}')

# Create output directories:
os.makedirs(train_image_path)
os.makedirs(train_label_path)
os.makedirs(val_image_path)
os.makedirs(val_label_path)
os.makedirs(test_image_path)
os.makedirs(test_label_path)

# Function to handle file copying safely with spaces in file paths
def safe_copy(src_path, dest_path, filename):
    src_file = os.path.join(src_path, filename)
    dest_file = os.path.join(dest_path, filename)
    command = f'cp "{src_file}" "{dest_file}"'  # Notice the quotation marks to handle spaces
    os.system(command)

# Function to remove extensions and create a mapping
def create_mapping(folder_path, extension):
    files = [f for f in os.listdir(folder_path) if f.endswith(extension)]
    return {os.path.splitext(f)[0]: f for f in files}

# Create mappings
image_mapping = create_mapping(image_full_path, '.npy')
label_mapping = create_mapping(label_full_path, '.txt')

# Creating data by matching image and label files
data = []
for base_name in image_mapping:
    if base_name in label_mapping:
        data.append((image_mapping[base_name], label_mapping[base_name]))

# Shuffle data to ensure random split
np.random.shuffle(data)

# Split data
train_ratio = 0.8
val_ratio = 0.1
train_index = int(len(data) * train_ratio)
val_index = int(len(data) * (train_ratio + val_ratio))
train_data = data[:train_index]
val_data = data[train_index:val_index]
test_data = data[val_index:]

# Copy files to respective directories
for image, label in train_data:
    safe_copy(image_full_path, train_image_path, image)
    safe_copy(label_full_path, train_label_path, label)

for image, label in val_data:
    safe_copy(image_full_path, val_image_path, image)
    safe_copy(label_full_path, val_label_path, label)

for image, label in test_data:
    safe_copy(image_full_path, test_image_path, image)
    safe_copy(label_full_path, test_label_path, label)

print('Data split successfully!')
