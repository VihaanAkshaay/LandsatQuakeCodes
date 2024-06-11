import os

# label paths:
train_label_path = 'train_yolo_labels_S24'
val_label_path = 'val_yolo_labels_S24'
test_label_path = 'test_yolo_labels_S24'


# Check if any files are empty:
def check_empty_files(folder_path):
    for f in os.listdir(folder_path):
        if os.path.getsize(os.path.join(folder_path, f)) == 0:
            print(f"Empty file found: {f}")
