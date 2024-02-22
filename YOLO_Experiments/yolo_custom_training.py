from ultralytics import YOLO

# Load a model
model = YOLO('model_weights/yolov8x-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
#results = model.train(data='landslide.yaml', epochs=100, imgsz=640)
results = model.train(project='project_folder',data = 'yaml_files/landslide-handpicked.yaml', epochs=50, imgsz=640)


##### Testing the model performance on test images #####

# Loading the best model
model_new = YOLO('project_folder/train/weights/best.pt')  

# Now we obtain: images, images_truth, images_predicted.
path_for_inference = 'datasets/landslide-seg-handpicked/images/test'

# Run inference on a folder of images and generate the predictions of the model on test images:
model.val()  # set model to inference mode (recommended for faster inference)
results = model_new(path_for_inference, save_dir='project_folder/test/predictions')  # run inference on all images in folder

print(results)  # print results to screen
