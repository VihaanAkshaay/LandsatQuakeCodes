#Imports

import os

import numpy as np
import matplotlib.pyplot as plt

import torch

import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn

import models
import pickle
import evalmetrics

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = [f for f in os.listdir(image_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx])

        image = np.load(image_path)
        mask = np.load(mask_path)

        # Convert numpy arrays to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32) # Assuming masks are integer labels

        return image, mask

image_dir = '/mnt/data4/vihaan/semantic_filtered_data/images/'
mask_dir = '/mnt/data4/vihaan/semantic_filtered_data/masks/'

print("normal stuff done")
dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir)

# Splitting the dataset into train, val, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)


## LOADING EUNET ######################## Training model on train dataset #####################################
#model = models.U_Net(img_ch=3,output_ch=1)
model = models.EdgeU1_Net(img_ch=3,output_ch=1)
        
# Defining  loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 30

# Set the device to use for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# Train the model
model.to(device)

#Tracking Losses
losst_list = []
lossv_list = []

for epoch in range(num_epochs):
    train_loss = 0.0
    
    #Train the model
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        #print(inputs.shape)
        labels = labels.to(device).unsqueeze(1) 
        #print('labels are')
        #print(labels.shape)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # Print statistics
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    losst_list.append(avg_train_loss)
            
    #Validate the training process
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1) 
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
    lossv_list.append(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Checking with metrics, we use cpu as device

device = torch.device("cpu")
model.to(device)

# Initialize accumulators for each metric
sensitivity_sum = 0
specificity_sum = 0
precision_sum = 0
F1_sum = 0
JS_sum = 0
DC_sum = 0
num_batches = 0

# Loop over the test dataset
for data, labels in test_loader:
    # Perform inference
    
    
    data = data.to(device)
    labels = labels.to(device)
    preds = model(data)

    # Evaluate the predictions
    sensitivity = evalmetrics.get_sensitivity(preds, labels)
    specificity = evalmetrics.get_specificity(preds, labels)
    precision = evalmetrics.get_precision(preds, labels)
    F1 = evalmetrics.get_F1(preds, labels)
    JS = evalmetrics.get_JS(preds, labels)
    DC = evalmetrics.get_DC(preds, labels)

    # Accumulate the metric values
    sensitivity_sum += sensitivity * len(data)
    specificity_sum += specificity * len(data)
    precision_sum += precision * len(data)
    F1_sum += F1 * len(data)
    JS_sum += JS * len(data)
    DC_sum += DC * len(data)
    num_batches += 1

# Compute the average of each metric
sensitivity_avg = sensitivity_sum / len(test_loader.dataset)
specificity_avg = specificity_sum / len(test_loader.dataset)
precision_avg = precision_sum / len(test_loader.dataset)
F1_avg = F1_sum / len(test_loader.dataset)
JS_avg = JS_sum / len(test_loader.dataset)
DC_avg = DC_sum / len(test_loader.dataset)

# Print the average of each metric
print("Average Sensitivity: {:.4f}".format(sensitivity_avg))
print("Average Specificity: {:.4f}".format(specificity_avg))
print("Average Precision: {:.4f}".format(precision_avg))
print("Average F1 Score: {:.4f}".format(F1_avg))
print("Average Jaccard Similarity: {:.4f}".format(JS_avg))
print("Average Dice Coefficient: {:.4f}".format(DC_avg))

with open('metrics.txt', 'a') as f:
    f.write(f"{model.__class__.__name__}, {sensitivity_avg:.4f}, {specificity_avg:.4f}, {precision_avg:.4f}, {F1_avg:.4f}, {JS_avg:.4f}, {DC_avg:.4f}\n")


# Save model weights
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    # add any other relevant information
}

torch.save(checkpoint, 'eunet_trained.pth')

# Open a file for writing
with open('train_losses_eunet.pkl', 'wb') as f:
    # Use pickle to dump the list to the file
    pickle.dump(losst_list, f)
    

# Open a file for writing
with open('val_losses_eunet.pkl', 'wb') as f:
    # Use pickle to dump the list to the file
    pickle.dump(lossv_list, f)

torch.cuda.empty_cache()


## LOADING UNET ######################## Training model on train dataset #####################################
#model = models.U_Net(img_ch=3,output_ch=1)
model = models.U_Net(img_ch=3,output_ch=1)
        
# Defining  loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 30

# Set the device to use for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Train the model
model.to(device)

#Tracking Losses
losst_list = []
lossv_list = []

for epoch in range(num_epochs):
    train_loss = 0.0
    
    #Train the model
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        #print(inputs.shape)
        labels = labels.to(device).unsqueeze(1) 
        #print('labels are')
        #print(labels.shape)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # Print statistics
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    losst_list.append(avg_train_loss)
            
    #Validate the training process
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1) 
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
    lossv_list.append(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Checking with metrics
device = torch.device("cpu")
model.to(device)

# Initialize accumulators for each metric
sensitivity_sum = 0
specificity_sum = 0
precision_sum = 0
F1_sum = 0
JS_sum = 0
DC_sum = 0
num_batches = 0

# Loop over the test dataset
for data, labels in test_loader:
    # Perform inference
    
    
    data = data.to(device)
    labels = labels.to(device)
    preds = model(data)

    # Evaluate the predictions
    sensitivity = evalmetrics.get_sensitivity(preds, labels)
    specificity = evalmetrics.get_specificity(preds, labels)
    precision = evalmetrics.get_precision(preds, labels)
    F1 = evalmetrics.get_F1(preds, labels)
    JS = evalmetrics.get_JS(preds, labels)
    DC = evalmetrics.get_DC(preds, labels)

    # Accumulate the metric values
    sensitivity_sum += sensitivity * len(data)
    specificity_sum += specificity * len(data)
    precision_sum += precision * len(data)
    F1_sum += F1 * len(data)
    JS_sum += JS * len(data)
    DC_sum += DC * len(data)
    num_batches += 1

# Compute the average of each metric
sensitivity_avg = sensitivity_sum / len(test_loader.dataset)
specificity_avg = specificity_sum / len(test_loader.dataset)
precision_avg = precision_sum / len(test_loader.dataset)
F1_avg = F1_sum / len(test_loader.dataset)
JS_avg = JS_sum / len(test_loader.dataset)
DC_avg = DC_sum / len(test_loader.dataset)

# Print the average of each metric
print("Average Sensitivity: {:.4f}".format(sensitivity_avg))
print("Average Specificity: {:.4f}".format(specificity_avg))
print("Average Precision: {:.4f}".format(precision_avg))
print("Average F1 Score: {:.4f}".format(F1_avg))
print("Average Jaccard Similarity: {:.4f}".format(JS_avg))
print("Average Dice Coefficient: {:.4f}".format(DC_avg))

with open('metrics.txt', 'a') as f:
    f.write(f"{model.__class__.__name__}, {sensitivity_avg:.4f}, {specificity_avg:.4f}, {precision_avg:.4f}, {F1_avg:.4f}, {JS_avg:.4f}, {DC_avg:.4f}\n")


# Save model weights
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    # add any other relevant information
}

torch.save(checkpoint, 'unet_trained.pth')

# Open a file for writing
with open('train_losses_unet.pkl', 'wb') as f:
    # Use pickle to dump the list to the file
    pickle.dump(losst_list, f)
    

# Open a file for writing
with open('val_losses_unet.pkl', 'wb') as f:
    # Use pickle to dump the list to the file
    pickle.dump(lossv_list, f)

torch.cuda.empty_cache()




## LOADING AUNET ######################## Training model on train dataset #####################################
#model = models.U_Net(img_ch=3,output_ch=1)
model = models.AttU_Net(img_ch=3,output_ch=1)
        
# Defining  loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 30

# Set the device to use for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Train the model
model.to(device)

#Tracking Losses
losst_list = []
lossv_list = []

for epoch in range(num_epochs):
    train_loss = 0.0
    
    #Train the model
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        #print(inputs.shape)
        labels = labels.to(device).unsqueeze(1) 
        #print('labels are')
        #print(labels.shape)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # Print statistics
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    losst_list.append(avg_train_loss)
            
    #Validate the training process
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1) 
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
    lossv_list.append(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


# Checking with metrics
device = torch.device("cpu")
model.to(device)

# Initialize accumulators for each metric
sensitivity_sum = 0
specificity_sum = 0
precision_sum = 0
F1_sum = 0
JS_sum = 0
DC_sum = 0
num_batches = 0

# Loop over the test dataset
for data, labels in test_loader:
    # Perform inference
    
    
    data = data.to(device)
    labels = labels.to(device)
    preds = model(data)

    # Evaluate the predictions
    sensitivity = evalmetrics.get_sensitivity(preds, labels)
    specificity = evalmetrics.get_specificity(preds, labels)
    precision = evalmetrics.get_precision(preds, labels)
    F1 = evalmetrics.get_F1(preds, labels)
    JS = evalmetrics.get_JS(preds, labels)
    DC = evalmetrics.get_DC(preds, labels)

    # Accumulate the metric values
    sensitivity_sum += sensitivity * len(data)
    specificity_sum += specificity * len(data)
    precision_sum += precision * len(data)
    F1_sum += F1 * len(data)
    JS_sum += JS * len(data)
    DC_sum += DC * len(data)
    num_batches += 1

# Compute the average of each metric
sensitivity_avg = sensitivity_sum / len(test_loader.dataset)
specificity_avg = specificity_sum / len(test_loader.dataset)
precision_avg = precision_sum / len(test_loader.dataset)
F1_avg = F1_sum / len(test_loader.dataset)
JS_avg = JS_sum / len(test_loader.dataset)
DC_avg = DC_sum / len(test_loader.dataset)

# Print the average of each metric
print("Average Sensitivity: {:.4f}".format(sensitivity_avg))
print("Average Specificity: {:.4f}".format(specificity_avg))
print("Average Precision: {:.4f}".format(precision_avg))
print("Average F1 Score: {:.4f}".format(F1_avg))
print("Average Jaccard Similarity: {:.4f}".format(JS_avg))
print("Average Dice Coefficient: {:.4f}".format(DC_avg))

# Write these average metrics to a file with model name
with open('metrics.txt', 'a') as f:
    f.write(f"{model.__class__.__name__}, {sensitivity_avg:.4f}, {specificity_avg:.4f}, {precision_avg:.4f}, {F1_avg:.4f}, {JS_avg:.4f}, {DC_avg:.4f}\n")


# Save model weights
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    # add any other relevant information
}

torch.save(checkpoint, 'unet_trained.pth')

# Open a file for writing
with open('train_losses_unet.pkl', 'wb') as f:
    # Use pickle to dump the list to the file
    pickle.dump(losst_list, f)
    

# Open a file for writing
with open('val_losses_unet.pkl', 'wb') as f:
    # Use pickle to dump the list to the file
    pickle.dump(lossv_list, f)

torch.cuda.empty_cache()

# Save model weights
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    # add any other relevant information
}

torch.save(checkpoint, 'aunet_trained.pth')

# Open a file for writing
with open('train_losses_aunet.pkl', 'wb') as f:
    # Use pickle to dump the list to the file
    pickle.dump(losst_list, f)
    

# Open a file for writing
with open('val_losses_aunet.pkl', 'wb') as f:
    # Use pickle to dump the list to the file
    pickle.dump(lossv_list, f)

