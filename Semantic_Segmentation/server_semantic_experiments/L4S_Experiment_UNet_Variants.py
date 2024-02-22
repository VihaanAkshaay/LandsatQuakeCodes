 # Imports

import os
import numpy as np

import os
import h5py
import torch
from torch.utils.data import Dataset,DataLoader
import utils
import models
import evalmetrics
import torch.nn as nn

# Data on the server is stored in /mnt/data2/vihaan/L4S_data/
path = '/mnt/data2/vihaan/L4S_data/TrainData/'

# Splitting the train folder into train, validation and test folders
full_dataset = utils.CustomDataset_L4S_Split_Train_Test_Val(img_dir=path + 'img', mask_dir=path + 'mask')

train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_list = [models.EdgeU1_Net(img_ch=14,output_ch=1), models.AttU_Net(img_ch=14,output_ch=1), models.U_Net(img_ch=14,output_ch=1)]



for model in model_list:

    # Preparing the model for training
    # Train the model
    model.to(device)

    #Tracking Losses
    losst_list = []
    lossv_list = []

    # Defining  loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 300


    for epoch in range(num_epochs):
        train_loss = 0.0
        
        #Train the model
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            
            inputs = inputs.float()
            inputs = inputs.to(device)
            #print(inputs.shape)
            labels = labels.to(device)
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
                inputs = inputs.float()
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
        lossv_list.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        # Write the losses and corresponding model name to a file
        with open('losses.txt', 'a') as f:
            f.write(f"{model.__class__.__name__}, {avg_train_loss:.4f}, {avg_val_loss:.4f}\n")

        # Evaluate the metrics
        #print('Evaluating the metrics')

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
        
        data = data.float()
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
    sensitivity_avg_u = sensitivity_sum / len(test_loader.dataset)
    specificity_avg_u = specificity_sum / len(test_loader.dataset)
    precision_avg_u = precision_sum / len(test_loader.dataset)
    F1_avg_u = F1_sum / len(test_loader.dataset)
    JS_avg_u = JS_sum / len(test_loader.dataset)
    DC_avg_u = DC_sum / len(test_loader.dataset)

    # Print the average of each metric
    print("Average Sensitivity: {:.4f}".format(sensitivity_avg_u))
    print("Average Specificity: {:.4f}".format(specificity_avg_u))
    print("Average Precision: {:.4f}".format(precision_avg_u))
    print("Average F1 Score: {:.4f}".format(F1_avg_u))
    print("Average Jaccard Similarity: {:.4f}".format(JS_avg_u))
    print("Average Dice Coefficient: {:.4f}".format(DC_avg_u))

    # Write these average metrics to a file with model name
    with open('metrics.txt', 'a') as f:
        f.write(f"{model.__class__.__name__}, {sensitivity_avg_u:.4f}, {specificity_avg_u:.4f}, {precision_avg_u:.4f}, {F1_avg_u:.4f}, {JS_avg_u:.4f}, {DC_avg_u:.4f}\n")
