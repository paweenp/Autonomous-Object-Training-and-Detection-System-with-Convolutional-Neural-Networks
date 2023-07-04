#   Performing LwF
#   ---------------------------------------------------------------------------------
#   1.  Load old and new datasets into ['train'] and ['validation']
#   2.  Load Network Structure i.e. AlexNet
#   3.  Decalre the SGD optimizer with weight decay = 5e-4 and loss function with default CrossEntropyLoss
#   4.  Train the Network with old datasets
#       4.1.    Show Epoch, Loss, and Accuracy
#       4.2.    Save the net
#   5.  Modify the fully classifier (fc) layer for increase classes
#   6.  Train the Network with LwF loss | lwf loss eq. --> lwf_loss = (1 - alpha) * new_loss + alpha * old_loss
#       6.1.    Show Epoch, Loss, and Accuracy
#       6.2.    Save the net
#   ---------------------------------------------------------------------------------


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import time
import os
import copy

from AlexNet import AlexNet
from helper_functions import imshow

cudnn.benchmark = True
plt.ion()   # interactive mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#   Initialize parameters
batch_size = 4
workers = 4
img_size = (227, 227)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#   Prepare Datasets
data_dir_old = '../../../../Dataset/flowers/'
data_dir_new = '../../../../Dataset/flowers_2/'
#   ---------------------------------------------------------------------------------
#   Old Dataset
#data_dir_old = '../../../../Dataset/cats_and_dogs/'
image_datasets_old = {x: datasets.ImageFolder(os.path.join(data_dir_old, x),
                                          data_transforms[x])
                  for x in ['train', 'validation']}
dataloaders_old = {x: torch.utils.data.DataLoader(image_datasets_old[x], batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
              for x in ['train', 'validation']}
dataset_sizes_old = {x: len(image_datasets_old[x]) for x in ['train', 'validation']}

#   ---------------------------------------------------------------------------------
#   New Dataset
image_datasets_new = {x: datasets.ImageFolder(os.path.join(data_dir_new, x),
                                          data_transforms[x])
                  for x in ['train', 'validation']}
dataloaders_new = {x: torch.utils.data.DataLoader(image_datasets_new[x], batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
              for x in ['train', 'validation']}
dataset_sizes_new = {x: len(image_datasets_new[x]) for x in ['train', 'validation']}

#   ---------------------------------------------------------------------------------

class_names_old = image_datasets_old['train'].classes
class_names_new = image_datasets_new['train'].classes

criterion = nn.CrossEntropyLoss()

#   Train with Normal Loss function
def train_normal(model, dataloader, criterion, optimizer, num_epochs):
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            #   Zero the parameter gradients
            optimizer.zero_grad()
            
            #   forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)    
        
        #   print statistics
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Loss: {epoch_loss:.4f}")
           
    return model

def lwf_loss(new_logits, old_logits, new_labels, old_labels, alpha=0.5):
    new_loss = criterion(new_logits, new_labels)
    old_loss = criterion(old_logits, old_labels)
    return (1 - alpha) * new_loss + alpha * old_loss

#   Train with LwF Loss function
def train_lwf(model, dataloaders, criterion, optimizer, num_epochs):
    model = model.to(device)
    old_dataloader = dataloaders[0]
    new_dataloader = dataloaders[1]
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        
        for phase in ["old", "new"]:
            if phase == "old":
                dataloader = old_dataloader
            else:
                dataloader = new_dataloader

            running_loss = 0.0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "new"):
                    logits = model(inputs)
                    if phase == "old":
                        old_logits = logits.detach()
                        old_labels = labels
                        continue
                    else:
                        new_logits = logits
                        new_labels = labels
                        loss = lwf_loss(new_logits, old_logits, new_labels, old_labels, alpha=0.5)

                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            print(f"{phase} Loss: {epoch_loss:.4f}")

    return model

#   The script will start from here
def main():
    
    #   Declare Network
    num_classes = len(class_names_old)
    model = AlexNet(num_classes)
    
    #   Declare SGD Optimizer with Cross Entropy Loss function
    learning_rate = 1e-3
    mmomentum = 0.9
    R = 5e-4    #   weight decay
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=mmomentum,
                          weight_decay=R)
    
    #   Train the Network with old dataset
    print("Train the network ... ")
    num_epochs = 10
    model = train_normal(model, dataloaders_old['train'], criterion, optimizer, num_epochs)
    
    #   Save the network checkpoint and optimizer
    print("Save model ...")
    state = {
        'model': model.state_dict(),
        'optim' : optimizer,        
    }
    torch.save(state, './model/normal_model.pth')
    
    #   Compute and store old outputs
    # old_loss = compute_oldloss(net, dataloaders_old['train'])
    
    #   Modify classification layer output in Network
    num_class = len(class_names_new) + len(class_names_old)
    num_ftrs = 4096
    model.fc2 = nn.Sequential(nn.Linear(num_ftrs ,num_class))

    #   Train the Network with new dataset
    print("Train the network with LwF ... ")
    alpha = 0.5
    model = train_lwf(model, [dataloaders_old['train'], dataloaders_new['train']], criterion, optimizer, num_epochs)
    
    #   Save the network checkpoint and optimizer
    print("Save model ...")
    state = {
        'model': model.state_dict(),
        'optim' : optimizer,        
    }
    torch.save(state, './model/lwf_model.pth')
    
if __name__ == '__main__':
    main()
    print('End of program')