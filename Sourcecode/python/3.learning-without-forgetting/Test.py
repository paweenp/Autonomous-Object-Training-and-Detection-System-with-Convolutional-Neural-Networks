import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from AlexNet import AlexNet

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#   ---------------------------------------------------------------------------------
#   New Dataset
#data_dir_new = '../../../../Dataset/flowers/'
data_dir_new = '../../../../Dataset/flowers_2/'
#data_dir_new = '../../../../Dataset/cats_and_dogs/'
image_datasets_new = {x: datasets.ImageFolder(os.path.join(data_dir_new, x),
                                          data_transforms[x])
                  for x in ['train', 'validation']}
dataloaders_new = {x: torch.utils.data.DataLoader(image_datasets_new[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'validation']}
dataset_sizes_new = {x: len(image_datasets_new[x]) for x in ['train', 'validation']}
class_names_new = image_datasets_new['train'].classes
#   ---------------------------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    
    class_names = ['dandelion', 'rose', 'sunflowers', 'tulips']
    num_classes = len(class_names)
    
    model = AlexNet(num_classes)
    #save_model_path = 'models/model.pth'
    #save_model_path = 'models/model_2.pth'
    save_model_path = 'model/lwf_model.pth'
    checkpoint = torch.load(save_model_path)
    model.load_state_dict(checkpoint['model'])
    
    model.to(device)
    print("Loaded model")

    
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    num_images = 6    

    with torch.no_grad():
        for inputs, labels in dataloaders_new['validation']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
            for j in range(inputs.size()[0]):
                images_so_far += 1
                
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])
                
                if images_so_far == num_images:
                    return
                
    
    
if __name__ == '__main__':
    main()
    print("End of program")