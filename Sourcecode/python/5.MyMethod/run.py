#   main
import  torch
import random

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_function import plot_transformed_images, find_classes
from model import Model

#   Prepare Datasets
#   Expected structure
#   {Foldername}
#       -   'Train'
#       -   'Validation'
#data_dir = '../../../../Dataset/flowers_dataset/flowers_1/'

image_path = Path("../../../../Dataset/flowers/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    # Setup train and testing paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    
    # Set seed
    random.seed(42)
    
    # 1. Get all image paths (* means "any combination")
    image_path_list = list(image_path.glob("*/*/*.jpg"))

    # 2. Get random image path
    random_image_path = random.choice(image_path_list)

    # 3. Get image class from path name (the image class is the name of the directory where the image is stored)
    image_class = random_image_path.parent.stem

    # Write transform for image
    img_dim = (227, 227)
    data_transform = transforms.Compose([
        # Resize the images to expected image dimension
        transforms.Resize(size=img_dim),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])
    
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                     transform=data_transform)
        
    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=1, # how many samples per batch?
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             num_workers=1, 
                             shuffle=False) # don't usually need to shuffle testing data
    
    #   Initialize model
    # Get class names as a list
    class_names = train_data.classes
    classes_len = len(class_names)
    init_lr = 1e-2
    batch_size = 16
    momentum = 9e-1
    weight_decay = 1e-4
    num_epochs = 1
    
    model = Model(classes_len, init_lr, num_epochs, 
                  batch_size, momentum, weight_decay)
    model.to(device)
    
    
    # Update representation via BackProp
    classes = [0, 1]
    model.update(device, train_dataloader, classes)
    model.eval()

    model.n_known = model.n_classes
    print ("model classes : %d, " % model.n_known)

    total = 0.0
    correct = 0.0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        preds = model.classify(images)
        preds = preds.to(device)
        
        total += labels.size(0)
        
        if preds == labels:
            correct += 1

    # Train Accuracy
    print ('Train Accuracy : %.2f ,' % (100.0 * correct / total))

    print("End of Script")
    
if __name__ == '__main__':
    main()
    
    
    