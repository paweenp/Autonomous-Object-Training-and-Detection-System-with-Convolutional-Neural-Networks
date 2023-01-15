import random
import torch
import torchvision
import data_setup
import utils
import matplotlib.pyplot as plt

from torchinfo import summary
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from going_modular import engine
from typing import List, Tuple

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):
    
    
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)
    
    plt.show()


if __name__ == '__main__':
    
    # --------------- CUSTOM DATASET -----------------------
    data_path = Path('inputs')
    image_path = data_path / "otto_data"

    # Setup train and testing paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Set seed
    random.seed(42) # <- try changing this and see what happens

    # 1. Get all image paths (* means "any combination")
    image_path_list = list(image_path.glob("*/*/*.jpg"))

    # 2. Get random image path
    random_image_path = random.choice(image_path_list)

    # 3. Get image class from path name (the image class is the name of the directory where the image is stored)
    image_class = random_image_path.parent.stem

    # Transform dataset into PyTorch
    # Write transform for image
    data_transform = transforms.Compose([
        # Resize the images to 64x64
        transforms.Resize(size=(64, 64)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])

    # View the transformed images
    #utils.plot_transformed_images(image_path_list, 
    #                        transform=data_transform, 
    #                        n=3)

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                      transform=data_transform, # transforms to perform on data (images)
                                      target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                     transform=data_transform)

    # Get class names as a list
    class_names = train_data.classes

    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, 
                                  batch_size=1, # how many samples per batch?
                                  num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                                  shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                                 batch_size=1, 
                                 num_workers=1, 
                                 shuffle=False) # don't usually need to shuffle testing



    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
    #                                                                               test_dir=test_dir,
    #                                                                               transform=normalize, # resize, convert images to between 0 & 1 and normalize them
    #                                                                               batch_size=32) # set mini-batch size to 32

    # Define the number of classes
    num_classes = 2

    # NEW: Setup the model with pretrained weights and send it to the target device (torchvision v0.13+)
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
 
    pretrain_model_path = "pretrain_efficientnet_b0.pt"
    torch.save(model.state_dict(), pretrain_model_path)
 
    # Load the pre-trained model
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #model_Path = 'models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
    #model = torch.load(model_Path)

    # Print a summary using torchinfo (uncomment for actual output)
    #summary(model=model, 
    #        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
    #        # col_names=["input_size"], # uncomment for smaller output
    #        col_names=["input_size", "output_size", "num_params", "trainable"],
    #        col_width=20,
    #        row_settings=["var_names"]
    #)

    # Freeze the base layers
    for param in model.features.parameters():
        param.requires_grad = False
     
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    
    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)
    
    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)
    
    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    # Set the random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()
    
    # Setup training and save the results
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=5,
                           device=device)
    
    posttrain_model_path = "posttrain_efficientnet_b0.pt"
    torch.save(model.state_dict(), posttrain_model_path)
    
    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
    
    
    from helper_functions import plot_loss_curves
    
    # Plot the loss curves of our model
    plot_loss_curves(results)
    
    
    num_images_to_plot = 3
    test_image_path_list = list(Path(test_dir).glob("*/*.jpg")) # get list all image paths from test data 
    test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                           k=num_images_to_plot) # randomly select 'k' image paths to pred and plot
    
    
    # Make predictions on and plot the images
    for image_path in test_image_path_sample:
        pred_and_plot_image(model=model, 
                            image_path=image_path,
                            class_names=class_names,
                            # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                            image_size=(224, 224))
        
    
    print("End of program")
    
    
    
    