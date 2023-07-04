import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, imgs_path, class_map, img_dim):
        self.imgs_path = imgs_path
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
                
        self.class_map = class_map
        self.img_dim = img_dim
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        
        class_id = self.class_map[class_name]
        
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        
        class_id = torch.tensor([class_id])
        return img_tensor, class_id