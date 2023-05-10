import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import numpy as np

from tqdm import tqdm

def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

class Model(nn.Module):
    def __init__(self, classes, init_lr, num_epochs, batch_size, momentum, weight_decay):
		# Hyper Parameters
        self.init_lr = init_lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Network architecture
        super(Model, self).__init__()
        self.model = models.resnet34(weights='IMAGENET1K_V1')
        self.model.apply(kaiming_normal_init)
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, classes, bias=False)
        self.fc = self.model.fc
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor = nn.DataParallel(self.feature_extractor) 
        
		# n_classes is incremented before processing new data in an iteration
		# n_known is set to n_classes after all data for an iteration has been processed
        self.n_classes = 0
        self.n_known = 0
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def increment_classes(self, new_classes):
        """Add n classes in the final fc layer"""
        n = len(new_classes)
		
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data
        
        if self.n_known == 0:
            new_out_features = n
        else:
            new_out_features = out_features + n
		
        self.model.fc = nn.Linear(in_features, new_out_features, bias=False)
        self.fc = self.model.fc
        
        kaiming_normal_init(self.fc.weight)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

    def classify(self, images):
        
        _, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)
        
        return preds

    def update(self, device, data_loader, classes):

        #   Specify new train classes
        new_classes = classes

        if len(new_classes) > 0:
            self.increment_classes(new_classes)
            self.to(device)

        optimizer = optim.SGD(self.parameters(), lr=self.init_lr,
                              momentum = self.momentum, 
                              weight_decay=self.weight_decay)

        dataset = data_loader.dataset
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)
        print(f"Model is on device: {next(self.parameters()).device}")
        with tqdm(total=self.num_epochs) as pbar:
            for epoch in range(self.num_epochs):
                for i, (images, labels) in enumerate(data_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    logits = self.forward(images)
                    logits.to(device)
                    loss = criterion(logits, labels)

                    loss.backward()
                    optimizer.step()

                    if (i+1) % 100 == 0:
                        tqdm.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, 
                                                                              self.num_epochs, 
                                                                              i+1, np.ceil(len(dataset)), 
                                                                              loss.data))

                pbar.update(1)
            
            