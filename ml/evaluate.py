from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as DL
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


from ml.model import GestureCNN
from ml import config

class Test():

    def __init__(self, model, trained_model_path, batch_size, num_epochs, device):
        """
        Load the trained parameters from disk and fit them into given model.
        """
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        #Load the learnt parameters and then initialize a model with them
        self.model.load_state_dict(torch.load(trained_model_path), map_location = self.device)
        self.model.to(self.device)
        self.model.eval()
    
    def set_transformations(self, mean_norm, std_norm):
        """
        Set the class object tranformations to be applied to the test dataset.
        """
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)
        ])
        
        self.test_transformations = transformations
    
    def set_loader(self, path_to_test):
        """
        Set the class object loader to load test data.
        """
        test_dataset = ImageFolder(root=path_to_test, transform=self.test_transformations)
        test_loader = DL(dataset=test_dataset, shuffle=False, batch_size=self.batch_size)

        self.test_loader = test_loader

        print(f"\nTest Loader set")
        print(f"Test Dataset Size: {len(test_dataset)}")
        print(f"Number of batches: {len(test_loader)}")
        print(f"Classes: {test_dataset.classes}")
        print(f"Class-to-LabelIndex Mapping: {test_dataset.class_to_idx}")
    
    
