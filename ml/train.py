import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as DL
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np


from ml.model import GestureCNN
from ml import config


class Train():

    def __init__(self, model, loss_criterion, optimizer, num_epochs, batch_size, device, batch_print_freq = 5):
        self.model = model
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.batch_print_freq = batch_print_freq
    
    def set_transformations(self, augmentations_dict, img_size, mean_norm, std_norm):
        """
        Sets train and validation set transformations as fields of the class.
        """
        train_transformations = transforms.Compose(
            [transforms.RandomRotation(degrees=augmentations_dict["ROTATION_DEGREES"]),
            transforms.RandomHorizontalFlip(p=augmentations_dict["HORIZONTAL_FLIP_PROB"]),
            transforms.RandomCrop(size=img_size, padding=augmentations_dict["RANDOM_CROP_PADDING"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)])
        
        val_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)
        ])

        self.train_transform = train_transformations
        self.val_transform = val_transformations

    def set_data_loaders(self, path_to_train, path_to_val):
        """
        Sets the loaders for the training and validation sets and applies transformations.
        """
        training_dataset = ImageFolder(root=path_to_train, transform=self.train_transform)
        validation_dataset = ImageFolder(root=path_to_val, transform=self.val_transform)

        train_loader = DL(dataset=training_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DL(dataset=validation_dataset, batch_size=self.batch_size, shuffle=False)

        self.train_loader = train_loader
        self.val_loader = val_loader

        print(f"\nDataset Loaders defined.")
        print(f"Train Dataset:")
        print(f"Classes: {training_dataset.classes}")
        print(f"Class-to-idx Mapping: {training_dataset.class_to_idx}")
        print(f"Validation Dataset:")
        print(f"Classes: {validation_dataset.classes}")
        print(f"Class-to-LabelIndex Mapping: {validation_dataset.class_to_idx}")
    
    def train_one_epoch(self):
        """
        Defines the training loop for 1 epoch.
        """
        self.model.train()

        epoch_loss = 0
        epoch_correct_pred = 0
        epoch_total_pred = 0

        num_batches = len(self.train_loader)

        for batch_idx, (images, labels) in enumerate(self.train_loader):

            actual_batch_size = labels.shape[0] #As the last batch may have fewer examples.

            self.optimizer.zero_grad()
            logit_outputs = self.model(images) #Forward pass
            loss = self.loss_criterion(logit_outputs, labels) #Calculate batch loss
            loss.backward() #Backward pass
            self.optimizer.step() #Update weights

            loss = loss.item()
            epoch_loss += loss
            
            output_predictions = torch.argmax(logit_outputs, dim=1)

            correct_pred = (output_predictions == labels).sum().item()
            total_pred = actual_batch_size

            batch_acc = (correct_pred/total_pred) * 100

            epoch_correct_pred += correct_pred
            epoch_total_pred += total_pred

            if batch_idx % self.batch_print_freq == 0:
                print(f"\nBatch Index: {batch_idx} ->")
                print(f"Batch Loss: {loss:.5f}")
                print(f"Batch Accuracy: {batch_acc}%")
        
        epoch_loss /= num_batches
        epoch_acc = 100 * (epoch_correct_pred/epoch_total_pred)

        return epoch_loss, epoch_acc


    def validate(self):
        """
        Validates the training on 1 epoch of the validation set.
        """
        self.model.eval()

        epoch_loss = 0
        epoch_correct_pred = 0
        epoch_total_pred = 0

        num_batches = len(self.val_loader)

        with torch.no_grad():
            for images, labels in self.val_loader:
                logit_outputs = self.model(images)
                loss = self.loss_criterion(logit_outputs, labels)
                loss = loss.item()
                loss += epoch_loss
                
                output_predictions = torch.argmax(logit_outputs, dim=1)

                correct_pred = (output_predictions == labels).sum().item()
                total_pred = labels.shape[0]

                epoch_correct_pred += correct_pred
                epoch_total_pred += total_pred

            epoch_loss /= num_batches
            epoch_acc = (epoch_correct_pred / epoch_total_pred) * 100
        
        return epoch_loss, epoch_acc
    

    def train_all_epochs(self):
        """
        Defines the training loop to train and validate through all epochs.
        """
        
        training_losses = []
        training_accuracies = []
        validation_losses = []
        validation_accuracies = []

        for epoch in range(self.num_epochs):
            print(f"\nStarting Training for Epoch Number {epoch+1}:")

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            training_losses.append(train_loss)
            training_accuracies.append(train_acc)
            validation_losses.append(val_loss)
            validation_accuracies.append(val_acc)

            print(f"\nEpoch {epoch+1} Complete!")
            print(f"Training Accuracy: {train_acc:.5f}")
            print(f"Validation Accuracy: {val_acc:.5f}")
        
        return training_losses, training_accuracies, validation_accuracies, validation_losses
            