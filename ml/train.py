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
import traceback
import pandas as pd


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
            transforms.ColorJitter(
                brightness=augmentations_dict["COLOR_JITTER_BRIGHTNESS"],
                contrast = augmentations_dict["COLOR_JITTER_CONTRAST"]
            ),
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
            images = images.to(self.device)
            labels = labels.to(self.device)

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
                images = images.to(self.device)
                labels = labels.to(self.device)
                logit_outputs = self.model(images)
                loss = self.loss_criterion(logit_outputs, labels)
                loss = loss.item()
                epoch_loss += loss
                
                output_predictions = torch.argmax(logit_outputs, dim=1)

                correct_pred = (output_predictions == labels).sum().item()
                total_pred = labels.shape[0]

                epoch_correct_pred += correct_pred
                epoch_total_pred += total_pred

            epoch_loss /= num_batches
            epoch_acc = (epoch_correct_pred / epoch_total_pred) * 100
        
        return epoch_loss, epoch_acc
    

    def train_all_epochs(self, model_save_path):
        """
        Defines the training loop to train and validate through all epochs.
        """
        
        training_losses = []
        training_accuracies = []
        validation_losses = []
        validation_accuracies = []

        best_val_acc = 0
        best_epoch = 0

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

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch+1
                torch.save(self.model.state_dict(), model_save_path)
                print(f"\nNEW BEST MODEL SAVED - Validation Accuracy: {val_acc:.5f}")

        print('\n' + "="*60)
        print(f"TRAINING COMPLETE!")
        print("="*60)
        print(f"Best Validation Accuracy: {best_val_acc:.3f}%")
        print(f"Best Epoch: {best_epoch}")
        print(f"Model saved to: {model_save_path}")

        stats = {
            "Training Losses": training_losses,
            "Training Accuracies": training_accuracies,
            "Validation Losses": validation_losses,
            "Validation Accuracies": validation_accuracies
        }

        stats_df = pd.DataFrame(stats)
        stats_df.index.name = "Epoch"

        print(f"\nTraining Summary:\n\n{stats_df}")

        return training_losses, training_accuracies, validation_accuracies, validation_losses

    
    def plot_training_curves(self, train_losses, train_accs, val_losses, val_accs, plot_path):
        """
        Plot and save training curves.
        """
        epochs = range(1, len(train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Loss plot
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining curves saved to: {plot_path}")
        
        plt.close()


if __name__ == '__main__':
    try:
        print("="*60)
        print("HAND GESTURE CNN TRAINING")
        print("="*60)
        
        # Create model
        print("\nInitializing model...")
        model = GestureCNN().to(config.DEVICE)
        print(f"Model created and moved to {config.DEVICE}!")
        
        #Create loss and optimizer
        cross_entropy_loss_criterion = nn.CrossEntropyLoss()
        adam_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        #Create trainer
        print("\nCreating trainer...")
        trainer = Train(
            model=model,
            loss_criterion=cross_entropy_loss_criterion,
            optimizer=adam_optimizer,
            num_epochs=config.NUM_EPOCHS,
            batch_size=config.BATCH_SIZE,
            device=config.DEVICE,
            batch_print_freq=config.PRINT_FREQUENCY
        )
        print("Trainer initialized!")
        
        #Set transforms
        print("\nSetting up data transformations...")
        trainer.set_transformations(
            augmentations_dict=config.AUGMENTATIONS,
            img_size=config.IMG_SIZE,
            mean_norm=config.NORMALIZE_MEAN,
            std_norm=config.NORMALIZE_STD
        )
        print("Augmentations configured!")

        #Set data loaders
        print("\nLoading datasets...")
        trainer.set_data_loaders(
            path_to_train=config.TRAIN_DIR,
            path_to_val=config.VAL_DIR
        )
        print(f"Training images: {len(trainer.train_loader.dataset)}")
        print(f"Validation images: {len(trainer.val_loader.dataset)}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Batches per epoch: {len(trainer.train_loader)}")
        
        #Train
        print("\nStarting training...")
        print("="*60)
        
        model_save_path = config.MODELS_DIR / config.MODEL_SAVE_NAME
        
        train_losses, train_accs, val_accs, val_losses = trainer.train_all_epochs(
            model_save_path=model_save_path
        )
        
        #Plot results
        print("\nGenerating training curves...")
        plot_path = config.RESULTS_DIR / 'training_curves.png'
        trainer.plot_training_curves(
            train_losses=train_losses,
            train_accs=train_accs,
            val_losses=val_losses,
            val_accs=val_accs,
            plot_path=plot_path
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModel saved to: {model_save_path}")
        print(f"Training curves saved to: {plot_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        traceback.print_exc()