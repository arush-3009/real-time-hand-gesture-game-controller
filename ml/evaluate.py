from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as DL
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import traceback


from ml.model import GestureCNN
from ml import config


class Evaluate():

    def __init__(self, model, trained_model_path, batch_size, device):
        """
        Load trained model parameters from disk.
        """
        self.device = device
        self.model = model
        self.batch_size = batch_size

        #Load learnt parameters into model
        self.model.load_state_dict(torch.load(trained_model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def set_transformations(self, mean_norm, std_norm):
        """
        Set transformations for test dataset - no augmentation.
        """
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)
        ])
        
        self.test_transformations = transformations
    
    def set_loader(self, path_to_test):
        """
        Set loader to load test data.
        """
        test_dataset = ImageFolder(root=path_to_test, transform=self.test_transformations)
        test_loader = DL(dataset=test_dataset, shuffle=False, batch_size=self.batch_size)

        self.test_loader = test_loader
        self.class_names = test_dataset.classes

        print(f"\nTest Loader set ->")
        print(f"Test Dataset Size: {len(test_dataset)}")
        print(f"Number of batches: {len(test_loader)}")
        print(f"Classes: {test_dataset.classes}")
        print(f"Class-to-LabelIndex Mapping: {test_dataset.class_to_idx}")
    
    def get_predictions(self):
        """
        Run model on entire test set and collect predictions.
        """
        all_predictions = []
        all_true_labels = []

        self.model.eval()

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logit_outputs = self.model(images)
                output_predictions = torch.argmax(logit_outputs, dim=1)

                #Move to CPU and convert to numpy
                all_predictions.append(output_predictions.cpu().numpy())
                all_true_labels.append(labels.cpu().numpy())

        #Combine all batches into single arrays
        self.predictions = np.concatenate(all_predictions)
        self.true_labels = np.concatenate(all_true_labels)

        print(f"\nPredictions collected for {len(self.predictions)} images.")
    
    def calculate_metrics(self):
        """
        Calculate accuracy and per-class accuracy.
        """
        #Overall accuracy
        correct = (self.predictions == self.true_labels).sum()
        total = len(self.true_labels)
        self.overall_accuracy = (correct / total) * 100

        #Per-class accuracy
        self.per_class_accuracy = {}
        for i, class_name in enumerate(self.class_names):
            mask = (self.true_labels == i)
            class_total = mask.sum()
            
            if class_total > 0:
                class_correct = (self.predictions[mask] == i).sum()
                class_acc = (class_correct / class_total) * 100
                self.per_class_accuracy[class_name] = class_acc

        print(f"\nMetrics calculated")
    
    def generate_confusion_matrix(self):
        """
        Generate confusion matrix using sklearn.
        """
        self.conf_matrix = confusion_matrix(self.true_labels, self.predictions)
        print(f"Confusion matrix generated")
    
    def generate_classification_report(self):
        """
        Generate precision, recall, f1-score report.
        """
        self.classification_rep = classification_report(
            self.true_labels,
            self.predictions,
            target_names=self.class_names,
            digits=2
        )
        print(f"Classification report generated")
    
    def plot_confusion_matrix(self, save_path):
        """
        Plot confusion matrix as heatmap and save.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        #Create heatmap using imshow
        im = ax.imshow(self.conf_matrix, cmap='Blues')

        #Set ticks and labels
        ax.set_xticks(np.arange(len(self.class_names)))
        ax.set_yticks(np.arange(len(self.class_names)))
        ax.set_xticklabels(self.class_names)
        ax.set_yticklabels(self.class_names)

        #Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        #Add text annotations
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                text = ax.text(j, i, self.conf_matrix[i, j],
                             ha="center", va="center", color="black", fontsize=12)

        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')

        #Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=15)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nConfusion matrix plot saved to: {save_path}")
    
    def print_results(self):
        """
        Print all evaluation results.
        """
        print('\n' + '='*60)
        print('TEST SET RESULTS')
        print('='*60)

        print(f"\nOverall Test Accuracy: {self.overall_accuracy:.5f}%")

        print(f"\nPer-Class Accuracy:")
        for class_name, acc in self.per_class_accuracy.items():
            print(f"  {class_name}: {acc:.3f}%")

        print(f"\nClassification Report:")
        print(self.classification_rep)

        print('='*60)


if __name__ == '__main__':
    try:
        print('='*60)
        print('MODEL EVALUATION')
        print('='*60)

        #Load model
        print("\nInitializing model...")
        model = GestureCNN()
        trained_model_path = config.MODELS_DIR / config.MODEL_SAVE_NAME

        evaluator = Evaluate(
            model=model,
            trained_model_path=trained_model_path,
            batch_size=32,
            device=config.DEVICE
        )
        print(f"Model loaded from: {trained_model_path}")

        #Set transformations
        print("\nSetting up transformations...")
        evaluator.set_transformations(
            mean_norm=config.NORMALIZE_MEAN,
            std_norm=config.NORMALIZE_STD
        )

        #Set loader
        print("\nLoading test dataset...")
        evaluator.set_loader(path_to_test=config.TEST_DIR)

        #Get predictions
        print("\nRunning model on test set...")
        evaluator.get_predictions()

        #Calculate metrics
        print("\nCalculating metrics...")
        evaluator.calculate_metrics()
        evaluator.generate_confusion_matrix()
        evaluator.generate_classification_report()

        #Plot confusion matrix
        cm_save_path = config.RESULTS_DIR / 'confusion_matrix.png'
        evaluator.plot_confusion_matrix(save_path=cm_save_path)

        #Print results
        evaluator.print_results()

        print('\n' + '='*60)
        print('EVALUATION COMPLETE!')
        print('='*60)

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        traceback.print_exc()