import torch
import torch.nn as nn

class GestureCNN(nn.Module):
    """
    A CNN model that classifies hand gestures into 1 of 5 classes.
    """
    def __init__(self):
        super().__init__()

        #Define the convolution layers

        #First Convolution Layer and Batch Normalization
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)

        #Second Convolution Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        #Third Convolution Layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        #Define the activation function
        self.relu = nn.ReLU()

        #Define the Pooling Layer which uses Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Define the Fully Connected Layers and Dropout
        self.fc1 = nn.Linear(in_features=(128*28*28), out_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=5)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Computing the forward pass on input x.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.reshape(-1, (128*28*28))

        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.dropout(x)

        x = self.fc2(x)

        return x
