import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_BT(nn.Module):
    def __init__(self, params):
        super(CNN_BT, self).__init__()

        # Extract parameters from the input dictionary
        Cin, Hin, Win = params["shape_in"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        # Convolution layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)

        # Dynamically calculate the number of features for the fully connected layer
        self.num_flatten = self._get_flatten_size(Cin, Hin, Win)

        # Fully connected layers
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)  # Use calculated flatten size
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def _forward_conv(self, X):
        """
        Helper function to pass input through convolutional layers only
        and calculate the flattened size.
        """
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        return X

    def _get_flatten_size(self, Cin, Hin, Win):
        """
        Dynamically calculate the flatten size by passing a dummy input through conv layers.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, Cin, Hin, Win)
            output = self._forward_conv(dummy_input)
            flatten_size = output.view(1, -1).size(1)
        return flatten_size

    def forward(self, X):
        # Pass input through conv layers
        X = self._forward_conv(X)

        # Flatten the tensor
        X = X.view(X.size(0), -1)

        # Pass through fully connected layers
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)

        # Output probabilities using softmax
        return F.log_softmax(X, dim=1)

# Example parameters for the model
params = {
    "shape_in": (3, 256, 256),  # Input shape (channels, height, width)
    "initial_filters": 8,        # Initial number of filters
    "num_fc1": 100,              # Number of neurons in the first fully connected layer
    "dropout_rate": 0.25,        # Dropout rate
    "num_classes": 2             # Number of output classes
}

# Create the model
model = CNN_BT(params)
