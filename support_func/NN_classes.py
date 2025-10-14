import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNetCustom(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNetCustom, self).__init__()

        # Convolution Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=3,  # input has 3 channels (RGB)
            out_channels=96,  # 96 filters
            kernel_size=11,  # 11x11 filters
            stride=4,
            padding=0,  # no padding (as "–" in table)
        )

        # Convolution Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2,  # from the table: "padding=2"
        )

        # Convolution Layer 3
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
        )

        # Convolution Layer 4
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
        )

        # Convolution Layer 5
        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
        )

        # MaxPooling layers (3x3 kernel, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # Dropouts
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        # Fully connected layers
        # After conv5 + MaxPool => feature maps: 256 filters of size 6×6
        # => flattened size = 256 * 6 * 6 = 9216
        # But the table references 6×6×256 => if we follow exactly, that’s 9216.
        # The table also lists 4096 neurons for FC1 and FC2, and 2 for FC3.

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)  # 2 outputs for your Table 2

        self.sm = nn.Softmax(1)

    def forward(self, x):
        # 1) Convolution layer 1 => ReLU => MaxPool
        x = self.conv1(x)  # shape: [B, 96, 55, 55] if input is 227x227
        x = F.relu(x)
        x = self.pool(x)  # shape: [B, 96, 27, 27]

        # 2) Convolution layer 2 => ReLU => MaxPool
        x = self.conv2(x)  # shape: [B, 256, 27, 27]
        x = F.relu(x)
        x = self.pool(x)  # shape: [B, 256, 13, 13]

        # 3) Convolution layer 3 => ReLU
        x = self.conv3(x)  # shape: [B, 384, 13, 13]
        x = F.relu(x)

        # 4) Convolution layer 4 => ReLU
        x = self.conv4(x)  # shape: [B, 384, 13, 13]
        x = F.relu(x)

        # 5) Convolution layer 5 => ReLU => MaxPool
        x = self.conv5(x)  # shape: [B, 256, 13, 13]
        x = F.relu(x)
        x = self.pool(x)  # shape: [B, 256, 6, 6]

        # Flatten the feature maps
        x = torch.flatten(x, 1)  # shape: [B, 256*6*6 = 9216]

        # Dropout 1 -> FC1 -> ReLU
        x = self.dropout1(x)
        x = self.fc1(x)  # shape: [B, 4096]
        x = F.relu(x)

        # Dropout 2 -> FC2 -> ReLU
        x = self.dropout2(x)
        x = self.fc2(x)  # shape: [B, 4096]
        x = F.relu(x)

        # FC3 -> ReLU (as described in the table, though unusual for final layer)
        x = self.fc3(x)  # shape: [B, 2]

        return self.sm(x)
