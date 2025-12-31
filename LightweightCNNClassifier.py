import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightCNNClassifier(nn.Module):
    def __init__(self, input_dim=64, dropout=0.4):  # default 0.4, can tune 0.3–0.6
        super().__init__()
        # Input: (batch, 64)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm1d(16)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(16, 2)  # binary classification

    def forward(self, x):
        # x: (batch, 64)
        x = x.unsqueeze(1)  # (batch, 1, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global average pooling over sequence length
        x = x.mean(dim=2)  # (batch, 16)
        x = self.dropout(x)
        x = self.fc(x)     # (batch, 2) logits
        return x
