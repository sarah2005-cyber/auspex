# ===== Residual Unit =====
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Residual Unit with Dropout =====
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualUnit(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)  # 🔹 spatial dropout for CNNs
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)              # 🔹 dropout between conv1 and conv2
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


# ===== S-ResNet Stream (5 residual units per stage) =====
class SResNetStream5Res(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        # Conv-A group: 8 channels
        self.convA = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnA = nn.BatchNorm2d(8)
        self.resA = nn.Sequential(*[ResidualUnit(8) for _ in range(5)])  # 5 residual blocks
        self.poolA = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Conv-B group: 12 channels
        self.convB = nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnB = nn.BatchNorm2d(12)
        self.resB = nn.Sequential(*[ResidualUnit(12) for _ in range(5)])  # 5 residual blocks
        self.poolB = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Conv-C group: 24 channels
        self.convC = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnC = nn.BatchNorm2d(24)
        self.resC = nn.Sequential(*[ResidualUnit(24) for _ in range(5)])  # 5 residual blocks

    def forward(self, x):
        # Conv-A
        x = F.relu(self.bnA(self.convA(x)))
        x = self.resA(x)
        x = self.poolA(x)

        # Conv-B
        x = F.relu(self.bnB(self.convB(x)))
        x = self.resB(x)
        x = self.poolB(x)

        # Conv-C
        x = F.relu(self.bnC(self.convC(x)))
        x = self.resC(x)

        # Global Average Pooling -> (batch, 24)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)


# ===== Dual Stream with 5-residual CNN =====
class DualStreamSResNet5Res(nn.Module):
    def __init__(self):
        super().__init__()
        self.stream1024 = SResNetStream5Res(input_channels=1)
        self.stream512 = SResNetStream5Res(input_channels=1)

    def forward(self, spec1024, spec512):
        f1024 = self.stream1024(spec1024)  # (batch, 24)
        f512 = self.stream512(spec512)     # (batch, 24)
        return torch.cat([f1024, f512], dim=1)  # (batch, 48)
