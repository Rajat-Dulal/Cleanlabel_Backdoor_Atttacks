"""CNNs used in Phase 2 (SIT746).

SurrogateResNet : plain 5-conv-layer CNN with BatchNorm (despite the name, no residual
                  connections). Used as the guidance surrogate AND as the same-architecture victim.
TransferResNet  : genuine residual network (ResidualBlock x6). Used as the cross-architecture
                  (transfer-learning) victim.
Both expose forward(x, return_features=False) -> logits or (logits, 256-d feature).
"""
import torch.nn as nn


class SurrogateResNet(nn.Module):
    """Small CNN, fast to train on CIFAR-10 from scratch (~2-3 min / 15 epochs on a T4)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> 256-d feature vector
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_features: bool = False):
        feat = self.features(x).flatten(1)
        out = self.classifier(feat)
        return (out, feat) if return_features else out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:  # projection shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class TransferResNet(nn.Module):
    """Small ResNet-style CNN for the CIFAR-10 cross-architecture (transfer) experiments."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.layer1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_features: bool = False):
        x = self.layer3(self.layer2(self.layer1(self.stem(x))))
        feat = self.pool(x).flatten(1)  # 256-d
        out = self.classifier(feat)
        return (out, feat) if return_features else out


ARCHITECTURES = {"surrogate": SurrogateResNet, "resnet": TransferResNet}


def build_model(arch: str, num_classes: int = 10):
    if arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {list(ARCHITECTURES)}; got {arch!r}")
    return ARCHITECTURES[arch](num_classes)
