import torch
import torch.nn as nn
import torchvision.models as models


class MRNetPlane(nn.Module):
    """One plane model: AlexNet features per slice, global average pool, max over slices, one logit."""

    def __init__(self):
        super().__init__()
        alex = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.features = alex.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)                       # grayscale to 3 channels for ImageNet weights
        f = self.gap(self.features(x)).flatten(1)      # (n_slices, 256)
        v = f.max(dim=0).values                        # (256,) exam-level vector
        return self.fc(v)                              # (1,) logit