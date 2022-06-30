import torch
import torch.nn as nn

class Testmodel(nn.Module):
    def __init__(self, modelconfig):
        super(Testmodel, self).__init__()
        dropout = modelconfig.dropout if hasattr(modelconfig, 'dropout') else 0.3
        self.model = nn.Sequential(
            nn.Conv3d(1, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(256),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.model(x)