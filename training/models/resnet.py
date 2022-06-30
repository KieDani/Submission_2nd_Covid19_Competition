import torch
import torch.nn as nn
import torchvision


class ResNet(nn.Module):
    def __init__(self, modelconfig):
        super(ResNet, self).__init__()
        self.model = torchvision.models.video.r3d_18(
            pretrained= modelconfig.pretrained if hasattr(modelconfig, "pretrained") else False
        )
        self.model.fc = nn.Linear(512, 2)
        self.model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

    def forward(self, x):
        return self.model(x)