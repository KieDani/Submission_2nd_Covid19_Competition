import monai
import torch
import torch.nn as nn

def get_net():
    """returns a unet model instance."""

    num_classes = 2
    net = monai.networks.nets.BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels= 1 if num_classes == 2 else num_classes,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
    )
    return net


class UnetSegmentation(nn.Module):
    def __init__(self, modelconfig):
        super(UnetSegmentation, self).__init__()
        self.net = get_net()

    def forward(self, x):
        return self.net(x)