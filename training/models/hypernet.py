# -*- coding: utf-8 -*-
"""
Created on 26.01.22

"""
import torch
from timm.models import register_model
from timm.models.layers import trunc_normal_, DropPath
from torch import nn
from torch.nn import Conv2d, LeakyReLU, Conv3d
import torch.nn.functional as F

from training.models.convnext import LayerNorm3d, LayerNorm


class KernelNet2d(nn.Module):

    def __init__(self, x_dim, y_dim, ch_in, ch_out):
        super().__init__()
        num_c = int(ch_in * ch_out)
        self.conv1 = Conv2d(2, 16, (1, 1))
        self.act = LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = Conv2d(16, 16, (1, 1))
        self.conv3 = Conv2d(16, 4, (1, 1))

        self.conv4 = Conv2d(4, num_c, (1, 1))
        self.reshape = (ch_out, ch_in, x_dim, y_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, pos):
        x = self.conv1(pos)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = torch.reshape(x, self.reshape)
        return x


class KernelNet3d(nn.Module):

    def __init__(self, x_dim, y_dim, z_dim, ch_in, ch_out):
        super().__init__()
        num_c = int(ch_in * ch_out)
        self.conv1 = Conv3d(3, 16, (1, 1, 1))
        self.act = LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = Conv3d(16, 16, (1, 1, 1))
        self.conv3 = Conv3d(16, 4, (1, 1, 1))

        self.conv4 = Conv3d(4, num_c, (1, 1, 1))
        self.reshape = (ch_out, ch_in, x_dim, y_dim, z_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, pos):
        x = self.conv1(pos)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = torch.reshape(x, self.reshape)
        return x


class HyperConv2d(nn.Module):

    def __init__(self, x_dim, y_dim, ch_in, ch_out, depthwise=True):
        super().__init__()
        self.kernel_size_x = x_dim
        self.kernel_size_y = y_dim
        self.groups = ch_in if depthwise else 1
        assert not depthwise or ch_out % ch_in == 0  # for depthwise conv, #output channel has to be dividable by #input channel
        ch_in = ch_in if not depthwise else ch_out // ch_in
        self.hypernet = KernelNet2d(x_dim, y_dim, ch_in, ch_out)
        self.kernel_pos = torch.nn.Parameter(self.kernel_positions(), requires_grad=False)

    def kernel_positions(self):
        xx_range = torch.arange(-(self.kernel_size_x - 1) / 2, (self.kernel_size_x + 1) / 2, dtype=torch.float)
        yy_range = torch.arange(-(self.kernel_size_y - 1) / 2, (self.kernel_size_y + 1) / 2, dtype=torch.float)

        xx_range = torch.tile(torch.unsqueeze(xx_range, -1), [1, self.kernel_size_y])
        yy_range = torch.tile(torch.unsqueeze(yy_range, 0), [self.kernel_size_x, 1])

        xx_range = torch.unsqueeze(xx_range, 0)
        yy_range = torch.unsqueeze(yy_range, 0)

        pos = torch.concat([xx_range, yy_range], 0)

        pos = torch.unsqueeze(pos, 0)

        return pos

    def forward(self, x):
        kernel = self.hypernet(self.kernel_pos)
        x = F.conv2d(x, kernel, padding='same', groups=self.groups)
        return x


class HyperConv3d(nn.Module):

    def __init__(self, x_dim, y_dim, z_dim, ch_in, ch_out, depthwise=True):
        super().__init__()
        self.kernel_size_x = x_dim
        self.kernel_size_y = y_dim
        self.kernel_size_z = z_dim
        self.groups = ch_in if depthwise else 1
        assert not depthwise or ch_out % ch_in == 0  # for depthwise conv, #output channel has to be dividable by #input channel
        ch_in = ch_in if not depthwise else ch_out // ch_in
        self.hypernet = KernelNet3d(x_dim, y_dim, z_dim, ch_in, ch_out)
        self.kernel_pos = torch.nn.Parameter(self.kernel_positions(), requires_grad=False)

    def kernel_positions(self):
        xx_range = torch.arange(-(self.kernel_size_x - 1) / 2, (self.kernel_size_x + 1) / 2, dtype=torch.float)
        yy_range = torch.arange(-(self.kernel_size_y - 1) / 2, (self.kernel_size_y + 1) / 2, dtype=torch.float)
        zz_range = torch.arange(-(self.kernel_size_z - 1) / 2, (self.kernel_size_z + 1) / 2, dtype=torch.float)

        xx_range = torch.tile(xx_range[:, None, None], [1, self.kernel_size_y, self.kernel_size_z])
        yy_range = torch.tile(yy_range[None, :, None], [self.kernel_size_x, 1, self.kernel_size_z])
        zz_range = torch.tile(zz_range[None, None, :], [self.kernel_size_x, self.kernel_size_y, 1])

        xx_range = torch.unsqueeze(xx_range, 0)
        yy_range = torch.unsqueeze(yy_range, 0)
        zz_range = torch.unsqueeze(zz_range, 0)

        pos = torch.concat([xx_range, yy_range, zz_range], 0)

        pos = torch.unsqueeze(pos, 0)

        return pos

    def forward(self, x):
        kernel = self.hypernet(self.kernel_pos)
        x = F.conv3d(x, kernel, padding='same', groups=self.groups)
        return x


class HyperBlock3d(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, depthwise=True):
        super().__init__()
        self.hyperconv = HyperConv3d(7, 7, 7, dim, dim, depthwise=depthwise)  # nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm3d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.hyperconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, D, H, W, C) -> (N, C, D, H, W)

        x = input + self.drop_path(x)
        return x


class HyperBlock2d(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, depthwise=True):
        super().__init__()
        self.hyperconv = HyperConv2d(7, 7, dim, dim, depthwise=depthwise)  # nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.hyperconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class HyperConvNeXt2d(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
      + Hyper-Convolution Networks for Biomedical Image Segmentation (WACV 2022)
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=1, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[HyperBlock2d(dim=dims[i], drop_path=dp_rates[cur + j],
                               layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class HyperConvNeXt3d(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
      + Hyper-Convolution Networks for Biomedical Image Segmentation (WACV 2022)
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm3d(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm3d(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[HyperBlock3d(dim=dims[i], drop_path=dp_rates[cur + j],
                               layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-3, -2, -1]))  # global average pooling, (N, C, D, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class HyperConvNeXt3dSTOIC(nn.Module):
    def __init__(self, modelconfig):
        super(HyperConvNeXt3dSTOIC, self).__init__()
        self.model = HyperConvNeXt3d(in_chans=1, num_classes=2)

    def forward(self, x):
        x = x.expand(-1, 1, -1, -1, -1)
        return self.model(x)


@register_model
def hyperconvnext(pretrained=False, **kwargs):
    model = HyperConvNeXt2d(**kwargs)
    return model


if __name__ == "__main__":
    # c = HyperConv3d(3, 3, 3, 16, 32)
    # inp = torch.randn((2, 16, 128, 128, 128))
    # c = HyperConv2d(3, 3, 16, 32)
    # inp = torch.randn((2, 16, 128, 128))
    # res = c(inp)
    model = HyperConvNeXt3d(in_chans=1, num_classes=2).cuda()
    # model = ConvNeXt3dSTOIC(ConvNeXt3DConfig()).cuda()

    from torchsummary import summary

    summary(model, (1, 128, 128, 128))
    print()
