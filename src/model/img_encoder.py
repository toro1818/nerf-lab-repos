import torch
import torch.nn as nn
import torch.nn.functional as F


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier",**kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu


        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """Applies a U-net to Extract images' feature for cascade

       :Parm:
           base_channels: u-net base channels

    """
    def __init__(self, base_channels=8, ):
        super(UNet, self).__init__()
        self.index_interp = "bilinear"
        self.upsample_interp = "bilinear"
        self.latent = None
        self.latent_size = 56 # todo 暂时固定写
        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, stride=1, padding=1),
            Conv2d(base_channels, base_channels, 3, stride=1, padding=1)
        )
        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1)
        )
        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1)
        )
        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]
        self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
        self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

        self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
        self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
        self.out_channels.append(2 * base_channels)
        self.out_channels.append(base_channels)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        if uv.shape[0] == 1 and self.latent.shape[0] > 1:
            uv = uv.expand(self.latent.shape[0], -1, -1)

        # 因为此处使用torch.nn.functional.grid_sample进行采样，坐标需要进行归一化，所以uv坐标需要÷(H,W) TODO
        if len(image_size) > 0:
            if len(image_size) == 1:
                image_size = (image_size, image_size)
            # scale = self.latent_scaling / image_size
            scale = (2 / image_size)
            uv = uv * scale - 1.0

        uv = uv.unsqueeze(2)  # (B, N, 1, 2)
        samples = F.grid_sample(
            self.latent, # input: (NB*NV,latent_size,H/2,W/2)=(9,512,150,200)
            uv, # grid: (NB*NV,ray_batch_size*sample,1,2)(9,8192,1,2)
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )
        # sample.shape (NB*NV,latent_size,ray_batch_size*sample,1)=(9,512,8192,1)
        return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        pre_features = conv2
        latents = []
        latents.append(self.out1(pre_features))
        pre_features = self.deconv1(conv1, pre_features)
        latents.append(self.out2(pre_features))
        pre_features = self.deconv2(conv0, pre_features)
        latents.append(self.out3(pre_features))


        # 暂时只使用最后的输出
        latent_sz = latents[2].shape[-2:]
        align_corners = None if self.index_interp == "nearest " else True
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode=self.upsample_interp,
                align_corners=align_corners,
            )
        self.latent = torch.cat(latents, dim=1)

        return self.latent


class UNet4transformer(nn.Module):
    """Applies a U-net to Extract images' feature for cascade

       :Parm:
           base_channels: u-net base channels

    """
    def __init__(self, base_channels=8, ):
        super(UNet4transformer, self).__init__()
        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, stride=1, padding=1),
            Conv2d(base_channels, base_channels, 3, stride=1, padding=1)
        )
        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1)
        )
        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1)
        )
        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]
        self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
        self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

        self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
        self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
        self.out_channels.append(2 * base_channels)
        self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        pre_features = conv2
        outputs = {}
        outputs["stage1"] = self.out1(pre_features)
        pre_features = self.deconv1(conv1, pre_features)
        outputs["stage2"] = self.out2(pre_features)
        pre_features = self.deconv2(conv0, pre_features)
        outputs["stage3"] = self.out3(pre_features)

        return outputs
