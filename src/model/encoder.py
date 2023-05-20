# coding:utf-8
"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import src.util as util
from src.model.custom_encoder import ConvEncoder
import torch.autograd.profiler as profiler
from src.model.module import CBAMLayer
from src.model.img_encoder import UNet4transformer
from src.model.transformer import FMT_with_pathway


class TransformerEncoder(nn.Module):
    """
    Transformer image encoder
    """

    def __init__(
        self,
        feature_net="unet",
        base_channels=8,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
    ):
        """
        :param feature_net feature extract network. Here I will reuse the spatial | unet | global | FPN
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()
        self.latent = None
        if feature_net == "unet": # spatial | global | unet | FPN
            self.feature_net = UNet4transformer(base_channels=base_channels)
        # Transformer
        self.FMT_with_pathway = FMT_with_pathway()

        self.feature_scale = feature_scale
        self.latent_size = 56 # todo 暂时固定写

        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )
        # self.latent (B, L, H, W)
        # for smoothing
        self.dim_reduction_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, bias=False)
        self.dim_reduction_2 = nn.Conv2d(base_channels * 2, base_channels * 1, 1, bias=False)

        self.smooth_1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False)
        self.smooth_2 = nn.Conv2d(base_channels * 1, base_channels * 1, 3, padding=1, bias=False)

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
        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            # 因为此处使用torch.nn.functional.grid_sample进行采样，坐标需要进行归一化，所以uv坐标需要÷(H,W) TODO
            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            samples = F.grid_sample(
                self.latent, # input: (NB*NV,latent_size,H/2,W/2)=(9,512,150,200)
                uv, # grid: (NB*NV,ray_batch_size*sample,1,2)(9,8192,1,2)
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )

            # sample.shape (NB*NV,latent_size,ray_batch_size*sample,1)=(9,512,8192,1)
            return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x, nv):
        """
        Use transformer
        :param x image (B, C, H, W)     B=b*nv
        :param nv num of views per object
        :return latent (B, latent_size, H, W) B=NB*NV, latent_size = 512 (default)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)
        B, C, H, W = x.size()
        x = x.reshape(B // nv, nv, C, H, W).permute(1, 0, 2, 3, 4)
        x_lists = [x[i] for i in range(nv)]
        # 1. feature extract
        feature_lists = [self.feature_net(x_lists[i]) for i in range(nv)]
        # 2. transformer: only apply on stage1
        feature_lists_stage1 = [feature_mul_stage["stage1"] for feature_mul_stage in feature_lists]
        feature_lists_stage1 = self.FMT_with_pathway(feature_lists_stage1)

        # 3. smooth + upsample: to transmit info from stage1(after transformer) to stage2,3
        latents = []
        for idx,feature_lists in enumerate(feature_lists):
            feature_lists[idx]["stage1"] = feature_lists_stage1[idx]
            feature_lists[idx]["stage2"] = self.smooth_1(self._upsample_add(self.dim_reduction_1(feature_lists[idx]["stage1"]),feature_lists[idx]["stage2"]))
            feature_lists[idx]["stage3"] = self.smooth_2(self._upsample_add(self.dim_reduction_2(feature_lists[idx]["stage2"]),feature_lists[idx]["stage3"]))
            latents.append(torch.cat(feature_lists[idx], dim=1).unsqueeze(1)) # dim channel
        self.latent = torch.cat(latents,dim=1).reshape(B, C, H, W)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            feature_net=conf.get_string("feature_net", "unet"),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
        )

class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
        use_cbam=False,
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = util.get_norm_layer(norm_type)

        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        else:
            print("Using torchvision", backbone, "encoder")
            self.model = getattr(torchvision.models, backbone)(
                pretrained=pretrained, norm_layer=norm_layer
            )
            # Following 2 lines need to be uncommented for older configs
            self.model.fc = nn.Sequential()
            self.model.avgpool = nn.Sequential()
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )
        # self.latent (B, L, H, W)
        # CBAM
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.CBAMLists = nn.ModuleList()
            for i in range(num_layers):
                latent = [64, 64, 128, 256, 512, 1024][i]
                self.CBAMLists.append(CBAMLayer(latent))

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
        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            # 因为此处使用torch.nn.functional.grid_sample进行采样，坐标需要进行归一化，所以uv坐标需要÷(H,W) TODO
            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            samples = F.grid_sample(
                self.latent, # input: (NB*NV,latent_size,H/2,W/2)=(9,512,150,200)
                uv, # grid: (NB*NV,ray_batch_size*sample,1,2)(9,8192,1,2)
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )

            # sample.shape (NB*NV,latent_size,ray_batch_size*sample,1)=(9,512,8192,1)
            return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W) B=NB*NV, latent_size = 512 (default)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                # CBAM
                if self.use_cbam:
                    latents[i] = self.CBAMLists[i](latents[i])  # [B,C,H/2,W/2] C:64,64,128,256
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
            self.latent = torch.cat(latents, dim=1)
        # # CBAM
        # if self.use_cbam:
        #     self.latent = self.CBAMLayer(self.latent)  # [B,512,H/2,W/2]
        # TODO 我知道是归一化，但是这里第三步看不懂
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0  # 没有看懂
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
            use_cbam=conf.get_bool("use_cbam", False), # TODO, remember to change when eval if dont use conf
        )


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)


    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)

        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
        )
