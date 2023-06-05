from .encoder import SpatialEncoder, ImageEncoder, TransformerEncoder
from .mlp import ImplicitNet
from .resnetfc import ResnetFC
from .img_encoder import UNet
from .mlp_separate import MLP


def make_mlp(conf, d_in, d_latent=0, allow_empty=False, **kwargs):
    mlp_type = conf.get_string("type", "mlp")  # mlp | resnet | mlp_separate
    if mlp_type == "mlp":
        net = ImplicitNet.from_conf(conf, d_in + d_latent, **kwargs)
    elif mlp_type == "resnet":
        net = ResnetFC.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == "mlp_separate":
        net = MLP.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == "empty" and allow_empty:
        net = None
    else:
        raise NotImplementedError("Unsupported MLP type")
    return net


def make_encoder(conf, **kwargs):
    enc_type = conf.get_string("type", "spatial")  # spatial | global | unet | FPN | transformer
    if enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    elif enc_type == "unet":
        net = UNet(16)
    elif enc_type == "transformer":
        net = TransformerEncoder(feature_net="unet", base_channels=8)
    else:
        raise NotImplementedError("Unsupported encoder type")
    print("Use --"+enc_type+"-- as image encoder")
    return net
