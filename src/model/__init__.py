from .models import PixelNeRFNet
from .models_with_freq_mask import PixelNeRFNet as PixelNeRFNet1

def make_model(conf, *args, **kwargs):
    """ Placeholder to allow more model types """
    model_type = conf.get_string("type", "pixelnerf")  # single
    if model_type == "pixelnerf":
        net = PixelNeRFNet(conf, *args, **kwargs)
    elif model_type == "pixelnerf1":
        net = PixelNeRFNet1(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
