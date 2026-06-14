"""Model factory: U-Net and DeepLabV3+ with a shared ResNet34 ImageNet encoder."""
import segmentation_models_pytorch as smp

import config as C


def build_model(name, encoder=None):
    """name in {'unet', 'deeplabv3plus'}. encoder defaults to config.ENCODER."""
    kw = dict(
        encoder_name=encoder or C.ENCODER,
        encoder_weights=C.ENCODER_WEIGHTS,
        in_channels=3,
        classes=C.NUM_CLASSES,
    )
    name = name.lower()
    if name == "unet":
        return smp.Unet(**kw)
    if name in ("deeplabv3plus", "deeplab", "dlv3+"):
        return smp.DeepLabV3Plus(**kw)
    raise ValueError(f"Unknown model '{name}'. Use 'unet' or 'deeplabv3plus'.")


def count_params(model):
    return sum(p.numel() for p in model.parameters())
