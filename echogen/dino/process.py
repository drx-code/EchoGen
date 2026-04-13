import torch
import torch.nn as nn

from echogen.dino.encoder_utils import load_encoders
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize, Resize
CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)
DINOV3_DEFAULT_MEAN = (0.485, 0.456, 0.406)
DINOV3_DEFAULT_STD = (0.229, 0.224, 0.225)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    x = (x + 1.) /  2.
    if 'clip' in enc_type:
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov3' in enc_type:
        res = 256 if resolution < 512 else 512
        x = Resize((res, res), antialias=True)(x)
        # x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bilinear', antialias=True)
        x = Normalize(DINOV3_DEFAULT_MEAN, DINOV3_DEFAULT_STD)(x)
    elif 'dinov1' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    return x
