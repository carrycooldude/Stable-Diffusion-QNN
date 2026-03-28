import numpy as np
from PIL import Image

# Quantization Constants (from metadata.yaml)
Q_TEXT_EMB_S, Q_TEXT_EMB_ZP = 0.0003629151, 25915
Q_UNET_LATENT_S, Q_UNET_LATENT_ZP = 0.0003279145, 34760
Q_UNET_TIMESTEP_S, Q_UNET_TIMESTEP_ZP = 0.014770733, 0
Q_UNET_TEXT_EMB_S, Q_UNET_TEXT_EMB_ZP = 0.0003463204, 23638
Q_UNET_OUT_LATENT_S, Q_UNET_OUT_LATENT_ZP = 0.0001840945, 30388
Q_VAE_LATENT_S, Q_VAE_LATENT_ZP = 0.0003278456, 34752
Q_VAE_IMAGE_S, Q_VAE_IMAGE_ZP = 1.5259021893143654e-05, 0

def quantize(arr, scale, zp):
    """
    Quantizes a float array to uint16 based on scale and zero_point.
    """
    q = np.round(arr / scale + zp)
    return np.clip(q, 0, 65535).astype(np.uint16)

def dequantize(arr, scale, zp):
    """
    Dequantizes a uint16 array back to float32.
    """
    return (arr.astype(np.float32) - zp) * scale

def postprocess_image(image_data):
    """
    Converts VAE output [1, 512, 512, 3] to a PIL Image.
    """
    # Dequantize first
    image = dequantize(image_data, Q_VAE_IMAGE_S, Q_VAE_IMAGE_ZP)
    
    # [1, 512, 512, 3] -> [0, 1]
    image = (image / 2 + 0.5).clip(0, 1)
    
    # [0, 1] -> [0, 255]
    image = (image[0] * 255).astype(np.uint8)
    
    return Image.fromarray(image)
