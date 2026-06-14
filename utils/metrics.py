import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original, dehazed):
    """
    PSNR compares two images pixel-by-pixel.
    Higher dB = the dehazed image is closer to the ground truth.
    """
    return psnr(original, dehazed, data_range=255)

def calculate_ssim(original, dehazed):
    """
    SSIM measures structural similarity as humans perceive it (0 to 1).
    'channel_axis=2' tells it we have a color image (H, W, 3).
    """
    return ssim(original, dehazed, channel_axis=2, data_range=255)

def resize_to_match(image, target):
    """
    PSNR/SSIM require both images to be the SAME size.
    Ground-truth and hazy images can differ by a few pixels,
    so we resize the dehazed result to match the ground truth.
    """
    h, w = target.shape[:2]
    return cv2.resize(image, (w, h))
