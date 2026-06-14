import cv2
import numpy as np

def guided_filter(guide, src, radius=60, eps=1e-3):
    """
    Edge-aware smoothing. 'guide' steers the filtering (the gray image),
    'src' is what we smooth (the coarse transmission map).
    Keeps transmission sharp at real edges, smooth elsewhere.
    """
    guide = guide.astype(np.float64)
    src = src.astype(np.float64)

    mean_guide = cv2.boxFilter(guide, cv2.CV_64F, (radius, radius))
    mean_src = cv2.boxFilter(src, cv2.CV_64F, (radius, radius))
    mean_gs = cv2.boxFilter(guide * src, cv2.CV_64F, (radius, radius))
    cov_gs = mean_gs - mean_guide * mean_src

    mean_gg = cv2.boxFilter(guide * guide, cv2.CV_64F, (radius, radius))
    var_g = mean_gg - mean_guide * mean_guide

    a = cov_gs / (var_g + eps)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

    return mean_a * guide + mean_b


def get_dark_channel(image, patch_size=15):
    """
    The dark channel is the minimum pixel value across all color
    channels, within a local patch. In haze-free outdoor images,
    this value is usually very low (close to 0) in non-sky regions.
    Haze makes it brighter — that's the clue we exploit.
    """
    # Take the minimum across the 3 color channels (B, G, R)
    min_channel = np.min(image, axis=2)

    # Erode with a patch-sized kernel = local minimum over the patch
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    return dark_channel
def estimate_atmospheric_light(image, dark_channel):
    """
    Atmospheric light A = the color of the most haze-opaque region.
    We find the top 0.1% brightest pixels in the dark channel,
    then pick the brightest among those in the original image.
    """
    h, w = dark_channel.shape
    num_pixels = h * w
    num_brightest = max(int(num_pixels * 0.001), 1)  # top 0.1%

    # Flatten and find indices of the brightest dark-channel pixels
    dark_flat = dark_channel.reshape(num_pixels)
    indices = dark_flat.argsort()[-num_brightest:]

    # Among those pixels, take the highest intensity in the original image
    image_flat = image.reshape(num_pixels, 3)
    brightest = image_flat[indices]

    atmospheric_light = brightest.max(axis=0)
    return atmospheric_light
def estimate_transmission(image, atmospheric_light, omega=0.95, patch_size=15):
    """
    Transmission t(x) = how much scene light reaches the camera.
    t close to 1 = clear, t close to 0 = thick haze.
    'omega' (0.95) keeps a tiny bit of haze for natural-looking depth.
    """
    # Normalize the image by the atmospheric light
    normalized = image / atmospheric_light

    # Dark channel of the normalized image
    transmission = 1 - omega * get_dark_channel(normalized, patch_size)
    return transmission
def dehaze_image(image, patch_size=15, omega=0.85, t0=0.1):
    """
    Full Dark Channel Prior dehazing pipeline.
    Input:  hazy image (BGR, uint8, 0-255)
    Output: dehazed image (BGR, uint8, 0-255)
    """
    # Work in float [0, 1] for precise math
    img = image.astype(np.float64) / 255.0

    # Step 1: dark channel
    dark = get_dark_channel(img, patch_size)

    # Step 2: atmospheric light
    A = estimate_atmospheric_light(img, dark)

    # Step 3: transmission map (coarse)
    transmission = estimate_transmission(img, A, omega, patch_size)

    # Step 3.5: refine transmission with guided filter (NEW)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    transmission = guided_filter(gray, transmission, radius=60, eps=1e-3)

    # Step 4: clamp transmission so we never divide by ~0
    transmission = np.clip(transmission, t0, 1.0)


    # Step 5: recover the clear image  J = (I - A) / t + A
    result = np.empty_like(img)
    for c in range(3):
        result[:, :, c] = (img[:, :, c] - A[c]) / transmission + A[c]

    # Clip back to valid range and convert to uint8
    result = np.clip(result, 0, 1) * 255
    return result.astype(np.uint8)
