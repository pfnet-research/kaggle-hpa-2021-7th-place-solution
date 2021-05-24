from typing import Optional

import albumentations as A
import numpy as np


def create_green_channel(base_image: np.ndarray, noise_scale: float = 25.0, max_shift: int = 20) -> np.ndarray:
    """Data augmentation

    Args:
        base_image: (height, width)
        noise_scale (int): original image range is about 0~255
        max_shift (int):
    Returns:
        green_image: (height, width)
    """
    green_image = np.zeros_like(base_image)
    # 1. Adding gaussian noise
    base_image = base_image + np.random.randn(*base_image.shape) * noise_scale
    # base_image = np.clip(base_image, 0.0, 255.0)  # is it necessary??, maybe not.
    height, width = base_image.shape
    # 2. shift randomly, up to 2 pixel
    x_shift = np.random.randint(-max_shift, max_shift + 1)
    x_min = max(0, x_shift)
    x_max = min(width, width + x_shift)
    x_min2 = max(0, -x_shift)
    x_max2 = min(width, width - x_shift)
    y_shift = np.random.randint(-max_shift, max_shift + 1)
    y_min = max(0, y_shift)
    y_max = min(height, height + y_shift)
    y_min2 = max(0, -y_shift)
    y_max2 = min(height, height - y_shift)
    green_image[x_min:x_max, y_min:y_max] = base_image[x_min2:x_max2, y_min2:y_max2]
    return green_image


def create_green_channel_albumentations(base_image: np.ndarray, transform: Optional[A.Compose] = None) -> np.ndarray:
    """Data augmentation

    Args:
        base_image: (height, width)
        transform (None or A.Compose): albumentations transform
    Returns:
        green_image: (height, width)
    """
    if transform is None:
        # For backward compatibility
        return create_green_channel(base_image)
    return transform(image=base_image)["image"]
