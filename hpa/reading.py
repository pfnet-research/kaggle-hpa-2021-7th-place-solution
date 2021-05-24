import cv2
import numpy as np

from somen.pfio_utility import DirectoryInZip


def read_gray(directory: DirectoryInZip, path: str) -> np.ndarray:
    with directory.open(path, "rb") as fp:
        buf = fp.read()
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
    return img


def read_rgby(directory: DirectoryInZip, prefix: str, channel_last: bool = True) -> np.ndarray:
    imgs = []
    for color in ["red", "green", "blue", "yellow"]:
        img = read_gray(directory, f"{prefix}_{color}.png")
        imgs.append(img)

    if channel_last:
        return np.stack(imgs, axis=-1)
    else:
        return np.stack(imgs, axis=0)


def read_vis_rgb(directory: DirectoryInZip, prefix: str) -> np.ndarray:
    bgry = read_rgby(directory, prefix, True)
    rgb = np.zeros((*bgry.shape[:2], 3), dtype=bgry.dtype)
    # red = (red + yellow) / 2, green = (green + yellow) / 2, blue=blue
    rgb[..., 0] = (bgry[..., 0].astype(np.uint16) + bgry[..., 3]) // 2
    rgb[..., 1] = (bgry[..., 1].astype(np.uint16) + bgry[..., 3]) // 2
    rgb[..., 2] = bgry[..., 2]
    return rgb


def read_cellseg_input(directory: DirectoryInZip, prefix: str) -> np.ndarray:
    imgs = []
    for color in ["red", "yellow", "blue"]:
        img = read_gray(directory, f"{prefix}_{color}.png")
        imgs.append(img)
    return np.stack(imgs, axis=-1)
