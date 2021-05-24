import base64
import zlib

import numpy as np
from pycocotools import _mask as coco_mask


def encode_binary_mask(mask: np.ndarray) -> str:
    """Converts a binary mask into OID challenge encoding ascii text.

    The original implemantation is below.
    https://gist.github.com/pculliton/209398a2a52867580c6103e25e55d93c
    """

    # check input mask --
    if mask.dtype != np.bool8:
        raise ValueError("encode_binary_mask expects a binary mask, received dtype == %s" % mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError("encode_binary_mask expects a 2d mask, received shape == %s" % mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode()
