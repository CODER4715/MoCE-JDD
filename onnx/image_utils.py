# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image

def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.
    From C x H x W [0..1] to  H x W x C [0...255]
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        # Ensure channel is the last dimension
        if ar.shape[0] < ar.shape[-1]:
             ar = ar.transpose(1, 2, 0)
    return Image.fromarray(ar)

def save_image(name, image_np, output_path="output/", return_path=False):
    """
    Saves a NumPy image to a file with corrected path handling.

    Args:
        name (str): The base name of the file (without extension).
        image_np (np.ndarray): The image data in C x H x W format [0..1].
        output_path (str): The directory to save the image in.
        return_path (bool): If True, returns the full path of the saved image.
    """
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        p = np_to_pil(image_np)
        
        # FIXED: Use os.path.join for cross-platform path compatibility
        full_path = os.path.join(output_path, f"{name}.png")
        
        p.save(full_path)
        
        # FIXED: Return the full path if requested
        if return_path:
            return full_path
            
    except Exception as e:
        print(f"Error saving image {name} to {output_path}: {e}")

    return None
