from pathlib import Path
import numpy as np
from PIL import Image


def extract_image_data(file_path: str | Path, size: tuple[int, int] = (224, 224)) -> dict:
    """
    Converts an image file into:
    1. A normalized RGB number grid (spatial tensor) of shape (H, W, 3)
    2. A 2D frequency spectrum map (for viewing repeating/frequency patterns)
    """ 
    path = Path(file_path)

    # Open image and force 3 color channels: Red, Green, Blue ("RGB")
    img = Image.open(path).convert("RGB")
    original_size = img.size  # (width, height)

    # Resize to a fixed size like 224x224 so every image has the same shape
    resized_img = img.resize(size)

    # Turn pixels (0 to 255) into decimals (0.0 to 1.0) inside a NumPy array
    spatial_tensor = np.asarray(resized_img, dtype=np.float32) / 255.0

    # Convert to grayscale (single brightness grid) for frequency analysis
    gray = np.asarray(resized_img.convert("L"), dtype=np.float32)

    # 2D Fast Fourier Transform -> shift zero-frequency to center -> log scale for visibility
    fft_2d = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft_2d)
    frequency_map = np.log1p(np.abs(fft_shifted))

    # Scale frequency_map to 0.0 - 1.0 so Streamlit can display it cleanly as an image
    max_val = frequency_map.max()
    if max_val > 0:
        frequency_vis = frequency_map / max_val
    else:
        frequency_vis = frequency_map

    return {
        "original_size": original_size,
        "tensor_shape": spatial_tensor.shape,
        "spatial_tensor": spatial_tensor,
        "frequency_map": frequency_map,
        "frequency_vis": frequency_vis,
    }