import cv2
import numpy as np
import pydicom
from skimage import exposure
import os


def load_slice(path):
    """Read one DICOM slice as float32 pixel values."""
    return pydicom.dcmread(path).pixel_array.astype(np.float32)


def build_reference(paths, max_slices=50):
    """Pool pixel values from training slices only."""
    pixels = []
    for p in paths[:max_slices]:
        pixels.append(load_slice(p).ravel())

    return np.concatenate(pixels)


def preprocess_slice(path, reference, size=224):
    """Histogram-match, resize, and normalize a DICOM slice."""
    img = load_slice(path)

    img = exposure.match_histograms(
        img,
        reference.reshape(-1),
        channel_axis=None
    )

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img)

    return img.astype(np.float32)


def load_series(series_dir, reference, size=224):
    """Stack all slices of one series into (n_slices, size, size)."""
    files = sorted(f for f in os.listdir(series_dir) if f.endswith(".dcm"))
    return np.stack([preprocess_slice(os.path.join(series_dir, f), reference, size) for f in files])
