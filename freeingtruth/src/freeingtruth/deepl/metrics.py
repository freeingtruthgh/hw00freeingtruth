#!/usr/bin/env python

import sys
import os
import glob
import getopt
import numpy as np
import pandas as pd
from PIL import Image

from scipy.ndimage import convolve, uniform_filter
from skimage.feature import graycomatrix


def load_image_grayscale(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def variance_of_laplacian(img):
    lap_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    lap = convolve(img, lap_kernel, mode="reflect")
    return float(np.var(lap))


def tenengrad(img):
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    gx = convolve(img, sobel_x, mode="reflect")
    gy = convolve(img, sobel_y, mode="reflect")
    mag = np.sqrt(gx ** 2 + gy ** 2)

    return float(np.var(mag))


def high_frequency_energy_ratio(img, alpha=0.1):
    h, w = img.shape
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    mag = np.abs(fft_shift)

    cy, cx = h // 2, w // 2
    r = int(alpha * min(h, w))

    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    low_mask = dist2 <= r ** 2
    high_mask = ~low_mask

    total_energy = np.sum(mag)
    if total_energy == 0:
        return 0.0

    high_energy = np.sum(mag[high_mask])
    return float(high_energy / total_energy)


def mean_local_std(img, window_size=7):
    mean = uniform_filter(img, size=window_size, mode="reflect")
    mean_sq = uniform_filter(img ** 2, size=window_size, mode="reflect")
    var = np.maximum(0.0, mean_sq - mean ** 2)
    std = np.sqrt(var)
    return float(np.mean(std))


def glcm_contrast(img, levels=16, distance=1, angle=0):
    quantized = np.floor(img * (levels - 1)).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=[distance],
        angles=[angle],
        levels=levels,
        symmetric=True,
        normed=True
    )

    p = glcm[:, :, 0, 0]
    i = np.arange(levels).reshape(-1, 1)
    j = np.arange(levels).reshape(1, -1)
    contrast = np.sum(((i - j) ** 2) * p)

    return float(contrast)


def compute_metrics_for_image(path):
    img = load_image_grayscale(path)

    return {
        "image_path": path,
        "variance_of_laplacian": variance_of_laplacian(img),
        "tenengrad": tenengrad(img),
        "high_frequency_energy_ratio": high_frequency_energy_ratio(img),
        "mean_local_std": mean_local_std(img),
        "glcm_contrast": glcm_contrast(img),
    }


def summarize_metrics(df):
    summary = df.drop(columns=["image_path"]).agg(["mean", "std", "min", "max"])
    return summary


def main(argv):
    image_dir = "results/Images"
    tag = "gan"
    csv_path = None
    summary_csv_path = None

    try:
        opts, args = getopt.getopt(
            argv,
            "hi:t:c:s:",
            ["image_dir=", "tag=", "csv_path=", "summary_csv_path="]
        )
    except getopt.GetoptError:
        print(f"Check options by typing:\n{__file__} -h")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(f"\n{__file__} [OPTIONS]")
            print("\t-h, --help\t\tGet help")
            print("\t-i, --image_dir\t\tDirectory containing generated images")
            print("\t-t, --tag\t\tTag used for output filenames")
            print("\t-c, --csv_path\t\tPath to save per-image metrics CSV")
            print("\t-s, --summary_csv_path\tPath to save summary metrics CSV")
            sys.exit()

        elif opt in ("-i", "--image_dir"):
            image_dir = arg
        elif opt in ("-t", "--tag"):
            tag = arg
        elif opt in ("-c", "--csv_path"):
            csv_path = arg
        elif opt in ("-s", "--summary_csv_path"):
            summary_csv_path = arg

    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory does not exist: {image_dir}")

    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in directory: {image_dir}")

    rows = []
    for path in image_paths:
        rows.append(compute_metrics_for_image(path))

    df = pd.DataFrame(rows)
    summary = summarize_metrics(df)

    if csv_path is None:
        csv_path = os.path.join(image_dir, f"{tag}_metrics.csv")

    if summary_csv_path is None:
        summary_csv_path = os.path.join(image_dir, f"{tag}_metrics_summary.csv")

    df.to_csv(csv_path, index=False)
    summary.to_csv(summary_csv_path)

    print(f"Saved per-image metrics to {csv_path}")
    print(f"Saved summary metrics to {summary_csv_path}")
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1:])