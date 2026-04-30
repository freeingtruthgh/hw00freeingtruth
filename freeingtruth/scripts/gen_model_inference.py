#!/usr/bin/env python

import sys
import getopt
import os
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from freeingtruth import deepl


def load_onnx_session(onnx_path):
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def denormalize_images(images):
    """
    images expected in [-1, 1], shape (N, C, H, W)
    returns images in [0, 1], shape (N, H, W, C)
    """
    images = np.nan_to_num(images, nan=0.0, posinf=1.0, neginf=-1.0)
    images = (images + 1.0) / 2.0
    images = np.clip(images, 0.0, 1.0)
    images = np.transpose(images, (0, 2, 3, 1))
    return images


def save_grid(images, filename, title="Generated Images"):
    """
    images: numpy array, shape (25, H, W, C), values in [0, 1]
    """
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        if images.shape[-1] == 1:
            ax.imshow(images[i].squeeze(), cmap="gray")
        else:
            ax.imshow(images[i])
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_individual_images(images, out_dir, tag):
    """
    images: numpy array of shape (N, H, W, C), values in [0, 1]
    """
    os.makedirs(out_dir, exist_ok=True)

    for i, img in enumerate(images):
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
        img_path = os.path.join(out_dir, f"{tag}_{i:02d}.png")
        Image.fromarray(img_uint8).save(img_path)


def generate_gan_images(session, latent_dim=128, n_samples=25):
    z = np.random.randn(n_samples, latent_dim).astype(np.float32)

    generated = session.run(
        ["generated_image"],
        {"latent_vector": z}
    )[0]

    return generated


def generate_vae_images(session, latent_dim=128, n_samples=25):
    z = np.random.randn(n_samples, latent_dim).astype(np.float32)

    generated = session.run(
        ["generated_image"],
        {"latent_vector": z}
    )[0]

    return generated


def linear_beta_schedule(n_steps=1000, min_beta=1e-4, max_beta=0.02):
    betas = np.linspace(min_beta, max_beta, n_steps, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars


def generate_diffusion_images(session, n_samples=25, c=3, h=64, w=64, n_steps=1000, min_beta=1e-4, max_beta=0.02,):
    betas, alphas, alpha_bars = linear_beta_schedule(n_steps=n_steps, min_beta=min_beta, max_beta=max_beta,)

    x = np.random.randn(n_samples, c, h, w).astype(np.float32)

    for t in reversed(range(n_steps)):
        t_batch = np.full((n_samples,), t, dtype=np.int64)

        eta_theta = session.run(["predicted_noise"], {"noisy_image": x.astype(np.float32), "timestep": t_batch,})[0]

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]

        x = (1.0 / np.sqrt(alpha_t)) * (x - ((1.0 - alpha_t) / np.sqrt(1.0 - alpha_bar_t)) * eta_theta)

        if t > 0:
            z = np.random.randn(n_samples, c, h, w).astype(np.float32)
            beta_t = betas[t]
            sigma_t = np.sqrt(beta_t)
            x = x + sigma_t * z

    return x

def normalize_metric(values):
    values = np.array(values)
    return (values - values.mean()) / (values.std() + 1e-8)


def plot_metrics_box(df, save_path, title):
    metrics = ["variance_of_laplacian", "tenengrad", "high_frequency_energy_ratio", "mean_local_std", "glcm_contrast"]

    data = []
    for m in metrics:
        data.append(normalize_metric(df[m].values))

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, tick_labels=[
        "VoL", "Ten", "HFER", "MLSD", "GLCM"
    ])

    plt.title(title)
    plt.ylabel("Normalized Metric Value")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(argv):
    model_type = "GAN"
    onnx_path = "gan_final_model.onnx"
    latent_dim = 128
    n_samples = 25
    n_steps = 1000
    min_beta = 1e-4
    max_beta = 0.02
    image_size = 64
    out_dir = "results/Images"
    tag = "gan"

    try:
        opts, args = getopt.getopt(argv,"hm:p:l:n:s:b:B:i:o:t:",["model_type=","onnx_path=","latent_dim=","n_samples=","n_steps=","min_beta=","max_beta=","image_size=","out_dir=","tag=",])
    
    except getopt.GetoptError:
        print(f"Check options by typing:\n{__file__} -h")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(f"\n{__file__} [OPTIONS]")
            print("\t -h, --help\t\t Get help")
            print("\t -m, --model_type\t Model type (VAE, GAN, DIFFUSION)")
            print("\t -p, --onnx_path\t Path to ONNX model")
            print("\t -l, --latent_dim\t Latent dimension for GAN/VAE")
            print("\t -n, --n_samples\t Number of images to generate")
            print("\t -s, --n_steps\t\t Diffusion steps")
            print("\t -b, --min_beta\t\t Diffusion minimum beta")
            print("\t -B, --max_beta\t\t Diffusion maximum beta")
            print("\t -i, --image_size\t Image size (default 64)")
            print("\t -o, --out_dir\t\t Output directory")
            print("\t -t, --tag\t\t Output filename tag")
            sys.exit()

        elif opt in ("-m", "--model_type"):
            model_type = arg.upper()
        elif opt in ("-p", "--onnx_path"):
            onnx_path = arg
        elif opt in ("-l", "--latent_dim"):
            latent_dim = int(arg)
        elif opt in ("-n", "--n_samples"):
            n_samples = int(arg)
        elif opt in ("-s", "--n_steps"):
            n_steps = int(arg)
        elif opt in ("-b", "--min_beta"):
            min_beta = float(arg)
        elif opt in ("-B", "--max_beta"):
            max_beta = float(arg)
        elif opt in ("-i", "--image_size"):
            image_size = int(arg)
        elif opt in ("-o", "--out_dir"):
            out_dir = arg
        elif opt in ("-t", "--tag"):
            tag = arg

    os.makedirs(out_dir, exist_ok=True)

    session = load_onnx_session(onnx_path)

    if model_type == "GAN":
        generated = generate_gan_images(session=session, latent_dim=latent_dim, n_samples=n_samples,)

    elif model_type == "VAE":
        generated = generate_vae_images(session=session, latent_dim=latent_dim, n_samples=n_samples,)

    elif model_type == "DIFFUSION":
        generated = generate_diffusion_images(session=session, n_samples=n_samples, c=3, h=image_size, w=image_size, n_steps=n_steps, min_beta=min_beta, max_beta=max_beta,)

    else:
        raise ValueError("Unsupported model_type")

    generated = denormalize_images(generated)

    grid_filename = os.path.join(out_dir, f"{tag}_grid.png")
    save_grid(generated, grid_filename, title=f"{model_type} Generated Images")

    images_dir = os.path.join(out_dir, f"{tag}_samples")
    save_individual_images(generated, images_dir, tag)

    print(f"Saved generated image grid to {grid_filename}")
    print(f"Saved {n_samples} individual images to {images_dir}")

    rows = []
    for i in range(n_samples):
        img_path = os.path.join(images_dir, f"{tag}_{i:02d}.png")
        rows.append(deepl.compute_metrics_for_image(img_path))

    df = pd.DataFrame(rows)
    summary = deepl.summarize_metrics(df)

    metrics_csv = os.path.join(out_dir, f"{tag}_metrics.csv")
    summary_csv = os.path.join(out_dir, f"{tag}_metrics_summary.csv")
    plot_path = os.path.join(out_dir, f"{tag}_metrics_boxplot.png")

    df.to_csv(metrics_csv, index=False)
    summary.to_csv(summary_csv)
    plot_metrics_box(df, plot_path, title=f"{model_type} Metrics Box Plot")
    
    print(f"Saved per-image metrics to {metrics_csv}")
    print(f"Saved summary metrics to {summary_csv}")
    print(f"Saved metrics box plot to {plot_path}")

if __name__ == "__main__":
    main(sys.argv[1:])