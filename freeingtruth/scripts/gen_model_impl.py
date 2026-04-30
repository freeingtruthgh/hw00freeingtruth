#!/usr/bin/env python

import sys
import getopt
from freeingtruth import deepl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import zipfile
import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms

class CelebAZipDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform

        # Open zip once to collect all image filenames
        with zipfile.ZipFile(zip_path, 'r') as zf:
            self.image_names = sorted([
                name for name in zf.namelist()
                if name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Re-open zip per worker (required for DataLoader multiprocessing)
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open(self.image_names[idx]) as f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img


def main(argv):
    eta = 0.01
    epochs = 10
    loss = "BCELoss"
    optimizer = "Adam"
    model_type = "GAN"
    device = "cuda"
    save_epoch = 5
    train_ratio = 1.0
    val_ratio = 1.0

    try:
        opts, args = getopt.getopt(argv, "hr:e:l:o:m:D:x:T:v:", ["eta=", "epochs=", "loss=", "optimizer=", "model_type=", "device=", "save_epoch=", "train_ratio=", "val_ratio="])

    except getopt.GetoptError:
        print('Check options by typing:\n{} -h'.format(__file__))
        sys.exit(2)

    print("OPTS: {}".format(opts))
    for opt, arg in opts:
        if opt == '-h':
            print('\n{} [OPTIONS]'.format(__file__))
            print('\t -h, --help\t\t Get help')
            print('\t -r, --eta\t Learning rate')
            print('\t -e, --epochs\t\t Number of training epochs')
            print('\t -l, --loss\t Loss function (BCELoss, BCEWithLogitsLoss, MSELoss, VAELoss)')
            print('\t -o, --optimizer\t Optimizer function (Adam, SGD)')
            print('\t -m, --model_type\t Type of model to train (VAE, GAN, DIFFUSION)')
            print('\t -D, --device\t\t Device (cpu, cuda)')
            print('\t -x, --save_epoch\t\t save intermediate ONNX file every x epochs')
            print('\t -T, --train_ratio\t\t ratio of training dataset to use for training')
            print('\t -v, --val_ratio\t\t ratio of validation dataset to use for training')
            sys.exit()
        elif opt in ("-r", "--eta"):
            eta = float(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-l", "--loss"):
            loss = arg
        elif opt in ("-o", "--optimizer"):
            optimizer = arg
        elif opt in ("-m", "--model_type"):
            model_type = arg.upper()
        elif opt in ("-D", "--device"):
            device = arg
        elif opt in ("-x", "--save_epoch"):
            save_epoch = int(arg)
        elif opt in ("-T", "--train_ratio"):
            train_ratio = float(arg)
        elif opt in ("-v", "--val_ratio"):
            val_ratio = float(arg)

    # Map string to nn function
    loss_map = {
        "CrossEntropyLoss": nn.CrossEntropyLoss,
        "BCELoss": nn.BCELoss,
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
        "MSELoss": nn.MSELoss,
        "VAELoss": deepl.VAELoss,
    }

    optimizer_map = {
        "Adam": optim.Adam,
        "SGD": optim.SGD 
    }

    loss_fn = loss_map[loss]()
    optimizer_cls = optimizer_map[optimizer]

    device = torch.device(device)
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # to [-1, 1]
    ])

    dataset = CelebAZipDataset(
        zip_path='/data/CPE_487-587/img_align_celeba.zip',
        transform=transform
    )

    # ---------------------------------------------------
    # First split full dataset into 80% train / 20% val
    # ---------------------------------------------------
    total_size = len(dataset)
    base_train_size = int(0.8 * total_size)
    base_val_size = total_size - base_train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [base_train_size, base_val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # ---------------------------------------------------------------------------
    # Select subset of training and validation sample
    train_size = int(len(train_dataset)*train_ratio)
    val_size = int(len(val_dataset)*val_ratio)

    train_dataset = Subset(train_dataset, range(train_size))
    val_dataset = Subset(val_dataset, range(val_size))

    print(f"Full dataset size: {total_size}")
    print(f"Base train split size: {base_train_size}")
    print(f"Base val split size: {base_val_size}")
    print(f"Final train subset size: {len(train_dataset)}")
    print(f"Final val subset size: {len(val_dataset)}")
    # ---------------------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    batch = next(iter(train_loader))
    print(f"Batch shape: {batch.shape}")
    print(f"Value range: [{batch.min():.2f}, {batch.max():.2f}]")
    
    # instantiate model and GenModelTrainer
    if model_type == "VAE":
        model = deepl.VAE().to(device)
        optimizer = optimizer_cls(model.parameters(), lr=eta)
    elif model_type == "DIFFUSION":
        model = deepl.DDPM(device=device)
        optimizer = optimizer_cls(model.parameters(), lr=eta)
    elif model_type == "GAN":
        model = deepl.Discriminator().to(device)
        model2 = deepl.Generator().to(device)
        if optimizer == "Adam":
            optimizer_D = optimizer_cls(model.parameters(), lr=eta, betas=(0.5, 0.999))
            optimizer_G = optimizer_cls(model2.parameters(), lr=eta, betas=(0.5, 0.999))
        else:
            optimizer_D = optimizer_cls(model.parameters(), lr=eta)
            optimizer_G = optimizer_cls(model2.parameters(), lr=eta)
    else:
        raise ValueError("Unsupported model_type")

    if model_type == "GAN":
        trainer = deepl.GenModelTrainer(train_loader,val_loader,eta,epochs,loss_fn,optimizer_D,optimizer_G,model_type,model,model2,device, x=save_epoch)
    else:
        trainer = deepl.GenModelTrainer(train_loader,val_loader,eta,epochs,loss_fn,optimizer,model_type=model_type,model=model,device=device,x=save_epoch)

    # Train model and save ONNX file
    trainer.train()
    
    
    print("Training complete.")



if __name__ == "__main__":
   main(sys.argv[1:])