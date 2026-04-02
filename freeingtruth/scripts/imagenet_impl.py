#!/usr/bin/env python

import sys
import getopt
from freeingtruth import deepl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
from zoneinfo import ZoneInfo
from datasets import load_dataset
from datasets import load_from_disk
import subprocess
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

def main(argv):
    eta = 0.01
    epoch = 10000
    loss = "CrossEntropyLoss"
    optimizer = "SGD"
    device = "cuda"
    train_ratio = 0.008
    val_ratio = 0.002
    tag = "hw03"

    try:
        opts, args = getopt.getopt(argv, "hr:e:l:o:D:T:v:t:", ["eta=", "epoch=", "loss=", "optimizer=", "device=", "train_ratio=", "val_ratio=", "tag="])

    except getopt.GetoptError:
        print('Check options by typing:\n{} -h'.format(__file__))
        sys.exit(2)

    print("OPTS: {}".format(opts))
    for opt, arg in opts:
        if opt == '-h':
            print('\n{} [OPTIONS]'.format(__file__))
            print('\t -h, --help\t\t Get help')
            print('\t -r, --eta\t Learning rate')
            print('\t -e, --epoch\t\t Number of training epochs')
            print('\t -l, --loss\t Loss function (BCE, BCELogitsLoss, MSE)')
            print('\t -o, --optimizer\t Optimizer function (Adam, SGD)')
            print('\t -D, --device\t\t Device (cpu, cuda)')
            print('\t -T, --train_ratio\t\t ratio of training dataset to use for training')
            print('\t -v, --val_ratio\t\t ratio of validation dataset to use for training')
            print('\t -t, --tag\t\t keyword for CVS filename')
            sys.exit()
        elif opt in ("-r", "--eta"):
            eta = float(arg)
        elif opt in ("-e", "--epoch"):
            epoch = int(arg)
        elif opt in ("-l", "--loss"):
            loss = arg
        elif opt in ("-o", "--optimizer"):
            optimizer = arg
        elif opt in ("-D", "--device"):
            device = arg
        elif opt in ("-T", "--train_ratio"):
            train_ratio = float(arg)
        elif opt in ("-v", "--val_ratio"):
            val_ratio = float(arg)
        elif opt in ("-t", "--tag"):
            tag = arg

    # Map string to nn function
    loss_map = {
        "CrossEntropyLoss": nn.CrossEntropyLoss,
        "BCELoss": nn.BCELoss,
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
        "MSELoss": nn.MSELoss
    }

    optimizer_map = {
        "Adam": optim.Adam,
        "SGD": optim.SGD 
    }

    loss_fn = loss_map[loss]()
    optimizer_fn = optimizer_map[optimizer]
    
    # Load dataset
    # dataset = load_dataset(
    #     "ILSVRC/imagenet-1k",
    #     cache_dir = "/data/CPE_487-587/imagenet-1k"
    # )
    dataset = load_from_disk("/data/CPE_487-587/imagenet-1k-arrow")

    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    num_classes = len(train_dataset.features['label'].names)
    print(f"Number of classes: {num_classes}")

    # Select subset of training and validation sample
    train_size = int(len(dataset['train'])*train_ratio)
    val_size = int(len(dataset['validation'])* val_ratio)

    train_dataset = dataset['train'].select(range(train_size))
    val_dataset = dataset['validation'].select(range(val_size))

    # Transformer images
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Apply transforms
    def preprocess_train(examples):
        images = [train_transform(img.convert('RGB')) for img in examples['image']]
        labels = examples['label']

        return{
            'pixel_values': images,
            'labels': labels
        }

    def preprocess_val(examples):
        images = [val_transform(img.convert('RGB')) for img in examples['image']]
        labels = examples['label']

        return{
            'pixel_values': images,
            'labels': labels
        }

    train_dataset = train_dataset.with_transform(preprocess_train)
    val_dataset = val_dataset.with_transform(preprocess_val)

    def collate_fn(batch):
        # Extract pixel_values and labels for each item
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])

        return {
            'pixel_values': pixel_values,
            'labels': labels
        }

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size = 128,
        shuffle = True,
        pin_memory=True,    # Important for faster GPU transfer
        collate_fn = collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = 128,
        shuffle=False,
        pin_memory=True,
    )

    # save an example training and validation image:
    train_example = train_dataset[0]
    val_example = val_dataset[0]
    torch.save(train_example, f"results/{tag}_train_example.pt")
    torch.save(val_example, f"results/{tag}_val_example.pt")

    # Choose least utilizated GPU _____________________________________________________________________________________
    # def get_best_gpu(strategy="utilization"):
    #     """
    #     Select best GPU by 'utilization' or 'memory'.
    #     """
    #     if strategy == "memory":
    #         # Use Pytorch directly for free memory
    #         free_mem = []
    #         for i in range(torch.cuda.device_count()):
    #             props = torch.cuda.mem_get_info(i)  # (free, total)
    #             free_mem.append(props[0])
    #         return free_mem.index(max(free_mem))

    #     elif strategy == "utilization":
    #         result = subprocess.run(
    #             ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
    #             capture_output=True, text=True
    #         )
    #         utilizations = [int(x.strip()) for x in result.stdout.strip().split("\n")]
    #         return utilizations.index(min(utilizations))
        
    # # Pick strategy: "utilization" or "memory"
    # device_id = get_best_gpu(strategy="utilization")
    # device = torch.device(f"cuda:{device_id}")
    # print(f"Selected GPU: {device_id}")

    device = torch.device("cuda")
    
    # instantiate ImageNetCNN and CNNTrainer
    model = deepl.ImageNetCNN(num_classes).to(device)
    trainer = deepl.CNNTrainer(train_loader,val_loader,eta,epoch,loss_fn,optimizer_fn(model.parameters(), lr=eta, momentum = 0.9, weight_decay = 1e-4),model,device)

    # Train ImageNetCNN
    trainer.train()

    # plot loss and accuracy for training and validation
    trainer.evaluation()

    # Save the trained model
    trainer.save()
    
    
    print("Training complete.")



if __name__ == "__main__":
   main(sys.argv[1:])

