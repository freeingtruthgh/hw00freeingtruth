#!/usr/bin/env python

import sys
import os
import glob
import getopt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from freeingtruth import deepl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATA_DIR = "/data/CPE_487-587/ACCDataset"

class ACCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)   # (N, 1, 11)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)   # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y": self.y[idx]
        }


def extract_prefix(filepath):
    name = os.path.basename(filepath)
    return name.split("_decoded_")[0]


def collect_pairs(data_dir):
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))

    speed_files = [f for f in all_files if "decoded_wheel_speed_fl.csv" in f]
    acc_files = [f for f in all_files if "decoded_acc_status.csv" in f]

    acc_map = {extract_prefix(f): f for f in acc_files}

    pairs = []
    for sf in speed_files:
        prefix = extract_prefix(sf)
        if prefix in acc_map:
            pairs.append((sf, acc_map[prefix]))

    print(f"Paired experiments: {len(pairs)}")
    return sorted(pairs)


def load_speed(filepath):
    df = pd.read_csv(filepath, usecols=["Time", "Message"]).copy()
    df = df.rename(columns={"Message": "speed_kmh"})
    df["v"] = df["speed_kmh"] / 3.6
    return df[["Time", "v"]].sort_values("Time").reset_index(drop=True)


def load_acc(filepath):
    df = pd.read_csv(filepath, usecols=["Time", "Message"]).copy()
    df = df.rename(columns={"Message": "acc_status"})

    # Duplicate timestamps exist; keep the last observed state
    df = df.drop_duplicates(subset=["Time"], keep="last")

    # Binarize label: enabled (6) => 1, otherwise 0
    df["label"] = (df["acc_status"] == 6).astype(np.float32)
    return df[["Time", "label"]].sort_values("Time").reset_index(drop=True)


def align_data(speed_df, acc_df):
    merged = pd.merge_asof(
        speed_df,
        acc_df,
        on="Time",
        direction="backward"
    )

    merged = merged.dropna(subset=["label"]).reset_index(drop=True)
    return merged


def build_windows(aligned_df, k=10):
    v = aligned_df["v"].to_numpy(dtype=np.float32)
    labels = aligned_df["label"].to_numpy(dtype=np.float32)

    X = []
    y = []

    for t in range(k, len(aligned_df)):
        x_t = v[t-k:t+1][::-1].copy()   # [v_t, v_{t-1}, ..., v_{t-k}]
        y_t = labels[t]

        X.append(x_t)
        y.append(y_t)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_dataset_from_pairs(pairs, k=10):
    X_all = []
    y_all = []

    for speed_file, acc_file in pairs:
        print(f"Processing {os.path.basename(speed_file)}")

        speed_df = load_speed(speed_file)
        acc_df = load_acc(acc_file)
        aligned_df = align_data(speed_df, acc_df)
        X, y = build_windows(aligned_df, k=k)

        X_all.append(X)
        y_all.append(y)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return X_all, y_all


def normalize_data(X_train, X_val):
    mean = X_train.mean()
    std = X_train.std()

    if std < 1e-8:
        std = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    norm_stats = {
        "mean": float(mean),
        "std": float(std)
    }

    return X_train, X_val, norm_stats

def main(argv):
    eta = 0.01
    epoch = 50
    loss = "DiceLoss"
    optimizer = "Adam"
    device = "cuda"
    train_ratio = 0.8
    batch_size = 256
    tag = "accnet"

    try:
        opts, args = getopt.getopt(argv, "hr:e:l:o:D:T:b:t:", ["eta=", "epoch=", "loss=", "optimizer=", "device=", "train_ratio=", "batch_size=", "tag="])

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
            print('\t -b, --batch_size\t\t Batch size for training')
            print('\t -t, --tag\t\t keyword for saved files')
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
        elif opt in ("-b", "--batch_size"):
            val_ratio = int(arg)
        elif opt in ("-t", "--tag"):
            tag = arg

    # Map string to nn function
    loss_map = {
        "DiceLoss": deepl.DiceLoss,
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
    
    # collect paired experiments
    pairs = collect_pairs(DATA_DIR)

    train_pairs, val_pairs = train_test_split(pairs, train_size=train_ratio, random_state=42, shuffle=True)

    print(f"Train experiments: {len(train_pairs)}")
    print(f"Val experiments: {len(val_pairs)}")

    # build datasets
    X_train, y_train = build_dataset_from_pairs(train_pairs, k=10)
    X_val, y_val = build_dataset_from_pairs(val_pairs, k=10)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Train positive ratio: {y_train.mean():.6f}")

    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"Val positive ratio: {y_val.mean():.6f}")

    # normalize using train stats only
    X_train, X_val, norm_stats = normalize_data(X_train, X_val)
    print(f"Normalization stats: {norm_stats}")

    train_dataset = ACCDataset(X_train, y_train)
    val_dataset = ACCDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

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
    model = deepl.ACCNet().to(device)
    trainer = deepl.ACCTrainer(train_loader,val_loader,eta,epoch,loss_fn,optimizer_fn(model.parameters(), lr=eta, weight_decay = 1e-4),model,device)

    # Train ImageNetCNN
    trainer.train()

    # plot loss and accuracy for training and validation
    trainer.evaluation()

    # Save the trained model
    trainer.save()
    
    
    print("Training complete.")



if __name__ == "__main__":
   main(sys.argv[1:])

