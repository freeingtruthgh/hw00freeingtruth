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

def main(argv):
    data_file_location = "../data/Android_Malware.csv"
    eta = 0.01
    epoch = 5000
    loss = "CrossEntropyLoss"
    optimizer = "Adam"
    device = "cuda"
    tag = "hw02"

    try:
        opts, args = getopt.getopt(argv, "hd:r:e:l:o:D:t:", ["data_file_location=", "eta=", "epoch=", "loss=", "optimizer=", "device=", "tag="])

    except getopt.GetoptError:
        print('Check options by typing:\n{} -h'.format(__file__))
        sys.exit(2)

    print("OPTS: {}".format(opts))
    for opt, arg in opts:
        if opt == '-h':
            print('\n{} [OPTIONS]'.format(__file__))
            print('\t -h, --help\t\t Get help')
            print('\t -d, --data_file_location\t Path to data file')
            print('\t -r, --eta\t Learning rate')
            print('\t -e, --epoch\t\t Number of training epochs')
            print('\t -l, --loss\t Loss function (BCE, BCELogitsLoss, MSE)')
            print('\t -o, --optimizer\t Optimizer function (Adam, SGD)')
            print('\t -D, --device\t\t Device (cpu, cuda)')
            print('\t -t, --tag\t\t keyword for CVS filename')
            sys.exit()
        elif opt in ("-d", "--data_file_location"):
            data_file_location = arg
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
    
    # Read data
    df = pd.read_csv(data_file_location, low_memory=False)
    # filter out features not helpful in classification:
    Filter_out_columns = ["Flow ID"," Source IP"," Source Port"," Destination Port"," Protocol"," Timestamp"]
    df_filtered = df.drop(columns=Filter_out_columns)
    x = df_filtered.iloc[:,1:-1]
    x = x.apply(pd.to_numeric, errors="coerce")     # Force numeric conversion for all features, i.e (BENIGN, SCAREWARE, or other to NaN)
    x = x.fillna(x.mean())      # Use mean to replace NaN
    y = df_filtered.iloc[:,-1]
    # Change label from string to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # define num_class
    num_class = len(le.classes_)
    # Convert to tensor
    X_tensor = torch.tensor(x.values,dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(y_encoded,dtype=torch.torch.long).to(device)
    # Split into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42)
    # define d_features
    d_features = X_train.size(1)
    
    # instantiate SimpleNN and ClassTrainer
    model = deepl.SimpleNN(num_class,d_features).to(device)
    trainer = deepl.ClassTrainer(X_train,Y_train,eta,epoch,loss_fn,optimizer_fn,model,device)
    trainer.train()
    trainer.test(X_test, Y_test)
    train_f1, train_precision, train_recall, train_accuracy, test_f1, test_precision, test_recall, test_accuracy = trainer.evaluation()

    # Save final Accuracy, F1 score, precision, and recall for both train and test data 
    nme = ["Trained", "Test"]
    acc = [train_accuracy, test_accuracy]
    f1 = [train_f1, test_f1]
    pre = [train_precision, test_precision]
    re = [train_recall, test_recall]
    data = {'Data': nme, 'Accuracy': acc, 'F1 Score': f1, 'Precision': pre, 'Recall': re}
    data_file = pd.DataFrame(data)
    timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y%m%d%H%M%S")
    csv_filename = f"{tag}_{timestamp}.csv"
    data_file.to_csv(csv_filename, index=False)
    
    
    print("Training complete.")



if __name__ == "__main__":
   main(sys.argv[1:])
    