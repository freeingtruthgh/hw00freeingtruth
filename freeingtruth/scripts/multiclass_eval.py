#!/usr/bin/env python

import sys
import getopt
import pandas as pd
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from zoneinfo import ZoneInfo

def main(argv):
    tag = "hw02"

    try:
        opts, args = getopt.getopt(argv, "ht:", ["tag="])

    except getopt.GetoptError:
        print('Check options by typing:\n{} -h'.format(__file__))
        sys.exit(2)

    print("OPTS: {}".format(opts))
    for opt, arg in opts:
        if opt == '-h':
            print('\n{} [OPTIONS]'.format(__file__))
            print('\t -h, --help\t\t Get help')
            print('\t -t, --tag\t\t keyword for CSV filename')
            sys.exit()
        elif opt in ("-t", "--tag"):
            tag = arg

    # find all CSV files with the keyword
    csv_files = glob.glob(f"*{tag}*.csv")
    print(f"Found {len(csv_files)} files")
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found for tag '{tag}")
    
    
    # Aggregate all training and test metrics
    df_list = []

    for filename in csv_files:
        df = pd.read_csv(filename)
        df_list.append(df)

    aggregated_df = pd.concat(df_list, ignore_index=True)
    
    # Produce Boxplot of Accuracy, F1 score, Precision, and Recall for both training and test
    box = aggregated_df.boxplot(column=["Accuracy", "F1 Score", "Precision", "Recall"])
    fig = box.get_figure()
    timestamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y%m%d%H%M%S")
    fig_filename = f"{tag}_{timestamp}.pdf"
    fig.savefig(fig_filename)

    print(f"Saved boxplot to {fig_filename}")

if __name__ == "__main__":
    main(sys.argv[1:])
