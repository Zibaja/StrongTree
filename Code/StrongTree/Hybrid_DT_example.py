import argparse
from pathlib import Path
import numpy as np  
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Tree import Tree
from Hybrid_utils import *


if __name__ == "__main__":
    DataDIR = Path.cwd().parent.parent/'DataSets'
    random_state = 41
    parser = argparse.ArgumentParser(description='Run experiments for Hybrid Decision Tree')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--depth', type=str, default=None)
    parser.add_argument('--timelimit', type=str, default=None)
    parser.add_argument('--lambda', type=str, default=None)
    parser.add_argument('--input_sample', type=str, default=None)
    args = parser.parse_args()
    # for i in args.__dict__:
    #     print(i, args.__dict__[i])

    random_state = 41
    #split information
    train_proportion=0.8
    # Load data
    dataset_name = 'monk1_enc.csv'
    Datapath = DataDIR/dataset_name
    my_data = Dataset.from_csv(Datapath, dataset_name)
  
    X, y, features, prediction = my_data.split_data_as_dict(
                            {"train" : train_proportion, "test" : 1-train_proportion}, random_state_param = random_state)
    print(X['train'].shape, prediction)
    #convert numpy arrays to dataframes for HyRS and CRL
    df_X = my_data.to_df_from_dict(X)


    # Tree structure: We create a tree object of depth d
    depth = int(args.depth)
    tree = Tree(depth)

    # Set parameters


    # Define a hybrid model
    bbox = RandomForestClassifier(random_state=42, min_samples_split=10, max_depth=10)

