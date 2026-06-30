import argparse
from pathlib import Path
import numpy as np  
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Tree import Tree
from Hybrid_utils import *
from HybridDT import FlowOCT, HybridDT
import time



if __name__ == "__main__":
    DataDIR = Path.cwd().parent.parent/'DataSets'
    parser = argparse.ArgumentParser(description='Run experiments for Hybrid Decision Tree')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--depth', type=str, default=None)
    parser.add_argument('--timelimit', type=str, default=None)
    parser.add_argument('--lambda', type=str, default=None)
    parser.add_argument('--input_sample', type=str, default=None)
    args = parser.parse_args()

    eta_function = lambda X,lambdaValue : min([ (1 / X.shape[0]) / 2, lambdaValue / 2])
    random_state = 42
    time_limit=300
    #split information
    train_proportion=0.8
    # Load data
    dataset_name = 'compas.csv'
    Datapath = DataDIR/dataset_name
    my_data = Dataset.from_csv(Datapath, dataset_name)
  
    X, y, features, prediction = my_data.split_data_as_dict(
                            {"train" : train_proportion, "test" : 1-train_proportion}, random_state_param = random_state)
    print(X['train'].shape, prediction)
    #convert numpy arrays to dataframes for HyRS and CRL
    df_X = my_data.to_df_from_dict(X)
    

    # Tree structure: We create a tree object of depth d
    if args.depth:
        depth = int(args.depth)
  
    start_time = time.time()
    # Define a hybrid model
    bbox = RandomForestClassifier(random_state=random_state, min_samples_split=10, max_depth=10)

    HybridDT_classifier = HybridDT(black_box_classifier=bbox,depth = 3, lambdaa = 0.0001, eta = eta_function(X['train'], 0.0001),min_transp = 0.7,
                                    estimator = 'FlowOCT', verbosity = ['hybrid'], random_state=random_state, bb_pretrained=False )
    
    HybridDT_classifier.fit(df_X['train'], y['train'], features, time_limit=time_limit)

    end_time = time.time()
    solving_time = end_time - start_time

    HybridDT_classifier.tree_to_string()

    if HybridDT_classifier.check_is_fitted():
        pred ={'train':HybridDT_classifier.predict(df_X['train'])[0], 'test':HybridDT_classifier.predict(df_X['test'])[0]}
        pred_type ={'train':HybridDT_classifier.predict(df_X['train'])[1], 'test':HybridDT_classifier.predict(df_X['test'])[1]}
        # pred,pred_type = HybridDT_classifier.predict(df_X['train'])
        train_acc = get_acc(y['train'], pred['train'])
        test_acc = get_acc(y['test'], pred['test'])
        print("-"*50)
        print(f"training method is {HybridDT_classifier.estimator}")
        print(f"Hybrid DT is trained on {dataset_name} of size {df_X['train'].shape}\ntraining time is {solving_time}")
        print(f"train accuracy is {train_acc:.2f}")
        print(f"test accuracy is {test_acc:.2f}")
        print(f"train transparency ratio is {np.mean(pred_type['train'])}")
        print(f"test transparency ratio is {np.mean(pred_type['test'])}")
        print(f"Number of branch nodes is {np.sum(list(HybridDT_classifier.b.values()))}")
        print(f"Number of leaf nodes is {np.sum(list(HybridDT_classifier.p.values()))}")
        print(f"Optimality Gap is {HybridDT_classifier.optgap}")
        print(f"Status is  {HybridDT_classifier.status}")
        print(f"callback_history: {HybridDT_classifier.callback_history}")
        
    #to draw the tree
    # HybridDT_classifier.plot_tree(filename="my_hybrid_tree.png", show=False)