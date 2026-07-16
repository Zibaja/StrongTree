import argparse
from pathlib import Path
import numpy as np  
from sklearn.ensemble import RandomForestClassifier
from Hybrid_utils import *
# from HybridDTCompletenew_with_CPSAT import  HybridDT
from HybridDTCompletenew import  HybridDT
import time
from black_box_models import BlackBox


if __name__ == "__main__":

    estimators = ['FlowOCT', 'BendersOCT', 'CPSATOCT']
    DataDIR = Path.cwd().parent.parent/'DataSets'
    parser = argparse.ArgumentParser(description='Run experiments for Hybrid Decision Tree')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--depth', type=str, default=None)
    parser.add_argument('--timelimit', type=str, default=None)
    parser.add_argument('--lambdaa', type=str, default=None)
    parser.add_argument('--min_transp', type=str, default=None) #here , we either pass eta or min_transp
    parser.add_argument('--eta', type=str, default=None)
    parser.add_argument('--seed', type=str, default=None)
    parser.add_argument('--estimator_id', type=int, default=None)
    args = parser.parse_args()

    eta_function = lambda X,lambdaValue : min([ (1 / X.shape[0]) / 2, lambdaValue / 2])



    seed = int(args.seed)
    random_state = 42 + int(args.seed)
    dataset_name = str(args.dataset)
    depth = int(args.depth)
    lambdaa = float(args.lambdaa)
    estimator = estimators[args.estimator_id]
    if args.eta :
        eta = float(args.eta)
    if args.min_transp:
        min_transp = float(args.min_transp)
    time_limit = 1800

    #split data into three parts, train, test and validation of the same size
    splits = {"train": 2000, "test":2000, "validation":2000}
    # Load data
    Datapath = DataDIR/f'{dataset_name}.csv'
    my_data = Dataset.from_csv(Datapath, dataset_name)
  
    X, y, features, prediction = my_data.split_data_as_dict_withsize(splits,
                             random_state_param = seed)
    print(X['train'].shape, prediction)

    df_X = my_data.to_df_from_dict(X)
    
    #set the Parametres:
    h_params = {
            "depth":depth,
            "lambdaa": lambdaa,
            "eta" : eta if estimator=='BendersOCT' else eta_function(X['train'], lambdaa),
        }
    if estimator in ['FlowOCT', 'CPSATOCT']:
        h_params.update({"min_transp": min_transp})

  
    # Retrieve the BB
    bbox_type = 'random_forest'
    model_path = Path("models") / f"{dataset_name}_{bbox_type}_{seed}.pickle"
    if not model_path.exists():
        raise ValueError(f"Black box model not found at {model_path}. Please ensure the black box is trained and saved before running this experiment.")

    print("Loading the Black Box")

    bbox = BlackBox(bb_type=bbox_type).load(model_path) # it is an object from out BlackBox class, which has a predict method and a fit method. It is not the sklearn object.


    start_time = time.time()
    print(f"fit the interpretable part using {estimator}")

    HybridDT_classifier = HybridDT(black_box_classifier=bbox,**h_params,
                                    estimator = estimator, verbosity = ['hybrid'], random_state=random_state, bb_pretrained=True ) # if you only path the bb object, bb_pretrained= False, so it will be fit inside the fitting of HybridDT
    
    HybridDT_classifier.fit(df_X['train'], y['train'], features, time_limit=time_limit)

    end_time = time.time()
    solving_time = end_time - start_time

    HybridDT_classifier.tree_to_string()

    if HybridDT_classifier.check_is_fitted():
        pred =  {k: HybridDT_classifier.predict(df_X[k])[0] for k in splits.keys()}
        pred_type = {k: HybridDT_classifier.predict(df_X[k])[1] for k in splits.keys()}
        acc = {k: get_acc(y[k], pred[k]) for k in splits.keys()}
        transp_ratio= {k: np.mean(pred_type[k]) for k in splits.keys()}
        print("-"*50)
        print(f"training method is {HybridDT_classifier.estimator}")
        print(f"Hybrid DT is trained on {dataset_name} of size {df_X['train'].shape}\ntraining time is {solving_time}")
        print(f"stand alone black box train acc is {HybridDT_classifier.black_box_train_acc:.2f}")
        print(f"train accuracy is {acc['train']:.2f}")
        print(f"test accuracy is {acc['test']:.2f}")
        print(f"train transparency ratio is {transp_ratio['train']}")
        print(f"test transparency ratio is {transp_ratio['test']}")
        print(f"Number of branch nodes is {np.sum(list(HybridDT_classifier.b.values()))}")
        print(f"Number of leaf nodes is {np.sum(list(HybridDT_classifier.p.values()))}")
        print(f"Optimality Gap is {HybridDT_classifier.optgap}")
        print(f"Status is  {HybridDT_classifier.status}")

  


        
    #to draw the tree
    # HybridDT_classifier.plot_tree(filename="my_hybrid_tree.png", show=False)