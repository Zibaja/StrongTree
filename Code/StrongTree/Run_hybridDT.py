
import argparse
from pathlib import Path
import numpy as np  
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Tree import Tree
from Hybrid_utils import *
# from HybridDT import FlowOCT, HybridDT 
from HybridDTCompletenew import  HybridDT
import time
import subprocess
import pickle
from black_box_models import BlackBox

ESTIMATORS = {

    "HybridDTClassifier_post": {
        "build": lambda bbox, h: HybridDT(
            black_box_classifier=bbox,
            depth = h["depth"],
            lambdaa=h["lambdaa"],
            eta = h["eta"],
            min_transp=h["min_transp"],
            estimator=h["estimator"],
           
        ),
        "fit": lambda model, X, y, h: model.fit(X, y, features=h["features"],time_limit=h["time_limit"]),
        "hparams": {
            "depth":[1,2,3,4,5],
            "lambdaa": [10**-2, 10**-3, 10**-4],
            "eta" : [0.001, 0.00215443, 0.00464159, 0.01, 0.02154435,
                        0.04641589, 0.1, 0.21544347, 0.46415888, 1.0],
            "min_transp": [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95]
        }
    } }

eta_function = lambda X,lambdaValue : min([ (1 / X.shape[0]) / 2, lambdaValue / 2])

n_seeds = 5
DATASETS = ["compas", "adult", "acs_employ"]
SEEDS = [1,2,3,4,5]


EXPERIMENTS = {k:[] for k in ['FlowOCT', 'BendersOCT']}

for dataset in DATASETS:
    for seed in SEEDS:
        for model in ESTIMATORS:
            for d in ESTIMATORS[model]['hparams']['depth']:
                for lambdaa_val in ESTIMATORS[model]['hparams']['lambdaa']:
                        for alpha_val in ESTIMATORS[model]['hparams']['min_transp']:
                            EXPERIMENTS['FlowOCT'].append({
                                "dataset_name": dataset,
                                "model": model,
                                "depth":d ,
                                "lambdaa": lambdaa_val,
                                "min_transp": alpha_val,
                                "seed": seed
                            })
                        for eta_val in ESTIMATORS[model]['hparams']['eta']:
                            EXPERIMENTS['BendersOCT'].append({
                                "dataset_name": dataset,
                                "model": model,
                                "depth":d ,
                                "lambdaa": lambdaa_val,
                                "eta": eta_val,
                                "seed": seed
                            })


# print("number of experiments is",len(EXPERIMENTS['BendersOCT']))
#print(EXPERIMENTS['BendersOCT'])




def run_one_experiment(model, seed,dataset_name, depth, lambdaa,min_transp=None,eta=None,estimator='FlowOCT', time_limit=3600):

    DataDIR = Path.cwd().parent.parent/'DataSets'
 
    random_state = 42 + seed
    #split information
    # train_proportion=0.8 
    #{"train" : train_proportion, "test" : 1-train_proportion},

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
            "eta" : eta_function(X['train'], lambdaa) if estimator=='FlowOCT' else eta,
            "min_transp": min_transp
        }

    start_time = time.time()

    # Define a hybrid model
    # bbox = RandomForestClassifier(random_state=random_state, min_samples_split=10, max_depth=10)



    # Retrieve the BB
    bbox_type = 'random_forest'
    model_path = Path("models") / f"{dataset_name}_{bbox_type}_{seed}.pickle"
    if not model_path.exists():
        ValueError(f"Black box model not found at {model_path}. Please ensure the black box is trained and saved before running this experiment.")

    print("Loading the Black Box")

    bbox = BlackBox(bb_type=bbox_type).load(model_path) # it is an object from out BlackBox class, which has a predict method and a fit method. It is not the sklearn object.



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
        print(f"status: {HybridDT_classifier.status}")
        HybridDT_classifier.tree_to_string()
        #to draw the tree
        #HybridDT_classifier.plot_tree(filename="my_hybrid_tree.png", show=False)

    result = {"model": model,"dataset":dataset_name, "seed":seed,
              "depth": h_params["depth"],
              "min_transp" : h_params["min_transp"],
              "n_branches":np.sum(list(HybridDT_classifier.b.values())),
              "n_leaves": np.sum(list(HybridDT_classifier.p.values())),
               "train":{ "predictions": pred['train'],
              "pred_type": pred_type['train'], "acc": acc['train'], "transparency": transp_ratio['train']},
              "test":{ "predictions": pred['test'],
              "pred_type": pred_type['test'], "acc": acc['test'], "transparency": transp_ratio['test']},
              "validation":{ "predictions": pred['validation'],
              "pred_type": pred_type['validation'], "acc": acc['validation'], "transparency": transp_ratio['validation']},
              "only_bb_acc": HybridDT_classifier.black_box_train_acc,
              "status": HybridDT_classifier.status,
              "optgap" : HybridDT_classifier.optgap,
              "callback_history": HybridDT_classifier.callback_history,
              }
    return result


LOC_PATH = Path.home()/'programming'/'optimization'/'Strong_Tree'
REM_PATH = "zibaja@nibi.alliancecan.ca:/home/zibaja/scratch"

def send():
    print("Sending project to Compute Canada...")
    
    cmd_str = " ".join([
        "rsync -av",
        "--exclude '.venv'",
        "--exclude '__pycache__/'",
        "--exclude '*.pyc'",
        "--exclude '.git'",
        "--exclude 'StrongTree/Results/'", 
        "--exclude 'StrongTree/Plots/'", 
        "--exclude 'StrongTree/Code/StrongTree/plotsziba/'", 
         "--exclude 'paper/bootstrap_results/'",  
        f"{LOC_PATH}",
        f"{REM_PATH}"
    ])
    
    subprocess.run(cmd_str, shell=True)


def receive():
    print("Receiving results from Compute Canada...")
    src_path = Path(REM_PATH, "Strong_Tree", "StrongTree", "Results","HybridDT","Benders", "HybridDTClassifier_post/*")
    dst_path = Path(LOC_PATH, "StrongTree", "Results", "HybridDT","Benders", "HybridDTClassifier_post")
    # src_path = Path(REM_PATH, "Strong_Tree", "StrongTree", "Code","StrongTree", "results/*")
    # dst_path = Path(LOC_PATH, "StrongTree", "Code", "StrongTree", "results")

    cmd_str = f"rsync -av {src_path} {dst_path}"
    
    subprocess.run(cmd_str, shell=True)


def main ():
    
    parser = argparse.ArgumentParser(description='Run HybridDT experimnets')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--local_id', type=int, default=None)
    parser.add_argument('--estimator_id', type=int, default=None)
    parser.add_argument('--send', action='store_true', help='Sync project to Compute Canada')
    parser.add_argument('--receive', action='store_true', help='Fetch results from Compute Canada')
    
    args = parser.parse_args()

    
    if args.send:
        send()
        return

    if args.receive:
        receive()
        return

    estimators = ['FlowOCT', 'BendersOCT']

    filtered_experiments = []

    for cfg in EXPERIMENTS[estimators[args.estimator_id]]:
        if args.dataset is not None and cfg["dataset_name"] != args.dataset:
            continue
        if args.model is not None and cfg["model"] != args.model:
            continue
        if args.seed is not None and cfg["seed"] != args.seed:
            continue
        filtered_experiments.append(cfg)
        # print(cfg, filtered_experiments.index(cfg))

    #print(f"Total filtered jobs: {len(filtered_experiments)}")
    print("number of experimnets ",len(filtered_experiments))

    cfg = filtered_experiments[args.local_id]
    estimator = estimators[args.estimator_id]
    print(f"Solving the problem using {estimator}")
    print(f"Running configuration: {cfg}")
    
    results = run_one_experiment(**cfg,estimator=estimator, time_limit=3600)


    
    # Save results 
    output_dir = Path.cwd().parent.parent/'Results'/'HybridDT'/f"{cfg['model']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if estimator=='FlowOCT':
        output_dir = Path.cwd().parent.parent/'Results'/'HybridDT'/f"{cfg['model']}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / (
        f"{cfg['dataset_name']}_"
        f"seed{cfg['seed']}_"
        f"depth{cfg['depth']}_"
        f"lambdaa{cfg['lambdaa']}_"
        f"min_transp{cfg['min_transp']}.pkl")
    else:
        output_dir = Path.cwd().parent.parent/'Results'/'HybridDT'/'Benders'/f"{cfg['model']}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / (
        f"{cfg['dataset_name']}_"
        f"seed{cfg['seed']}_"
        f"depth{cfg['depth']}_"
        f"lambdaa{cfg['lambdaa']}_"
        f"eta{cfg['eta']}.pkl")

    with open(output_file, "wb") as f:
        pickle.dump(results, f)



 

if __name__ == "__main__":
    main()


