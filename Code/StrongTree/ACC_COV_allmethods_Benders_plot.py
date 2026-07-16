from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import pandas as pd




def interpolate_metric(x, y, kind='next',agg='mean', fill_value='extrapolate'): #''next
    """
    Interpolate y over x (automatically sorts x).
    
    Parameters:
        x: array-like, coverage values
        y: array-like, metric values
        kind: interpolation type ('linear', 'cubic', etc.)
        fill_value: how to handle extrapolation
    
    Returns:
        f: interp1d function
    """
    x = np.array(x)
    y = np.array(y)
    
    # sort by x
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]

   

    # Group by coverage as there might be several values for each coverage (more than one accuarcy for one coverage)
    bucket = defaultdict(list)
    for c, v in zip(x_sorted, y_sorted):
        bucket[c].append(v)

    unique_x = np.array(sorted(bucket.keys()))

    if agg == 'mean':
        agg_values = np.array([np.mean(bucket[c]) for c in unique_x])
    elif agg == 'max':
        agg_values = np.array([np.max(bucket[c]) for c in unique_x])
    else:
        raise ValueError("agg must be 'mean' or 'max'")


    if len(unique_x) < 2:
            return lambda x: np.full_like(x, agg_values[0], dtype=float)

    return interp1d(
        unique_x,
        agg_values,
        kind=kind,
        bounds_error=False,
        fill_value=fill_value
    )


import pickle
import re
from pathlib import Path

import pickle
import re
from pathlib import Path
import math


HYBRID_CORELS_PATTERN = re.compile(
    r"^(?P<dataset>.+?)_"
    r"HybridCORELSPostClassifier_"
    r"seed(?P<seed>\d+)_"
    r"depth(?P<depth>\d+)_"
    r"min_cov(?P<min_transp>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)_"
    r"lambdaValue(?P<lambdaa>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$"
)


HYBRID_DT_PATTERN = re.compile(
    r"^(?P<dataset>.+?)_"
    r"seed(?P<seed>\d+)_"
    r"depth(?P<depth>\d+)_"
    r"lambdaa(?P<lambdaa>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)_"
    r"min_transp(?P<min_transp>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$"
)


HYBRID_DT_BENDERS_PATTERN = re.compile(
    r"^(?P<dataset>.+?)_"
    r"seed(?P<seed>\d+)_"
    r"depth(?P<depth>\d+)_"
    r"lambdaa(?P<lambdaa>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)_"
    r"eta(?P<eta>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$"
)


def read_results(
    result_path,
    dataset_name,
    seed=None,
    depth=None,
    lambdaa_val=None,
    min_transp_val=None,
    eta_val=None,
    method="HybridDTClassifier_post",
):
    """
    Read pickle result files and filter them using exact filename parameters.

    Supported filename patterns
    ---------------------------
    HybridCORELSPostClassifier:
        compas_HybridCORELSPostClassifier_seed1_depth15_
        min_cov0.9_lambdaValue0.0001

    HybridDTClassifier_post:
        compas_seed1_depth1_lambdaa0.01_min_transp0.9

    HybridDTBenders:
        adult_seed1_depth1_lambdaa0.001_eta0.01

    Parameters
    ----------
    result_path : str or Path
        Directory containing result pickle files.

    dataset_name : str
        Dataset name, for example "adult" or "compas".

    seed : int, optional
        Exact seed value.

    depth : int, optional
        Exact tree depth.

    lambdaa_val : float, optional
        Exact lambda value.

    min_transp_val : float, optional
        Minimum transparency for HybridDTClassifier_post, or minimum
        coverage for HybridCORELSPostClassifier.

    eta_val : float, optional
        Eta value for HybridDTBenders.

    method : str
        One of:
        - "HybridCORELSPostClassifier"
        - "HybridDTClassifier_post"
        - "HybridDTBenders"

    Returns
    -------
    dict
        Dictionary mapping each matching filename stem to its loaded result.
    """

    result_path = Path(result_path)
    all_models = {}

    valid_methods = {
        "HybridCORELSPostClassifier",
        "HybridDTClassifier_post",
        "HybridDTBenders",
    }

    if method not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. Expected one of: "
            f"{sorted(valid_methods)}"
        )

    for file_path in result_path.iterdir():

        if not file_path.is_file():
            continue

        # Optional: only try to read pickle files.
        if file_path.suffix not in {".pkl", ".pickle", ""}:
            continue

        filename = file_path.stem

        if method == "HybridCORELSPostClassifier":
            match = HYBRID_CORELS_PATTERN.fullmatch(filename)

        elif method == "HybridDTClassifier_post":
            match = HYBRID_DT_PATTERN.fullmatch(filename)

        else:  # HybridDTBenders
            match = HYBRID_DT_BENDERS_PATTERN.fullmatch(filename)

        # Skip files that do not follow the method's naming convention.
        if match is None:
            continue

        parameters = match.groupdict()

        file_dataset = parameters["dataset"]
        file_seed = int(parameters["seed"])
        file_depth = int(parameters["depth"])
        file_lambdaa = float(parameters["lambdaa"])

        # Exact dataset and integer comparisons.
        if file_dataset != dataset_name:
            continue

        if seed is not None and file_seed != int(seed):
            continue

        if depth is not None and file_depth != int(depth):
            continue

        # Use isclose for safe floating-point comparisons.
        if (
            lambdaa_val is not None
            and not math.isclose(
                file_lambdaa,
                float(lambdaa_val),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            continue

        if method in {
            "HybridCORELSPostClassifier",
            "HybridDTClassifier_post",
        }:
            file_min_transp = float(parameters["min_transp"])

            if (
                min_transp_val is not None
                and not math.isclose(
                    file_min_transp,
                    float(min_transp_val),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ):
                continue

        elif method == "HybridDTBenders":
            file_eta = float(parameters["eta"])

            if (
                eta_val is not None
                and not math.isclose(
                    file_eta,
                    float(eta_val),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ):
                continue

        try:
            with open(file_path, "rb") as result_file:
                all_models[filename] = pickle.load(result_file)

        except (pickle.UnpicklingError, EOFError, OSError) as error:
            print(f"Could not read {file_path.name}: {error}")

    return all_models


def plot_interpolated_acc_vs_transparency(interpolated_by_seed, split, dataset_name, depth, method):
    """it plots the aggreagted values from interpolated function for each dataset , seed , depth and lambda value
        all lambda values apears in one plot
 
    """
    print(f"plotting for depth {depth}")
    # fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    desired_cov = np.linspace(0.1, 0.95, 50)
  
    
    if method == 'HybridDTClassifier_post':
        depth_method = depth
    else:
        depth_method = (2**depth)-1

    for lambdaa_val in [10**-2, 10**-3, 10**-4]:

        acc = np.array([interpolated_by_seed[dataset_name][s][split][depth_method][lambdaa_val]['acc'](desired_cov) for s in [1,2,3,4,5]])
        mean_acc = np.mean(acc, axis=0)
        std_acc = np.std(acc, axis=0)

        plt.plot(desired_cov, mean_acc, label=f"Lambda: {lambdaa_val}")
        plt.fill_between(desired_cov, mean_acc - std_acc, mean_acc + std_acc, alpha=0.25)


        plt.xlabel("Transparency")
        plt.ylabel("Accuracy")
        plt.title(f'{split.capitalize()}-Depth: {depth}')
        
        plt.legend()
        plt.grid(True)


    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plotsziba'/'ACC-paretofront'/method
    output_dir.mkdir(parents=True,exist_ok=True)
    
    output_file = output_dir / f"ACC_Cov_{method}_{dataset_name}_depth{depth}_{split}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


def plot_interpolated_acc_vs_transparency_allmethods(interpolated_by_seed, split, dataset_name, depth, pareto=True):
    """it plots the aggreagted values from interpolated function for each dataset , seed , depth 
 
    """
    
    # fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    desired_cov = np.linspace(0.1, 0.95, 50)
  
  

    for method in methods:
        if method == 'HybridCORELSPostClassifier':
            depth_method = (2**depth)-1
        else:
            depth_method = depth

        acc = np.array([interpolated_by_seed[method][dataset_name][s][split][depth_method]['acc'](desired_cov) for s in [1,2,3,4,5]])
        mean_acc = np.mean(acc, axis=0)
        std_acc = np.std(acc, axis=0)

        plt.plot(desired_cov, mean_acc, label=f"{method}")
        plt.fill_between(desired_cov, mean_acc - std_acc, mean_acc + std_acc, alpha=0.25)


        plt.xlabel("Transparency")
        plt.ylabel("Accuracy")
        plt.title(f'{split.capitalize()}-Depth: {depth}-{dataset_name}')
        
        plt.legend()
        plt.grid(True)


    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plotsziba'/'ACC-paretofront'
    output_dir.mkdir(parents=True,exist_ok=True)
    
    output_file = output_dir / f"ACC_Cov_bothmethods_{dataset_name}_depth{depth}_{split}.pdf"
    #plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()



def pareto_frontier_max_acc_max_transparency(points):
    """
    points: list of tuples (key, transparency, accuracy)
    Keeps validation Pareto frontier where both transparency and accuracy are maximized.
    """
    points = sorted(points, key=lambda x: (x[1], x[2]),reverse=True)  # sort by transparency

    frontier = []
    best_acc_so_far = -np.inf

    for key, transp, acc in points:
        if acc > best_acc_so_far:
            frontier.append((key, transp, acc))
            best_acc_so_far = acc

    return frontier




methods = ['HybridDTBenders','HybridDTClassifier_post','HybridCORELSPostClassifier'  ]

#read the results based on parameters (dataset, seed, depth, lambdaa, min_transp)

result_path = {"HybridDTClassifier_post": Path(Path("/Users/ziba/programming/optimization/Strong_Tree"), "StrongTree", "Results", "HybridDT", "HybridDTClassifier_post",),
               "HybridCORELSPostClassifier": Path("/Users/ziba/programming/optimization/HybridCorels-julien/HybridCORELS/paper/HybridCORELS_results/new"),
               "HybridDTBenders" : Path(Path("/Users/ziba/programming/optimization/Strong_Tree"), "StrongTree", "Results", "HybridDT","Benders",  "HybridDTClassifier_post",)
               }

#interpolate the accuracy based on the transparency for each seed and each split
# Store interpolated functions per seed and split
DATASETS = ["compas", "adult"]
SEEDS = [1,2,3,4,5]
splits = ["train", "test", "validation"]
depths_val = [1,2,3,4,5]
depths = {'HybridDTClassifier_post': depths_val, 
          'HybridCORELSPostClassifier': [(2**i)-1 for i in depths_val],
          'HybridDTBenders': depths_val}
lambda_values = [10**-2, 10**-3, 10**-4]
pareto = False
#method = 'HybridCORELSPostClassifier'
interpolated_by_seed = {method : {dataset_name:{
    seed: {
        split: {
            depth: {}
            for depth in depths[method]
        }
        for split in splits
    }
    for seed in SEEDS
} for dataset_name in DATASETS} for method in methods}


def main():
    
    for method in methods:
        print(f"Generating the plots for {method}")
        for dataset_name in DATASETS:
            for seed in SEEDS:
                for depth in depths[method]:
                    all_models = read_results(result_path[method],dataset_name,seed=seed,depth=depth,lambdaa_val=None,min_transp_val=None,eta_val=None,method=method)
                    validation_points = []

                    for key, model in all_models.items():
                        validation_points.append(
                            (
                                key,
                                model[0]["validation"]["transparency"] if method=='HybridCORELSPostClassifier' else model["validation"]["transparency"],
                                model[0]["validation"]["acc"] if method=='HybridCORELSPostClassifier' else model["validation"]["acc"]
                            )
                        )

                    pareto_points = pareto_frontier_max_acc_max_transparency(validation_points)
                    pareto_keys = [key for key, _, _ in pareto_points]
                    for split in ['train', 'test', 'validation']:    
                        transp = []
                        acc = []
                        #uncomment the following lines if you want to use all models instead of only the Pareto frontier models
                        if not pareto:
                            for model in all_models.values():
                                transp.append(model[0][split]["transparency"] if method=='HybridCORELSPostClassifier' else model[split]["transparency"] )
                                acc.append(model[0][split]["acc"] if method=='HybridCORELSPostClassifier' else model[split]["acc"] )
                        #comment the following lines if you want to use all models instead of only the Pareto frontier models
                        if pareto:
                            for key in pareto_keys:
                                model = all_models[key]
                                transp.append(model[0][split]["transparency"] if method=='HybridCORELSPostClassifier' else model[split]["transparency"] )
                                acc.append(model[0][split]["acc"] if method=='HybridCORELSPostClassifier' else model[split]["acc"] )
                        
                        interpolated_acc = interpolate_metric(transp,acc)

                        interpolated_by_seed[method][dataset_name][seed][split][depth]["acc"] = interpolated_acc


    for dataset_name in [ "compas"]:
        for depth in [1,2,3,4,5]:
            for split in splits:
    #             #plot the average accuracy and std for each split based on the transparency
                # plot_interpolated_acc_vs_transparency(interpolated_by_seed, split=split,depth=depth,dataset_name=dataset_name, method= method)
                plot_interpolated_acc_vs_transparency_allmethods(interpolated_by_seed, split=split,depth=depth,dataset_name=dataset_name, pareto=pareto)


if __name__=='__main__':
    main()