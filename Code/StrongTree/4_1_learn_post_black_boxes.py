import argparse
import numpy as np
from local_config import ccanada_expes
import os
#import warnings
#warnings.filterwarnings("ignore")
from Hybrid_utils import *
from black_box_models import BlackBox
from pathlib import Path

# MPI grid
bbox_types = ['random_forest'] # change to ['random_forest', 'ada_boost', 'gradient_boost'] for the final experiments 
rseeds = [1, 2, 3, 4, 5]
params = [] # will contain 1 * 5 = 5 values
for rs in rseeds:
    for bt in bbox_types:
        params.append([rs, bt]) 

if ccanada_expes:
    from mpi4py import MPI
    script_verbose = 0
    bbox_verbose = False
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
else:
    script_verbose = 1
    bbox_verbose = True
    rank = 0

param = params[rank]
rseed = param[0]
bbox_type = param[1]

parser = argparse.ArgumentParser()
# train data, last column is label
parser.add_argument("--dataset", type=int, help='Dataset ID', default=0)
#parser.add_argument("--bbox_type", type=str, help='Black box. Options: random_forest, ada_boost, gradient_boost', default='random_forest')
#parser.add_argument("--rseed", type=int, help='Random Data Split', default=0)
#parser.add_argument("--time_limit", type=int, help='Maximum run-time in seconds', default=3600)

args = parser.parse_args()

#time_limit = args.time_limit
time_limit = int(34.0 * 3600)
n_iters = 100

datasets = ["compas", "adult", "acs_employ"]
dataset_name = datasets[args.dataset]

# Get the data
if script_verbose > 0:
    print("Loading the data...")
splits = {"train": 2000, "test":2000, "valid":2000}
DataDIR = Path.cwd().parent.parent/'DataSets'
Datapath = DataDIR/f'{dataset_name}.csv'
my_data = Dataset.from_csv(Datapath, dataset_name)

X, y, features, prediction = my_data.split_data_as_dict_withsize(splits,
                            random_state_param = rseed)
df_X = my_data.to_df_from_dict(X)


if script_verbose > 0:
    print("Data loaded!")

#### Black Box ####

model_path = Path("models") / f"{dataset_name}_{bbox_type}_{rseed}.pickle"
model_path.parent.mkdir(parents=True, exist_ok=True)

if not model_path.exists():
    if script_verbose > 0:
        print("Fitting the Black Box\n")
    bbox = BlackBox(bb_type=bbox_type, verbosity=bbox_verbose, random_state_value=rseed, n_iter=n_iters, time_limit=time_limit, X_val=df_X["valid"], y_val=y["valid"])
    bbox.fit(df_X["train"], y["train"])
    bbox.save(str(model_path))
    train = 'Yes'
else:
    if script_verbose > 0:
        print("Loading the Black Box\n")
    bbox = BlackBox(bb_type=bbox_type).load(str(model_path))
    train = 'No'

# Black box performances
bbox_acc_train = np.mean(bbox.predict(df_X["train"]) == y["train"])
bbox_acc_v = np.mean(bbox.predict(df_X["valid"]) == y["valid"])
bbox_acc_t = np.mean(bbox.predict(df_X["test"]) == y["test"])
trials_details = bbox.trials_details
n_evals = bbox.n_evals
res = [[rseed, bbox_type, train, time_limit, n_iters, n_evals, bbox_acc_train, bbox_acc_v, bbox_acc_t, bbox.black_box_model, model_path, trials_details]]

# Gather the results for the 5 folds on process 0
if ccanada_expes:
    res = comm.gather(res, root=0)

if rank == 0 or not ccanada_expes:
    # save results
    results_path = Path("results") / f"results_4_1_learn_post_black_boxes_{dataset_name}.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # fileName = './results/results_4_1_learn_post_black_boxes_%s.csv' %(dataset_name) #_proportions
    import csv
    with open(results_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Seed', 'Black Box', 'Train?', 'Time Limit', 'max#iterations', 'actual#iterations', 'Training accuracy', 'Validation accuracy', 'Test accuracy', 'Model', 'Path', 'Trials Details'])
        for i in range(len(res)):
            if ccanada_expes:
                for j in range(len(res[i])):
                    csv_writer.writerow(res[i][j])
            else:
                csv_writer.writerow(res[i])

