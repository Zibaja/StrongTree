#!/bin/bash

#SBATCH -n 5
#SBATCH --mem-per-cpu=9000
#SBATCH --time=34:00:00
#SBATCH --job-name=expes_post
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-2

module load python/3.10
module load mpi4py


srun -n 5 python 4_1_learn_post_black_boxes.py --dataset=${SLURM_ARRAY_TASK_ID}



