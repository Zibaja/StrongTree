#!/bin/bash
#SBATCH --job-name=HybridDT
#SBATCH --array=0-2
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

# Load Python module 

module load python/3.10
module load gurobi/13.0.0


echo "Running on node: $(hostname)"
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"


# Create folders if not exist
mkdir -p logs
mkdir -p Results

# Run script
python -u Run_hybridDT.py \
    --dataset compas \
    --model  HybridDTClassifier_post\
    --local_id $SLURM_ARRAY_TASK_ID

echo "Job finished with exit code $?"