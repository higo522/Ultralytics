#!/bin/bash
#SBATCH --account=def-ncoops
#SBATCH --job-name=cvsweep
#SBATCH --mail-user=higo522@student.ubc.ca
#SBATCH --mail-type=END,FAIL          # one summary email for the whole array (add ARRAY_TASKS for per-fold emails)
#SBATCH --array=0-19                  # one task per fold (20 folds)
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12            # Narval A100 nodes: 12 cores per GPU
#SBATCH --mem=64G
#SBATCH --time=12:00:00               # adjust after timing one fold
#SBATCH --output=%x_%A_%a.out

module load StdEnv/2023 python/3.11 opencv

source ~/scratch/Ultralytics/venv/bin/activate

# Narval compute nodes have NO internet access:
# wandb must run offline (sync from the login node afterwards)
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

mkdir -p "$WANDB_DIR"

cd ~/scratch/Ultralytics
python ssh/scripts/CVsweep_narval.py "$SLURM_ARRAY_TASK_ID"
