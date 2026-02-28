#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-23:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --account=meisam-lab
#SBATCH --job-name=small-imagenet-convnext-private
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ACTIVATE VENV
source /lustre/hdd/LAS/meisam-lab/hosen/symlinks/imagenet/bin/activate

# DIAGNOSTICS (optional but useful)
echo "Running on node(s): $SLURM_NODELIST"
echo "Number of GPUs allocated: $SLURM_GPUS_ON_NODE"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"

# RUN PROGRAM
EPSILON=2.0
LEARNING_RATE=0.0005
LOG_FILE="logs/convnext_tiny_run_private_eps${EPSILON}_lr${LEARNING_RATE}.log"
python convnext_tiny.py \
  --use-differential-privacy \
  --target-epsilon "${EPSILON}" \
  --learning-rate "${LEARNING_RATE}" \
  > "${LOG_FILE}" 2>&1

echo "Job completed successfully"
