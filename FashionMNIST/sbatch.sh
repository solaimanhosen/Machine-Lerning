#!/bin/bash
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=8       # Number of cores per node
#SBATCH --time=0-2:00:00          # Walltime (DD-HH:MM:SS)
#SBATCH --mem=80G                 # Memory per node
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --account=meisam-lab      # Slurm account
#SBATCH --job-name=test           # Job name
#SBATCH --mail-user=hosen@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=test-out-%j.txt  # Stdout (%j = job ID)
#SBATCH --error=test-error-%j.txt # Stderr (%j = job ID)

# LOAD MODULES
module purge
module load python/3.9

# ACTIVATE VENV (already created once)
source /work/LAS/meisam-lab/hosen/envs/ML/bin/activate

# RUN PROGRAM
python main.py

echo "Job completed successfully"
