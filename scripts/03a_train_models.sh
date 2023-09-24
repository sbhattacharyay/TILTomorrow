#!/bin/bash
#SBATCH --job-name=v1_uncalibrated_TILTomorrow_training
#SBATCH --time=01:30:00
#SBATCH --array=0-1439
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=MENON-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --mail-type=ALL
#SBATCH --output=/home/sb2406/rds/hpc-work/TILTomorrow_model_outputs/v1-0/hpc_logs/training/TILTomorrow_training_v1-0_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 03a_train_models.py $SLURM_ARRAY_TASK_ID