#!/bin/bash
#SBATCH --job-name=v2_sensitivity_TILTomorrow_training
#SBATCH --time=01:30:00
#SBATCH --array=0-19
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=MENON-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --mail-type=ALL
#SBATCH --output=/home/sb2406/rds/hpc-work/TILTomorrow_model_outputs/v2-0/hpc_logs/training/sens_TILTomorrow_training_v2-0_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 04a_train_sensitivity_analysis_models.py $SLURM_ARRAY_TASK_ID