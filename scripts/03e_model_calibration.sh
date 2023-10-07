#!/bin/bash
#SBATCH -J model_calibration
#SBATCH -A MENON-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=27040
#SBATCH --array=0-479
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sb2406@cam.ac.uk
#SBATCH --output=/home/sb2406/rds/hpc-work/TILTomorrow_model_performance/v2-0/hpc_logs/calibration/TILTomorrow_val_calibration_v2-0_trial_%a.out

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 03e_model_calibration.py $SLURM_ARRAY_TASK_ID