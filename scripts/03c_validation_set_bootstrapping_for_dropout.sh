#!/bin/bash
#SBATCH -J v2_validation_set_bootstrapping
#SBATCH -A MENON-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=54080
#SBATCH --array=0-999
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sb2406@cam.ac.uk
#SBATCH --output=/home/sb2406/rds/hpc-work/TILTomorrow_model_performance/v2-0/hpc_logs/validation_bootstrapping/TILTomorrow_val_bootstrapping_v1-0_trial_%a.out

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 03c_validation_set_bootstrapping_for_dropout.py $SLURM_ARRAY_TASK_ID