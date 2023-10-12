#!/bin/bash
#SBATCH -J v1_TimeSHAP_calculation_TILTomorrow
#SBATCH -A MENON-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=54080
#SBATCH --array=0-9999
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sb2406@cam.ac.uk
#SBATCH --output=/home/sb2406/rds/hpc-work/TILTomorrow_model_interpretations/v1-0/hpc_logs/timeSHAP/TILTomorrow_timeSHAP_v1-0_trial_%a.out

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 05c_calculate_TimeSHAP.py $SLURM_ARRAY_TASK_ID