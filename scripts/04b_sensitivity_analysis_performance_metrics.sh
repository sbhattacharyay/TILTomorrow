#!/bin/bash
#SBATCH -J v2_sensitivity_performance
#SBATCH -A MENON-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=54080
#SBATCH --array=0-999
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sb2406@cam.ac.uk
#SBATCH --output=/home/sb2406/rds/hpc-work/TILTomorrow_model_performance/v2-0/hpc_logs/sensitivity_performance/TILTomorrow_sensitivity_bootstrapping_v2-0_trial_%a.out

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 04b_sensitivity_analysis_performance_metrics.py $SLURM_ARRAY_TASK_ID