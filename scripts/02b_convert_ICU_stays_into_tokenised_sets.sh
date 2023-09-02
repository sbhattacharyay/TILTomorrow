#!/bin/bash
#SBATCH -J ICU_stay_tokenisation
#SBATCH -A MENON-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=54080
#SBATCH --array=0-99
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sb2406@cam.ac.uk
#SBATCH --output=/home/sb2406/rds/hpc-work/tokens/hpc_logs/numeric_variable_tokenisation_trial_%a.out

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 02b_convert_ICU_stays_into_tokenised_sets.py $SLURM_ARRAY_TASK_ID