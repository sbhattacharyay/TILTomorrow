#### Master Script 3c: Calculate validation set calibration and discrimination for dropout ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate validation set calibration and discrimination based on provided bootstrapping resample row index

### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import random
import datetime
import warnings
import itertools
import numpy as np
import pandas as pd
import pickle as cp
from tqdm import tqdm
import seaborn as sns
import multiprocessing
from scipy import stats
from pathlib import Path
from shutil import rmtree
from ast import literal_eval
import matplotlib.pyplot as plt
from scipy.special import logit
from collections import Counter
from argparse import ArgumentParser
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

# StatsModel methods
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from statsmodels.nonparametric.smoothers_lowess import lowess

# Custom methods
from functions.model_building import load_model_outputs
from functions.analysis import prepare_df, calc_ORC, calc_AUC, calc_Somers_D, calc_thresh_calibration, calc_binary_calibration

## Define parameters for model training
# Set version code
VERSION = 'v2-0'

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Window indices at which to calculate performance metrics
PERF_WINDOW_INDICES = [1,2,3,4,5,6,9,13]

## Define and create relevant directories
# Define model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)

# Define model performance directory based on version code
model_perf_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_performance',VERSION)

# Define and create subdirectory to store validation set bootstrapping results
val_bs_dir = os.path.join(model_perf_dir,'validation_set_bootstrapping')
os.makedirs(val_bs_dir,exist_ok=True)

## Load fundamental information for model training
# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

# Load bootstrapping resample dataframe for validation set dropout
bs_resamples = pd.read_pickle(os.path.join(model_perf_dir,'val_dropout_bs_resamples.pkl'))

# Load validation set ORCs
uncalib_TILBasic_val_set_ORCs = pd.read_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_val_uncalibrated_ORCs.csv'))

# Load validation set threshold-level calibration metrics
uncalib_TILBasic_val_set_calibration = pd.read_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_val_uncalibrated_calibration_metrics.csv'))

# Load optimal tuning configurations for each window index based on validation set performance
opt_TILBasic_val_calibration_configs = pd.read_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_optimal_val_set_calibration_configurations.csv'))

### II. Calculate validation set calibration and discrimination based on provided bootstrapping resample row index
# Argument-induced bootstrapping functions
def main(array_task_id):
    
    # Extract current bootstrapping resample parameters
    curr_rs_idx = bs_resamples.RESAMPLE_IDX[array_task_id]
    curr_GUPIs = bs_resamples.GUPIs[array_task_id]
    
    # Load compiled validation set outputs
    uncalib_TILBasic_val_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_val_uncalibrated_outputs.pkl'))

    # Filter validation set outputs to current GUPI set
    uncalib_TILBasic_val_outputs = uncalib_TILBasic_val_outputs[uncalib_TILBasic_val_outputs.GUPI.isin(curr_GUPIs)].reset_index(drop=True)
    
    # Calculate intermediate values for TomorrowTILBasic validation set outputs
    prob_cols = [col for col in uncalib_TILBasic_val_outputs if col.startswith('Pr(TILBasic=')]
    logit_cols = [col for col in uncalib_TILBasic_val_outputs if col.startswith('z_TILBasic=')]
    prob_matrix = uncalib_TILBasic_val_outputs[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    index_vector = np.array(list(range(prob_matrix.shape[1])), ndmin=2).T
    uncalib_TILBasic_val_outputs['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    uncalib_TILBasic_val_outputs['PredLabel'] = prob_matrix.idxmax(axis=1)

    # Prepare validation set output dataframes for performance calculation
    filt_TILBasic_val_outputs = prepare_df(uncalib_TILBasic_val_outputs,PERF_WINDOW_INDICES)
    
    ## Calculate performance metrics on current validation set outputs
    # Calculate ORCs of TIL-Basic model on validation set outputs
    uncalib_TILBasic_val_set_ORCs = calc_ORC(filt_TILBasic_val_outputs,PERF_WINDOW_INDICES,True,'Calculating validation set ORC')

    # Calculate Somers' D of TIL-Basic model on validation set outputs
    uncalib_TILBasic_val_set_Somers_D = calc_Somers_D(filt_TILBasic_val_outputs,PERF_WINDOW_INDICES,True,'Calculating validation set Somers D')

    # Calculate threshold-level calibration metrics of TIL-Basic model on validation set outputs
    uncalib_TILBasic_val_set_thresh_calibration = calc_thresh_calibration(filt_TILBasic_val_outputs,PERF_WINDOW_INDICES,True,'Calculating validation set threshold-level calibration metrics')

    # Add macro-averages to threshold-level calibration metrics
    macro_average_thresh_calibration = uncalib_TILBasic_val_set_thresh_calibration.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
    macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(macro_average_thresh_calibration.shape[0])])
    uncalib_TILBasic_val_set_thresh_calibration = pd.concat([uncalib_TILBasic_val_set_thresh_calibration,macro_average_thresh_calibration],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)

    ## Save performance metrics from current resample's validation outputs
    # Save ORCs of TIL-Basic model from validation set outputs
    uncalib_TILBasic_val_set_ORCs.to_pickle(os.path.join(val_bs_dir,'TomorrowTILBasic_val_uncalibrated_ORCs_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    # Save Somers' D of TIL-Basic model from validation set outputs
    uncalib_TILBasic_val_set_Somers_D.to_pickle(os.path.join(val_bs_dir,'TomorrowTILBasic_val_uncalibrated_Somers_D_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    # Save threshold-level calibration metrics of TIL-Basic model from validation set outputs
    uncalib_TILBasic_val_set_thresh_calibration.to_pickle(os.path.join(val_bs_dir,'TomorrowTILBasic_val_uncalibrated_calibration_metrics_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)