#### Master Script 06a: Calculate testing set calibration and discrimination performance metrics for prediction of transitions in TIL(Basic) ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate testing set calibration and discrimination based on provided bootstrapping resample row index

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
from functions.analysis import thresh_trans, get_trans_probs, prepare_df, calc_trans_ORC, calc_trans_Somers_D, calc_trans_thresh_AUC

## Define parameters for model training
# Set version code
VERSION = 'v2-0'

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Window indices at which to calculate performance metrics
PERF_WINDOW_INDICES = [1,2,3,4,5,6,9,13]

## Define and create relevant directories
# Define directory in which CENTER-TBI data is stored
dir_CENTER_TBI = '/home/sb2406/rds/hpc-work/CENTER-TBI'

# Define and create subdirectory to store formatted TIL values
form_TIL_dir = os.path.join(dir_CENTER_TBI,'FormattedTIL')

# Define model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)

# Define model performance directory based on version code
model_perf_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_performance',VERSION)

# Define and create subdirectory to store testing set bootstrapping results
test_bs_dir = os.path.join(model_perf_dir,'testing_set_bootstrapping')

# Define and create subdirectory to store testing set bootstrapping results for post-hoc analysis
trans_pred_bs_dir = os.path.join(model_perf_dir,'trans_pred_bootstrapping')
os.makedirs(trans_pred_bs_dir,exist_ok=True)

## Load fundamental information for model training
# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

# Load parametric grid corresponding to sensitivity analysis
sens_analysis_grid = pd.read_csv(os.path.join(model_dir,'sens_analysis_grid.csv'))[['SENS_IDX','DROPOUT_VARS']].drop_duplicates(ignore_index=True)

# Load bootstrapping resample dataframe for testing set performance
bs_resamples = pd.read_pickle(os.path.join(model_perf_dir,'test_performance_bs_resamples.pkl'))

### II. Calculate testing set calibration and discrimination based on provided bootstrapping resample row index
# Argument-induced bootstrapping functions
def main(array_task_id):
    
    # Extract current bootstrapping resample parameters
    curr_rs_idx = bs_resamples.RESAMPLE_IDX[array_task_id]
    curr_GUPIs = bs_resamples.GUPIs[array_task_id]
    
    ## Prepare and compile testing set outputs from different variable-set models
    # Load full-model compiled testing set outputs
    calib_TILBasic_test_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_test_calibrated_outputs.pkl'))
    calib_TILBasic_test_outputs['DROPOUT_VARS'] = 'none'

    # Load dropped-variable compiled testing set outputs
    sens_calib_TILBasic_test_outputs = pd.read_pickle(os.path.join(model_dir,'sens_analysis_TomorrowTILBasic_compiled_test_calibrated_outputs.pkl')).merge(sens_analysis_grid,how='left').drop(columns='SENS_IDX')

    # Compile full-model and dropped-variable outputs
    compiled_test_outputs = pd.concat([calib_TILBasic_test_outputs,sens_calib_TILBasic_test_outputs],ignore_index=True)

    # Drop logit columns
    logit_cols = [col for col in compiled_test_outputs if col.startswith('z_TILBasic=')]
    compiled_test_outputs = compiled_test_outputs.drop(columns=logit_cols)

    # Calculate intermediate values for TomorrowTILBasic testing set outputs
    prob_cols = [col for col in compiled_test_outputs if col.startswith('Pr(TILBasic=')]
    prob_matrix = compiled_test_outputs[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    index_vector = np.array(list(range(prob_matrix.shape[1])), ndmin=2).T
    compiled_test_outputs['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    compiled_test_outputs['PredLabel'] = prob_matrix.idxmax(axis=1)

    # Load trivial, no information outputs
    no_info_TILBasic_outputs = pd.read_pickle(os.path.join(model_dir,'no_info_TomorrowTILBasic_compiled_outputs.pkl'))
    no_info_TILBasic_outputs['DROPOUT_VARS'] = 'last_TIL_only'

    # Compile full-model, dropped-variable, and last-TIL outputs
    compiled_test_outputs = pd.concat([compiled_test_outputs,no_info_TILBasic_outputs],ignore_index=True)

    # Filter testing set outputs to current GUPI set
    compiled_test_outputs = compiled_test_outputs[compiled_test_outputs.GUPI.isin(curr_GUPIs)].reset_index(drop=True)

    # Filter testing set outputs to optimal tuning configuration
    compiled_test_outputs = compiled_test_outputs[(compiled_test_outputs.TUNE_IDX==332)&(~compiled_test_outputs.DROPOUT_VARS.isin(['clinician_impressions','treatments']))].reset_index(drop=True)

    ## Determine points of model outputs corresponding to a transition in TILBasic
    # Load formatted TIL values
    formatted_TIL_values = pd.read_csv(os.path.join(form_TIL_dir,'formatted_TIL_values.csv'))[['GUPI','TILTimepoint','TILBasic']].rename(columns={'TILTimepoint':'WindowIdx'})

    # Merge available TIL values onto study windows
    merged_TIL_values = compiled_test_outputs[['GUPI','WindowIdx','TrueLabel']].drop_duplicates(ignore_index=True).merge(formatted_TIL_values,how='left')

    # Fill in missing current TILBasic values by using the last available TILBasic assessment
    merged_TIL_values['TILBasic'] = merged_TIL_values.groupby(['GUPI'],as_index=False).TILBasic.ffill()

    # Mark points of stasis, decrease, and increase
    merged_TIL_values['Stasis'] = (merged_TIL_values['TrueLabel'] == merged_TIL_values['TILBasic']).astype(int)
    merged_TIL_values['Decrease'] = (merged_TIL_values['TrueLabel'] < merged_TIL_values['TILBasic']).astype(int)
    merged_TIL_values['Increase'] = (merged_TIL_values['TrueLabel'] > merged_TIL_values['TILBasic']).astype(int)

    # Apply function to determine transitions across specific thresholds
    merged_TIL_values = thresh_trans(merged_TIL_values)

    # Add transition indicators to the test-set outputs
    compiled_test_outputs = compiled_test_outputs.merge(merged_TIL_values,how='left')

    # Calculate probabilities of decrease, stasis, and increase based on current TILBasic
    compiled_test_outputs = get_trans_probs(compiled_test_outputs)

    ## Prepare model outputs based on window indices
    # Prepare testing set output dataframes for performance calculation
    filt_test_outputs = prepare_df(compiled_test_outputs,PERF_WINDOW_INDICES)

    ## Iterate through post-hoc parametric grid and calculate testing set performance metrics
    # Create empty lists to store performance metrics
    test_scalar_metrics = []
    test_thresh_calibration_curves = []

    # Calculate ORCs of TIL-Basic model on testing set outputs
    curr_ORC = filt_test_outputs.groupby('DROPOUT_VARS',as_index=True).apply(lambda x: calc_trans_ORC(x,PERF_WINDOW_INDICES,True,'Calculating testing set ORC at points of transition')).reset_index().drop(columns=['level_1'])
    curr_ORC.insert(3,'THRESHOLD',['None' for idx in range(curr_ORC.shape[0])])

    # Calculate Somers' D of TIL-Basic model on testing set outputs
    curr_Somers_D = filt_test_outputs.groupby('DROPOUT_VARS',as_index=True).apply(lambda x: calc_trans_Somers_D(x,PERF_WINDOW_INDICES,True,'Calculating testing set Somers D at points of transition')).reset_index().drop(columns=['level_1'])
    curr_Somers_D.insert(3,'THRESHOLD',['None' for idx in range(curr_Somers_D.shape[0])])

    # Calculate threshold-level AUC of TIL-Basic model on testing set outputs
    curr_thresh_AUC = filt_test_outputs.groupby('DROPOUT_VARS',as_index=True).apply(lambda x: calc_trans_thresh_AUC(x,PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level AUC at points of transition')).reset_index().drop(columns=['level_1'])

    # Compile scalar metrics into single dataframe and label
    test_scalar_metrics = pd.concat([curr_ORC,curr_Somers_D,curr_thresh_AUC],ignore_index=True)
    test_scalar_metrics['RESAMPLE_IDX'] = curr_rs_idx

    ## Save performance metrics from current resample's testing outputs
    # Save scalar metrics of TIL-Basic model from testing set outputs
    test_scalar_metrics.to_pickle(os.path.join(trans_pred_bs_dir,'trans_pred_test_calibrated_metrics_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)