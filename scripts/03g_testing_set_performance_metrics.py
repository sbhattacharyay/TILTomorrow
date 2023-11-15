#### Master Script 3g: Calculate calibrated testing set calibration and discrimination for statistical inference ####
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
from functions.analysis import prepare_df, calc_ORC, calc_Somers_D, calc_thresh_AUC, calc_thresh_calibration, calc_test_thresh_calib_curves

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
os.makedirs(test_bs_dir,exist_ok=True)

## Load fundamental information for model training
# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

# Load bootstrapping resample dataframe for testing set performance
bs_resamples = pd.read_pickle(os.path.join(model_perf_dir,'test_performance_bs_resamples.pkl'))

### II. Calculate testing set calibration and discrimination based on provided bootstrapping resample row index
# Argument-induced bootstrapping functions
def main(array_task_id):
    
    # Extract current bootstrapping resample parameters
    curr_rs_idx = bs_resamples.RESAMPLE_IDX[array_task_id]
    curr_GUPIs = bs_resamples.GUPIs[array_task_id]
    
    # Load compiled testing set outputs
    calib_TILBasic_test_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_test_calibrated_outputs.pkl'))

    # Filter testing set outputs to current GUPI set
    calib_TILBasic_test_outputs = calib_TILBasic_test_outputs[calib_TILBasic_test_outputs.GUPI.isin(curr_GUPIs)].reset_index(drop=True)
    
    # Calculate intermediate values for TomorrowTILBasic testing set outputs
    prob_cols = [col for col in calib_TILBasic_test_outputs if col.startswith('Pr(TILBasic=')]
    logit_cols = [col for col in calib_TILBasic_test_outputs if col.startswith('z_TILBasic=')]
    prob_matrix = calib_TILBasic_test_outputs[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    index_vector = np.array(list(range(prob_matrix.shape[1])), ndmin=2).T
    calib_TILBasic_test_outputs['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    calib_TILBasic_test_outputs['PredLabel'] = prob_matrix.idxmax(axis=1)

    ## Determine points of model outputs corresponding to a transition in TILBasic
    # Load formatted TIL values
    formatted_TIL_values = pd.read_csv(os.path.join(form_TIL_dir,'formatted_TIL_values.csv'))[['GUPI','TILTimepoint','TILBasic']].rename(columns={'TILTimepoint':'WindowIdx'})

    # Load study window timestamps and outcomes
    study_window_timestamps_outcomes = pd.read_csv(os.path.join(form_TIL_dir,'study_window_timestamps_outcomes.csv'))[['GUPI','WindowIdx','TomorrowTILBasic']].rename(columns={'TomorrowTILBasic':'TrueLabel'})
    
    # Merge available TIL values onto study windows
    no_info_merged_outputs = study_window_timestamps_outcomes.merge(formatted_TIL_values,how='left')

    # Fill in missing current TILBasic values by using the last available TILBasic assessment
    no_info_merged_outputs['TILBasic'] = no_info_merged_outputs.groupby(['GUPI'],as_index=False).TILBasic.ffill()

    # Filter points of discordance between last available TILBasic and next TILBasic
    transition_points = no_info_merged_outputs[no_info_merged_outputs.TrueLabel!=no_info_merged_outputs.TILBasic][['GUPI','WindowIdx','TrueLabel']].reset_index(drop=True)

    # Create a subset of the testing set output dataframe at the transition points
    trans_calib_TILBasic_test_outputs = calib_TILBasic_test_outputs.merge(transition_points,how='inner')

    # ## Prepare model outputs based on window indices
    # # Prepare testing set output dataframes for performance calculation
    # filt_TILBasic_test_outputs = prepare_df(calib_TILBasic_test_outputs,PERF_WINDOW_INDICES)

    # Prepare testing set output dataframes for performance calculation at transition points
    trans_filt_TILBasic_test_outputs = prepare_df(trans_calib_TILBasic_test_outputs,PERF_WINDOW_INDICES)

    ## Calculate performance metrics on current testing set outputs
    # # Calculate ORCs of TIL-Basic model on testing set outputs
    # calib_TILBasic_test_set_ORCs = calc_ORC(filt_TILBasic_test_outputs,PERF_WINDOW_INDICES,True,'Calculating testing set ORC')

    # # Calculate Somers' D of TIL-Basic model on testing set outputs
    # calib_TILBasic_test_set_Somers_D = calc_Somers_D(filt_TILBasic_test_outputs,PERF_WINDOW_INDICES,True,'Calculating testing set Somers D')

    # # Calculate threshold-level AUC of TIL-Basic model on testing set outputs
    # calib_TILBasic_test_set_thresh_AUCs = calc_thresh_AUC(filt_TILBasic_test_outputs,PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level AUC')
    
    # # Calculate threshold-level calibration metrics of TIL-Basic model on testing set outputs
    # calib_TILBasic_test_set_thresh_calibration = calc_thresh_calibration(filt_TILBasic_test_outputs,PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level calibration metrics')
    
    # # Calculate threshold-level calibration curves of TIL-Basic model on testing set outputs
    # calib_TILBasic_test_set_thresh_calibration_curves = calc_test_thresh_calib_curves(filt_TILBasic_test_outputs,PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level calibration curves')

    # # Add macro-averages to threshold-level AUCs
    # macro_average_thresh_AUCs = calib_TILBasic_test_set_thresh_AUCs.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
    # macro_average_thresh_AUCs.insert(2,'THRESHOLD',['Average' for idx in range(macro_average_thresh_AUCs.shape[0])])
    # calib_TILBasic_test_set_thresh_AUCs = pd.concat([calib_TILBasic_test_set_thresh_AUCs,macro_average_thresh_AUCs],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)
    
    # # Add macro-averages to threshold-level calibration metrics
    # macro_average_thresh_calibration = calib_TILBasic_test_set_thresh_calibration.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
    # macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(macro_average_thresh_calibration.shape[0])])
    # calib_TILBasic_test_set_thresh_calibration = pd.concat([calib_TILBasic_test_set_thresh_calibration,macro_average_thresh_calibration],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)

    ## Calculate performance metrics on current testing set outputs at points of transition
    # Calculate ORCs of TIL-Basic model on testing set outputs
    trans_calib_TILBasic_test_set_ORCs = calc_ORC(trans_filt_TILBasic_test_outputs,PERF_WINDOW_INDICES,True,'Calculating testing set ORC at points of transition')

    # Calculate Somers' D of TIL-Basic model on testing set outputs
    trans_calib_TILBasic_test_set_Somers_D = calc_Somers_D(trans_filt_TILBasic_test_outputs,PERF_WINDOW_INDICES,True,'Calculating testing set Somers D at points of transition')

    # Calculate threshold-level AUC of TIL-Basic model on testing set outputs
    trans_calib_TILBasic_test_set_thresh_AUCs = calc_thresh_AUC(trans_filt_TILBasic_test_outputs,PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level AUC at points of transition')
    
    # Calculate threshold-level calibration metrics of TIL-Basic model on testing set outputs
    trans_calib_TILBasic_test_set_thresh_calibration = calc_thresh_calibration(trans_filt_TILBasic_test_outputs,PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level calibration metrics at points of transition')
    
    # Calculate threshold-level calibration curves of TIL-Basic model on testing set outputs
    trans_calib_TILBasic_test_set_thresh_calibration_curves = calc_test_thresh_calib_curves(trans_filt_TILBasic_test_outputs,PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level calibration curves at points of transition')

    # Add macro-averages to threshold-level AUCs
    trans_macro_average_thresh_AUCs = trans_calib_TILBasic_test_set_thresh_AUCs.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
    trans_macro_average_thresh_AUCs.insert(2,'THRESHOLD',['Average' for idx in range(trans_macro_average_thresh_AUCs.shape[0])])
    trans_calib_TILBasic_test_set_thresh_AUCs = pd.concat([trans_calib_TILBasic_test_set_thresh_AUCs,trans_macro_average_thresh_AUCs],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)
    
    # Add macro-averages to threshold-level calibration metrics
    trans_macro_average_thresh_calibration = trans_calib_TILBasic_test_set_thresh_calibration.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
    trans_macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(trans_macro_average_thresh_calibration.shape[0])])
    trans_calib_TILBasic_test_set_thresh_calibration = pd.concat([trans_calib_TILBasic_test_set_thresh_calibration,trans_macro_average_thresh_calibration],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)

    ## Save performance metrics from current resample's testing outputs
    # # Save ORCs of TIL-Basic model from testing set outputs
    # calib_TILBasic_test_set_ORCs.to_pickle(os.path.join(test_bs_dir,'TomorrowTILBasic_test_calibrated_ORCs_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    # # Save Somers' D of TIL-Basic model from testing set outputs
    # calib_TILBasic_test_set_Somers_D.to_pickle(os.path.join(test_bs_dir,'TomorrowTILBasic_test_calibrated_Somers_D_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    # # Save threshold-level calibration AUCs of TIL-Basic model from testing set outputs
    # calib_TILBasic_test_set_thresh_AUCs.to_pickle(os.path.join(test_bs_dir,'TomorrowTILBasic_test_calibrated_AUCs_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    
    # # Save threshold-level calibration metrics of TIL-Basic model from testing set outputs
    # calib_TILBasic_test_set_thresh_calibration.to_pickle(os.path.join(test_bs_dir,'TomorrowTILBasic_test_calibrated_calibration_metrics_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    
    # # Save threshold-level calibration curves of TIL-Basic model from testing set outputs
    # calib_TILBasic_test_set_thresh_calibration_curves.to_pickle(os.path.join(test_bs_dir,'TomorrowTILBasic_test_calibrated_calibration_curves_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    # Save ORCs of TIL-Basic model from testing set outputs at points of transition
    trans_calib_TILBasic_test_set_ORCs.to_pickle(os.path.join(test_bs_dir,'trans_TomorrowTILBasic_test_calibrated_ORCs_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    # Save Somers' D of TIL-Basic model from testing set outputs at points of transition
    trans_calib_TILBasic_test_set_Somers_D.to_pickle(os.path.join(test_bs_dir,'trans_TomorrowTILBasic_test_calibrated_Somers_D_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    # Save threshold-level calibration AUCs of TIL-Basic model from testing set outputs at points of transition
    trans_calib_TILBasic_test_set_thresh_AUCs.to_pickle(os.path.join(test_bs_dir,'trans_TomorrowTILBasic_test_calibrated_AUCs_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    
    # Save threshold-level calibration metrics of TIL-Basic model from testing set outputs at points of transition
    trans_calib_TILBasic_test_set_thresh_calibration.to_pickle(os.path.join(test_bs_dir,'trans_TomorrowTILBasic_test_calibrated_calibration_metrics_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    
    # Save threshold-level calibration curves of TIL-Basic model from testing set outputs at points of transition
    trans_calib_TILBasic_test_set_thresh_calibration_curves.to_pickle(os.path.join(test_bs_dir,'trans_TomorrowTILBasic_test_calibrated_calibration_curves_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)