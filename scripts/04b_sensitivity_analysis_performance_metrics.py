#### Master Script 4b: Calculate calibrated testing set calibration and discrimination for statistical inference in sensitivity analysis models ####
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
from functions.model_building import load_sens_model_outputs
from functions.analysis import prepare_df, calc_ORC, calc_Somers_D, calc_thresh_AUC, calc_thresh_calibration, calc_test_thresh_calib_curves

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

# Define and create subdirectory to store testing set bootstrapping results for sensitivity analysis
sens_bs_dir = os.path.join(model_perf_dir,'sensitivity_bootstrapping')
os.makedirs(sens_bs_dir,exist_ok=True)

## Load fundamental information for model training
# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

# Load bootstrapping resample dataframe for testing set performance
bs_resamples = pd.read_pickle(os.path.join(model_perf_dir,'test_performance_bs_resamples.pkl'))

# Load the optimised tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

### II. Compile calibrated testing set outputs from sensitivity analysis models if not yet completed
# If compiled output dataframe doesn't exist, create it
if not os.path.exists(os.path.join(model_dir,'sens_analysis_TomorrowTILBasic_compiled_test_calibrated_outputs.pkl')):
    
    # Search for all sensitivity analysis output files
    sens_files = []
    for path in Path(model_dir).rglob('sens_analysis_calibrated_test_predictions.csv'):
        sens_files.append(str(path.resolve()))
    
    # Characterise the output files found
    sens_file_info_df = pd.DataFrame({'FILE':sens_files,
                                      'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in sens_files],
                                      'FOLD':[int(re.search('/fold(.*)/tune', curr_file).group(1)) for curr_file in sens_files],
                                      'TUNE_IDX':[re.search('/tune(.*)/sens', curr_file).group(1) for curr_file in sens_files],
                                      'SENS_IDX':[int(re.search('/sens(.*)/sens', curr_file).group(1)) for curr_file in sens_files],
                                      'VERSION':[re.search('_outputs/(.*)/repeat', curr_file).group(1) for curr_file in sens_files],
                                      'CALIBRATION':[re.search('sens_analysis_(.*)_predictions.csv', curr_file).group(1) for curr_file in sens_files],
                                      'SET':[re.search('calibrated_(.*)_predictions.csv', curr_file).group(1) for curr_file in sens_files]
                                     }).sort_values(by=['REPEAT','FOLD','TUNE_IDX','SENS_IDX','SET']).reset_index(drop=True)
    sens_file_info_df['TUNE_IDX'] = sens_file_info_df['TUNE_IDX'].str.rsplit(pat='/', n=1).apply(lambda x: x[0]).astype(int)
    sens_file_info_df['CALIBRATION'] = sens_file_info_df['CALIBRATION'].str.rsplit(pat='_', n=1).apply(lambda x: x[0])
    
    # Merge outcome label to model output dataframe
    sens_file_info_df = sens_file_info_df.merge(tuning_grid[['TUNE_IDX','OUTCOME_LABEL']].drop_duplicates(ignore_index=True),how='left')

    # Load and compile calibrated TomorrowTILBasic testing set outputs
    calib_TILBasic_test_outputs = load_sens_model_outputs(sens_file_info_df[(sens_file_info_df.CALIBRATION=='calibrated')&
                                                                       (sens_file_info_df.SET=='test')&
                                                                       (sens_file_info_df.OUTCOME_LABEL=='TomorrowTILBasic')].reset_index(drop=True),
                                                          True,
                                                          'Loading calibrated TomorrowTILBasic testing set outputs').sort_values(by=['REPEAT','FOLD','TUNE_IDX','SENS_IDX','GUPI']).reset_index(drop=True)
    
    # Save compiled calibrated TomorrowTILBasic testing set outputs
    calib_TILBasic_test_outputs.to_pickle(os.path.join(model_dir,'sens_analysis_TomorrowTILBasic_compiled_test_calibrated_outputs.pkl'))
    
else:
    pass

### II. Calculate testing set calibration and discrimination based on provided bootstrapping resample row index
# Argument-induced bootstrapping functions
def main(array_task_id):
    
    # Extract current bootstrapping resample parameters
    curr_rs_idx = bs_resamples.RESAMPLE_IDX[array_task_id]
    curr_GUPIs = bs_resamples.GUPIs[array_task_id]
    
    # Load compiled testing set outputs
    calib_TILBasic_test_outputs = pd.read_pickle(os.path.join(model_dir,'sens_analysis_TomorrowTILBasic_compiled_test_calibrated_outputs.pkl'))

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

    # Prepare testing set output dataframes for performance calculation
    filt_TILBasic_test_outputs = prepare_df(calib_TILBasic_test_outputs,PERF_WINDOW_INDICES)
    
    ## Iterate through sensitivity indices and calculate testing set performance metrics
    # Create empty lists to store performance metrics
    sens_test_ORCs = []
    sens_test_Somers_D = []
    sens_test_thresh_AUCs = []
    sens_test_thresh_calibration = []
    sens_test_thresh_calibration_curves = []

    # Iterate through all unique sensitivity indices
    for curr_sens_idx in tqdm(filt_TILBasic_test_outputs.SENS_IDX.unique(),'Calculating testing set performance metrics for each sensitivity index'):
        
        # Calculate ORCs of TIL-Basic model on testing set outputs
        calib_TILBasic_test_set_ORCs = calc_ORC(filt_TILBasic_test_outputs[filt_TILBasic_test_outputs.SENS_IDX==curr_sens_idx].reset_index(drop=True),PERF_WINDOW_INDICES,True,'Calculating testing set ORC')

        # Calculate Somers' D of TIL-Basic model on testing set outputs
        calib_TILBasic_test_set_Somers_D = calc_Somers_D(filt_TILBasic_test_outputs[filt_TILBasic_test_outputs.SENS_IDX==curr_sens_idx].reset_index(drop=True),PERF_WINDOW_INDICES,True,'Calculating testing set Somers D')

        # Calculate threshold-level AUC of TIL-Basic model on testing set outputs
        calib_TILBasic_test_set_thresh_AUCs = calc_thresh_AUC(filt_TILBasic_test_outputs[filt_TILBasic_test_outputs.SENS_IDX==curr_sens_idx].reset_index(drop=True),PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level AUC')

        # Calculate threshold-level calibration metrics of TIL-Basic model on testing set outputs
        calib_TILBasic_test_set_thresh_calibration = calc_thresh_calibration(filt_TILBasic_test_outputs[filt_TILBasic_test_outputs.SENS_IDX==curr_sens_idx].reset_index(drop=True),PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level calibration metrics')

        # Calculate threshold-level calibration curves of TIL-Basic model on testing set outputs
        calib_TILBasic_test_set_thresh_calibration_curves = calc_test_thresh_calib_curves(filt_TILBasic_test_outputs[filt_TILBasic_test_outputs.SENS_IDX==curr_sens_idx].reset_index(drop=True),PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level calibration curves')

        # Add macro-averages to threshold-level AUCs
        macro_average_thresh_AUCs = calib_TILBasic_test_set_thresh_AUCs.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
        macro_average_thresh_AUCs.insert(2,'THRESHOLD',['Average' for idx in range(macro_average_thresh_AUCs.shape[0])])
        calib_TILBasic_test_set_thresh_AUCs = pd.concat([calib_TILBasic_test_set_thresh_AUCs,macro_average_thresh_AUCs],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)

        # Add macro-averages to threshold-level calibration metrics
        macro_average_thresh_calibration = calib_TILBasic_test_set_thresh_calibration.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
        macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(macro_average_thresh_calibration.shape[0])])
        calib_TILBasic_test_set_thresh_calibration = pd.concat([calib_TILBasic_test_set_thresh_calibration,macro_average_thresh_calibration],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)
        
        # Add sensitivity index information to all calculated dataframes
        calib_TILBasic_test_set_ORCs['SENS_IDX'] = curr_sens_idx
        calib_TILBasic_test_set_Somers_D['SENS_IDX'] = curr_sens_idx
        calib_TILBasic_test_set_thresh_AUCs['SENS_IDX'] = curr_sens_idx
        calib_TILBasic_test_set_thresh_calibration['SENS_IDX'] = curr_sens_idx
        calib_TILBasic_test_set_thresh_calibration_curves['SENS_IDX'] = curr_sens_idx
        
        # Append dataframes to running lists
        sens_test_ORCs.append(calib_TILBasic_test_set_ORCs)
        sens_test_Somers_D.append(calib_TILBasic_test_set_Somers_D)
        sens_test_thresh_AUCs.append(calib_TILBasic_test_set_thresh_AUCs)
        sens_test_thresh_calibration.append(calib_TILBasic_test_set_thresh_calibration)
        sens_test_thresh_calibration_curves.append(calib_TILBasic_test_set_thresh_calibration_curves)
    
    ## Calculate metric differences from full model
    # Compile lists of dataframes into single dataframe per metric
    sens_test_ORCs = pd.concat(sens_test_ORCs,ignore_index=True)
    sens_test_Somers_D = pd.concat(sens_test_Somers_D,ignore_index=True)
    sens_test_thresh_AUCs = pd.concat(sens_test_thresh_AUCs,ignore_index=True)
    sens_test_thresh_calibration = pd.concat(sens_test_thresh_calibration,ignore_index=True)
    sens_test_thresh_calibration_curves = pd.concat(sens_test_thresh_calibration_curves,ignore_index=True)

    # Concatenate scalar metrics into single dataframe for simplicity
    compiled_sens_test_metrics = pd.concat([sens_test_ORCs,sens_test_Somers_D,sens_test_thresh_AUCs,sens_test_thresh_calibration],ignore_index=True)
    compiled_sens_test_metrics.THRESHOLD = compiled_sens_test_metrics.THRESHOLD.fillna('None')
    
    # Load bootstrapping test set performance metrics from full model
    compiled_test_bootstrapping_metrics = pd.read_pickle(os.path.join(model_perf_dir,'test_bootstrapping_calibrated_metrics.pkl'))
    
    # Ascribe resampling indices based on place in dataframe
    compiled_test_bootstrapping_metrics['RESAMPLE_IDX'] = compiled_test_bootstrapping_metrics.groupby(['TUNE_IDX','WINDOW_IDX','METRIC','THRESHOLD']).cumcount()+1

    # Filter to current resampling index
    curr_full_test_metrics = compiled_test_bootstrapping_metrics[compiled_test_bootstrapping_metrics['RESAMPLE_IDX'] == curr_rs_idx].reset_index(drop=True).rename(columns={'VALUE':'FULL_MODEL_VALUE'})
    
    # Calculate difference for discrimination metrics
    compiled_sens_test_discrimination = compiled_sens_test_metrics[compiled_sens_test_metrics.METRIC.isin(['ORC','Somers D','AUC'])].merge(curr_full_test_metrics,how='left')
    compiled_sens_test_discrimination['SENS_DIFFERENCE'] = compiled_sens_test_discrimination.FULL_MODEL_VALUE - compiled_sens_test_discrimination.VALUE
    
    # Calculate error difference for calibration slope
    compiled_sens_test_calib_slope = compiled_sens_test_metrics[compiled_sens_test_metrics.METRIC.isin(['CALIB_SLOPE'])].merge(curr_full_test_metrics,how='left')
    compiled_sens_test_calib_slope['SENS_DIFFERENCE'] = (compiled_sens_test_calib_slope.FULL_MODEL_VALUE-1).abs() - (compiled_sens_test_calib_slope.VALUE-1).abs()
    
    # Calculate error difference for other calibration metrics
    compiled_sens_test_other_calib = compiled_sens_test_metrics[compiled_sens_test_metrics.METRIC.isin(['Emax','ICI'])].merge(curr_full_test_metrics,how='left')
    compiled_sens_test_other_calib['SENS_DIFFERENCE'] = (compiled_sens_test_other_calib.FULL_MODEL_VALUE).abs() - (compiled_sens_test_other_calib.VALUE).abs()
    
    ## Save performance metrics from current resample's testing outputs
    # Compile the difference-in-metric scores
    compiled_sens_test_differences = pd.concat([compiled_sens_test_discrimination,compiled_sens_test_calib_slope,compiled_sens_test_other_calib],ignore_index=True).drop(columns=['FULL_MODEL_VALUE'])
    
    # Reorder columns
    compiled_sens_test_differences = compiled_sens_test_differences[['RESAMPLE_IDX','TUNE_IDX','SENS_IDX','WINDOW_IDX','THRESHOLD','METRIC','VALUE','SENS_DIFFERENCE']]
    
    # Save scalar metrics of TIL-Basic model from testing set outputs
    compiled_sens_test_differences.to_pickle(os.path.join(sens_bs_dir,'sens_analysis_TomorrowTILBasic_test_calibrated_metrics_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))
    
    # Save threshold-level calibration curves of TIL-Basic model from testing set outputs
    sens_test_thresh_calibration_curves.to_pickle(os.path.join(sens_bs_dir,'sens_analysis_TomorrowTILBasic_test_calibrated_calibration_curves_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)