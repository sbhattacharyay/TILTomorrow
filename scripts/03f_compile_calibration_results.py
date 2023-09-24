#### Master Script 03f: Compile and examine calibration performance metrics ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile all calibration performance metrics
# III. Calculate average effect of calibration methods on each tuning configuration
# IV. Compile calibrated outputs corresponding to best configurations

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
import seaborn as sns
import multiprocessing
from scipy import stats
from pathlib import Path
from ast import literal_eval
import matplotlib.pyplot as plt
from collections import Counter
from argparse import ArgumentParser
from pandas.api.types import CategoricalDtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# TQDM for progress tracking
from tqdm import tqdm

## Define parameters for model training
# Set version code
VERSION = 'v1-0'

# Define number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Window indices at which to calculate performance metrics
PERF_WINDOW_INDICES = [1,2,3,4,5,6,9,13,20]

## Define and create relevant directories
# Initialise model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)

# Define model performance directory based on version code
model_perf_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_performance',VERSION)

# Define directory in which calibration performance results are stored
calibration_dir = os.path.join(model_perf_dir,'calibration_performance')

## Load fundamental information for model training
# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

# Load the post-dropout tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))

# Focus calibration efforts on TILBasic models
tuning_grid = tuning_grid[tuning_grid.OUTCOME_LABEL=='TomorrowTILBasic'].reset_index(drop=True)

### II. Compile all calibration performance metrics
# Search for all calibration metric files in the directory
calib_metric_files = []
for path in Path(os.path.join(calibration_dir)).rglob('*.pkl'):
    calib_metric_files.append(str(path.resolve()))
    
# Characterise calibration metric files
calib_file_info_df = pd.DataFrame({'file':calib_metric_files,
                                   'TUNE_IDX':[int(re.search('/tune_idx_(.*)_opt_', curr_file).group(1)) for curr_file in calib_metric_files],
                                   'OPTIMIZATION':[re.search('_opt_(.*)_window_idx_', curr_file).group(1) for curr_file in calib_metric_files],
                                   'WINDOW_IDX':[int(re.search('_window_idx_(.*)_scaling_', curr_file).group(1)) for curr_file in calib_metric_files],
                                   'SCALING':[re.search('_scaling_(.*).pkl', curr_file).group(1) for curr_file in calib_metric_files],
                                   'VERSION':[re.search('_performance/(.*)/calibration_performance', curr_file).group(1) for curr_file in calib_metric_files],
                                   'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in calib_metric_files],
                                   'FOLD':[int(re.search('/fold(.*)/tune_idx_', curr_file).group(1)) for curr_file in calib_metric_files]
                                  }).sort_values(['TUNE_IDX','REPEAT','FOLD','WINDOW_IDX','OPTIMIZATION','SCALING']).reset_index(drop=True)

# Load calibration metric files
calibration_metrics_df = pd.concat([pd.read_pickle(f) for f in tqdm(calib_metric_files)],ignore_index=True)

# Sort compiled calibration metric dataframe
calibration_metrics_df = calibration_metrics_df.sort_values(by=['TUNE_IDX','REPEAT','FOLD','WINDOW_IDX','THRESHOLD','OPTIMIZATION','METRIC','SET','CALIBRATION'],ignore_index=True).drop_duplicates(ignore_index=True)

# Save compiled pre- and post-calibration metrics
calibration_metrics_df.to_csv(os.path.join(calibration_dir,'compiled_calibration_metrics.csv'),index=False)

### III. Calculate average effect of calibration methods on each tuning configuration
## Load and prepare calibration metrics
# Load compiled pre- and post-calibration metrics
calibration_metrics_df = pd.read_csv(os.path.join(calibration_dir,'compiled_calibration_metrics.csv'))

# Average metric values across repeated cross-validation folds
ave_calibration_metrics_df = calibration_metrics_df.groupby(['TUNE_IDX','OPTIMIZATION','WINDOW_IDX','THRESHOLD','SET','CALIBRATION','METRIC'],as_index=False)['VALUE'].mean()

# Extract calibration slope values
calibration_slopes_df = ave_calibration_metrics_df[ave_calibration_metrics_df.METRIC=='CALIB_SLOPE'].reset_index(drop=True)

# Calculate error in calibration slopes
calibration_slopes_df['ERROR'] = (calibration_slopes_df['VALUE']-1).abs()

# Extract ORC values
ORCs_df = ave_calibration_metrics_df[ave_calibration_metrics_df.METRIC=='ORC'].reset_index(drop=True)

## Calculate effect of calibration methods on each metric
# For each model output combination, determine the lowest-error configuration
best_cal_slopes_combos = calibration_slopes_df.loc[calibration_slopes_df.groupby(['TUNE_IDX','WINDOW_IDX','THRESHOLD','SET','METRIC']).ERROR.idxmin()].reset_index(drop=True)

# Focus on macro-averaged calibration slopes
best_cal_slopes_combos = best_cal_slopes_combos[best_cal_slopes_combos.THRESHOLD=='Average'].reset_index(drop=True)

# Merge ORC information to best calibration slope combinations
best_cal_slopes_combos = best_cal_slopes_combos.merge(ORCs_df[['TUNE_IDX','OPTIMIZATION','WINDOW_IDX','SET','CALIBRATION','VALUE']].rename(columns={'VALUE':'ORC'}),how='left')

# For each model output combination, determine the best-discriminating configuration
best_ORC_combos = ORCs_df.loc[ORCs_df.groupby(['TUNE_IDX','WINDOW_IDX','THRESHOLD','SET','METRIC']).VALUE.idxmax()].reset_index(drop=True).rename(columns={'VALUE':'BEST_ORC'})

# Merge optmial ORC information to best calibration slope combinations
best_cal_slopes_combos = best_cal_slopes_combos.merge(best_ORC_combos[['TUNE_IDX','WINDOW_IDX','SET','BEST_ORC']],how='left')

# Calculate change from optimal ORC
best_cal_slopes_combos['CHANGE_IN_ORC'] = best_cal_slopes_combos['ORC'] - best_cal_slopes_combos['BEST_ORC']

# Save best calibration slope combination information
best_cal_slopes_combos.to_csv(os.path.join(calibration_dir,'best_calibration_configurations.csv'),index=False)

### IV. Compile calibrated outputs corresponding to best configurations
## Prepare environment for calibrated output extraction
# Load best calibration slope combination information
best_cal_slopes_combos = pd.read_csv(os.path.join(calibration_dir,'best_calibration_configurations.csv'))

# Determine unique partition-configuration combinations
config_partition_combos = calibration_metrics_df[['TUNE_IDX','REPEAT','FOLD']].drop_duplicates(ignore_index=True)

# Filter instances in each tuning configuration improved by calibration method (without dropping columns)
calibrated_instances = best_cal_slopes_combos[best_cal_slopes_combos.CALIBRATION!='None'].rename(columns={'WINDOW_IDX':'WindowIdx'})

# Expand calibrated instances to include all repeated cross-validation partitions
calibrated_instances = config_partition_combos.merge(calibrated_instances)

# Add file paths of corresponding calibrated model outputs
calibrated_instances['FILE_PATH'] = model_dir+'/'+'repeat'+str(calibrated_instances.REPEAT).zfill(2)+'/'+'fold'+str(calibrated_instances.FOLD).zfill(1)+'/'+'tune'+str(calibrated_instances.TUNE_IDX).zfill(4)+'/'+'set_'+calibrated_instances.SET+'_opt_'+calibrated_instances.OPTIMIZATION+'_window_idx_'+str(calibrated_instances.WindowIdx).zfill(2)+'_scaling_'+calibrated_instances.CALIBRATION+'.pkl'

# Check if created file path exists




# Load and filter compiled, uncalibrated validation set outputs
compiled_val_uncalib_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_val_uncalibrated_outputs.pkl'))
compiled_val_uncalib_outputs = compiled_val_uncalib_outputs[compiled_val_uncalib_outputs.TUNE_IDX.isin(best_cal_slopes_combos.TUNE_IDX)].reset_index(drop=True)

# Load compiled, uncalibrated testing set outputs
compiled_test_uncalib_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_test_uncalibrated_outputs.pkl'))
compiled_test_uncalib_outputs = compiled_test_uncalib_outputs[compiled_test_uncalib_outputs.TUNE_IDX.isin(best_cal_slopes_combos.TUNE_IDX)].reset_index(drop=True)

## First, extract instances in which uncalibrated output is optimal
# Deterime instances in each tuning configuration improved by calibration method
calibrated_instances = best_cal_slopes_combos[best_cal_slopes_combos.CALIBRATION!='None'][['TUNE_IDX','WINDOW_IDX','SET']].drop_duplicates(ignore_index=True).rename(columns={'WINDOW_IDX':'WindowIdx'})

# Merge calibrated instance information with compiled, uncalibrated validation set outputs
compiled_val_calib_outputs = compiled_val_uncalib_outputs.merge(calibrated_instances,indicator=True,how='left')

# Remove instances which are improved by calibration
compiled_val_calib_outputs = compiled_val_calib_outputs[compiled_val_calib_outputs._merge=='left_only'].drop(columns='_merge').reset_index(drop=True)

# Merge calibrated instance information with compiled, uncalibrated testing set outputs
compiled_test_calib_outputs = compiled_test_uncalib_outputs.merge(calibrated_instances,indicator=True,how='left')

# Remove instances which are improved by calibration
compiled_test_calib_outputs = compiled_test_calib_outputs[compiled_test_calib_outputs._merge=='left_only'].drop(columns='_merge').reset_index(drop=True)

## Next, extract optimal, calibrated outputs based on instances of minimum calibration slope error
# Filter instances in each tuning configuration improved by calibration method (without dropping columns)
calibrated_instances = best_cal_slopes_combos[best_cal_slopes_combos.CALIBRATION!='None'].rename(columns={'WINDOW_IDX':'WindowIdx'})

# Expand calibrated instances to include all repeated cross-validation partitions
calibrated_instances = config_partition_combos.merge(calibrated_instances)

# Add file paths of corresponding calibrated model outputs
calibrated_instances['FILE_PATH'] = model_dir+'/'+'repeat'+str(calibrated_instances.REPEAT).zfill(2)+'/'+'fold'+str(calibrated_instances.FOLD).zfill(1)+'/'+'tune'+str(calibrated_instances.TUNE_IDX).zfill(4)+'/'+'set_'+calibrated_instances.SET+'_opt_'+calibrated_instances.OPTIMIZATION+'_window_idx_'+str(calibrated_instances.WindowIdx).zfill(2)+'_scaling_'+calibrated_instances.CALIBRATION+'.pkl'





# # Extract unique tuning indices
# uniq_tune_idx = best_cal_slopes_combos.TUNE_IDX.unique()

# # Create running lists to store calibrated model outputs
# calibrated_val_outputs = []
# calibrated_test_outputs = []

# a = pd.read_pickle('../TILTomorrow_model_outputs/v1-0/repeat01/fold1/tune0277/set_test_opt_nominal_window_idx_03_scaling_T.pkl')
# b = pd.read_csv('../TILTomorrow_model_outputs/v1-0/repeat01/fold1/tune0277/uncalibrated_test_predictions.csv')

# c = pd.read_pickle('../TILTomorrow_model_outputs/v1-0/TomorrowTILBasic_compiled_val_uncalibrated_outputs.pkl')

# ## Iterate through unique tuning configuration indices to extract optimal calibrated outputs
# for curr_ti in tqdm(uniq_tune_idx,'Extract calibrated model outputs corresponding to best configurations'):
    
#     # Filter best calibration slope combination information to current tuning index
#     curr_best_cal_slopes_combos = best_cal_slopes_combos[best_cal_slopes_combos]

#     # Extract parameters of current partition-configuration combination
#     curr_repeat = calibrated_output_combos.REPEAT[curr_ti]
#     curr_fold = calibrated_output_combos.FOLD[curr_ti]
    
    
#     tune_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(1),'tune'+str(curr_ti).zfill(4))

#     if curr_ti == 69:
#         curr_best_combos = best_69_combos.copy()
        
#     elif curr_ti == 135:
#         curr_best_combos = best_135_combos.copy()
        
#     uncalib_val_preds = pd.read_csv(os.path.join(tune_dir,'uncalibrated_val_predictions.csv'))
#     uncalib_val_preds['WindowIdx'] = uncalib_val_preds.groupby('GUPI').cumcount(ascending=True)+1
#     uncalib_val_preds['REPEAT'] = curr_repeat
#     uncalib_val_preds['FOLD'] = curr_fold
#     calib_val_preds = []
#     calib_val_preds.append(uncalib_val_preds[uncalib_val_preds.WindowIdx >= 85].reset_index(drop=True))
#     none_window_indices = curr_best_combos[curr_best_combos.CALIBRATION == 'None'].WINDOW_IDX.unique()
#     calib_val_preds.append(uncalib_val_preds[uncalib_val_preds.WindowIdx.isin(none_window_indices)].reset_index(drop=True))
#     remaining_combos = curr_best_combos[curr_best_combos.CALIBRATION != 'None'].reset_index(drop=True)
#     for curr_rem_row_idx in range(remaining_combos.shape[0]):
#         curr_optimization = remaining_combos.OPTIMIZATION[curr_rem_row_idx]
#         curr_calibration = remaining_combos.CALIBRATION[curr_rem_row_idx]
#         curr_window_index = remaining_combos.WINDOW_IDX[curr_rem_row_idx]
#         curr_calib_preds = pd.read_pickle(os.path.join(tune_dir,'set_val_opt_'+curr_optimization+'_window_idx_'+str(curr_window_index).zfill(2)+'_scaling_'+curr_calibration+'.pkl'))
#         calib_val_preds.append(curr_calib_preds)
#     calib_val_preds = pd.concat(calib_val_preds,ignore_index=True).sort_values(['GUPI','WindowIdx']).reset_index(drop=True)
#     calib_val_preds.to_csv(os.path.join(tune_dir,'calibrated_val_predictions.csv'),index=False)
    
#     uncalib_test_preds = pd.read_csv(os.path.join(tune_dir,'uncalibrated_test_predictions.csv'))
#     uncalib_test_preds['WindowIdx'] = uncalib_test_preds.groupby('GUPI').cumcount(ascending=True)+1
#     uncalib_test_preds['REPEAT'] = curr_repeat
#     uncalib_test_preds['FOLD'] = curr_fold
#     calib_test_preds = []
#     calib_test_preds.append(uncalib_test_preds[uncalib_test_preds.WindowIdx >= 85].reset_index(drop=True))
#     none_window_indices = curr_best_combos[curr_best_combos.CALIBRATION == 'None'].WINDOW_IDX.unique()
#     calib_test_preds.append(uncalib_test_preds[uncalib_test_preds.WindowIdx.isin(none_window_indices)].reset_index(drop=True))
#     remaining_combos = curr_best_combos[curr_best_combos.CALIBRATION != 'None'].reset_index(drop=True)
#     for curr_rem_row_idx in range(remaining_combos.shape[0]):
#         curr_optimization = remaining_combos.OPTIMIZATION[curr_rem_row_idx]
#         curr_calibration = remaining_combos.CALIBRATION[curr_rem_row_idx]
#         curr_window_index = remaining_combos.WINDOW_IDX[curr_rem_row_idx]
#         curr_calib_preds = pd.read_pickle(os.path.join(tune_dir,'set_test_opt_'+curr_optimization+'_window_idx_'+str(curr_window_index).zfill(2)+'_scaling_'+curr_calibration+'.pkl'))
#         calib_test_preds.append(curr_calib_preds)
#     calib_test_preds = pd.concat(calib_test_preds,ignore_index=True).sort_values(['GUPI','WindowIdx']).reset_index(drop=True)
#     calib_test_preds.to_csv(os.path.join(tune_dir,'calibrated_test_predictions.csv'),index=False)