#### Master Script 03d: Compile validation set performance results for configuration dropout of TILTomorrow models ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile and save bootstrapped validation set performance dataframes
# III. Dropout configurations based on validation set calibration and discrimination information
# IV. Delete folders of underperforming configurations
# V. Visualise hyperparameter optimisation results

### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import random
import shutil
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
from argparse import ArgumentParser
from collections import Counter, OrderedDict
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

# HiPlot methods
import hiplot as hip

# StatsModel methods
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from statsmodels.nonparametric.smoothers_lowess import lowess

# Custom methods
from functions.analysis import prepare_df, calc_ORC, calc_AUC, calc_Somers_D, calc_thresh_calibration, calc_binary_calibration

## Define parameters for model training
# Set version code
VERSION = 'v2-0'

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Set number of resamples for bootstrapping-based testing set performance
NUM_RESAMP = 1000

# Window indices at which to calculate performance metrics
PERF_WINDOW_INDICES = [1,2,3,4,5,6,9,13]

## Define and create relevant directories
# Define model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)

# Define model performance directory based on version code
model_perf_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_performance',VERSION)

# Define subdirectory to store validation set bootstrapping results
val_bs_dir = os.path.join(model_perf_dir,'validation_set_bootstrapping')

## Load fundamental information for model training
# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

# Load the optimised tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

### II. Compile and save bootstrapped validation set performance dataframes
## Find and characterise all validation set performance files
# Search for all performance files
perf_files = []
for path in Path(val_bs_dir).rglob('TomorrowTILBasic_val_*'):
    perf_files.append(str(path.resolve()))
for path in Path(val_bs_dir).rglob('TomorrowHighIntensityTherapy_val_*'):
    perf_files.append(str(path.resolve()))

# Characterise the performance files found
perf_file_info_df = pd.DataFrame({'FILE':perf_files,
                                  'VERSION':[re.search('_performance/(.*)/validation_set_', curr_file).group(1) for curr_file in perf_files],
                                  'OUTCOME_LABEL':[re.search('_bootstrapping/(.*)_val_', curr_file).group(1) for curr_file in perf_files],
                                  'METRIC':[re.search('val_uncalibrated_(.*)_rs_', curr_file).group(1) for curr_file in perf_files],
                                  'RESAMPLE_IDX':[int(re.search('_rs_(.*).pkl', curr_file).group(1)) for curr_file in perf_files],
                                 }).sort_values(by=['METRIC','RESAMPLE_IDX']).reset_index(drop=True)

# Separate ORC and calibration file dataframes
orc_file_info_df = perf_file_info_df[perf_file_info_df.METRIC == 'ORCs'].reset_index(drop=True)
auc_file_info_df = perf_file_info_df[perf_file_info_df.METRIC == 'AUCs'].reset_index(drop=True)
somers_d_file_info_df = perf_file_info_df[perf_file_info_df.METRIC == 'Somers_D'].reset_index(drop=True)
thresh_calibration_file_info_df = perf_file_info_df[(perf_file_info_df.METRIC == 'calibration_metrics')&(perf_file_info_df.OUTCOME_LABEL == 'TomorrowTILBasic')].reset_index(drop=True)
binary_calibration_file_info_df = perf_file_info_df[(perf_file_info_df.METRIC == 'calibration_metrics')&(perf_file_info_df.OUTCOME_LABEL == 'TomorrowHighIntensityTherapy')].reset_index(drop=True)

## Load and compile validation set performance dataframes into single files
# Load validation set discrimination and calibration performance dataframes
TILBasic_compiled_val_orc = pd.concat([pd.read_pickle(f) for f in tqdm(orc_file_info_df.FILE,'Load and compile validation set ORC values')],ignore_index=True)
highTIL_compiled_val_auc = pd.concat([pd.read_pickle(f) for f in tqdm(auc_file_info_df.FILE,'Load and compile validation set AUC values')],ignore_index=True)
compiled_val_somers_d = pd.concat([pd.read_pickle(f) for f in tqdm(somers_d_file_info_df.FILE,'Load and compile validation set Somers D values')],ignore_index=True)
TILBasic_compiled_val_calibration = pd.concat([pd.read_pickle(f) for f in tqdm(thresh_calibration_file_info_df.FILE,'Load and compile validation set threshold-level calibration metrics')],ignore_index=True)
highTIL_compiled_val_calibration = pd.concat([pd.read_pickle(f) for f in tqdm(binary_calibration_file_info_df.FILE,'Load and compile validation set binary calibration metrics')],ignore_index=True)

# Concatenate dataframes
compiled_val_bootstrapping_metrics = pd.concat([TILBasic_compiled_val_orc,highTIL_compiled_val_auc,compiled_val_somers_d,TILBasic_compiled_val_calibration,highTIL_compiled_val_calibration],ignore_index=True)

# Save compiled validation set performance metrics
compiled_val_bootstrapping_metrics.to_pickle(os.path.join(model_perf_dir,'val_bootstrapping_uncalibrated_metrics.pkl'))

## Delete individual files once compiled dataframe has been saved
# Iterate through performance metric files and delete
_ = [os.remove(f) for f in tqdm(perf_file_info_df.FILE,'Clearing validation bootstrapping metric files after collection')]

### III. Dropout configurations based on validation set calibration and discrimination information
## Load and separate compiled validation bootstrapping performance metrics
# Load compiled validation set performance metrics
compiled_val_bootstrapping_metrics = pd.read_pickle(os.path.join(model_perf_dir,'val_bootstrapping_uncalibrated_metrics.pkl'))

# Fill in missing THRESHOLD value
compiled_val_bootstrapping_metrics['THRESHOLD'] = compiled_val_bootstrapping_metrics.THRESHOLD.fillna('None')

# Add a resample index column
compiled_val_bootstrapping_metrics['RESAMPLE_IDX'] = compiled_val_bootstrapping_metrics.groupby(['TUNE_IDX','WINDOW_IDX','METRIC','THRESHOLD']).cumcount()+1

# Extract metric by type
TILBasic_compiled_val_orc = compiled_val_bootstrapping_metrics[compiled_val_bootstrapping_metrics.METRIC=='ORC'].reset_index(drop=True)
highTIL_compiled_val_auc = compiled_val_bootstrapping_metrics[compiled_val_bootstrapping_metrics.METRIC=='AUC'].reset_index(drop=True)
TILBasic_compiled_val_calibration = compiled_val_bootstrapping_metrics[(compiled_val_bootstrapping_metrics.METRIC=='CALIB_SLOPE')&(compiled_val_bootstrapping_metrics.THRESHOLD!='None')].reset_index(drop=True)
highTIL_compiled_val_calibration = compiled_val_bootstrapping_metrics[(compiled_val_bootstrapping_metrics.METRIC=='CALIB_SLOPE')&(compiled_val_bootstrapping_metrics.THRESHOLD=='None')].reset_index(drop=True)

## Identify configurations with calibration slope 1 in confidence interval
# Calculate confidence intervals for each tuning configuration
TILBasic_val_CI_calibration_slope = TILBasic_compiled_val_calibration.groupby(['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC'],as_index=False)['VALUE'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'hi':lambda x: np.quantile(x.dropna(),.975),'resamples':'count'}).reset_index(drop=True)
highTIL_val_CI_calibration_slope = highTIL_compiled_val_calibration.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False)['VALUE'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'hi':lambda x: np.quantile(x.dropna(),.975),'resamples':'count'}).reset_index(drop=True)

# Mark TUNE_IDX/WINDOW_IDX combinations which are significantly calibrated
TILBasic_val_CI_calibration_slope['CALIBRATED'] = ((TILBasic_val_CI_calibration_slope['lo']<=1)&(TILBasic_val_CI_calibration_slope['hi']>=1))
highTIL_val_CI_calibration_slope['CALIBRATED'] = ((highTIL_val_CI_calibration_slope['lo']<=1)&(highTIL_val_CI_calibration_slope['hi']>=1))

# Concatenate tuning indices of significantly calibrated configurations for each window index
TILBasic_val_calibrated_TIs = TILBasic_val_CI_calibration_slope[(TILBasic_val_CI_calibration_slope.CALIBRATED)&(TILBasic_val_CI_calibration_slope.THRESHOLD=='Average')].groupby(['WINDOW_IDX'],as_index=False).TUNE_IDX.aggregate(list).rename(columns={'TUNE_IDX':'CALIB_TUNE_IDX'})
highTIL_val_calibrated_TIs = highTIL_val_CI_calibration_slope[highTIL_val_CI_calibration_slope.CALIBRATED].groupby(['WINDOW_IDX'],as_index=False).TUNE_IDX.aggregate(list).rename(columns={'TUNE_IDX':'CALIB_TUNE_IDX'})

## Identify configurations that are significantly worse calibrated than the optimal tuning index
# Load optimal tuning configurations for each window index based on validation set performance
TILBasic_opt_val_calibration_configs = pd.read_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_optimal_val_set_calibration_configurations.csv'))
highTIL_opt_val_calibration_configs = pd.read_csv(os.path.join(model_perf_dir,'TomorrowHighIntensityTherapy_optimal_val_set_calibration_configurations.csv'))

# Extract optimal-configuration TILBasic error per window index
TILBasic_opt_val_calibration_configs = TILBasic_opt_val_calibration_configs[['TUNE_IDX','WINDOW_IDX']].merge(TILBasic_compiled_val_calibration[TILBasic_compiled_val_calibration.THRESHOLD=='Average'],how='left').rename(columns={'TUNE_IDX':'OPT_TUNE_IDX'})
TILBasic_opt_val_calibration_configs['OPT_ERROR'] = (1-TILBasic_opt_val_calibration_configs['VALUE']).abs()

# Extract optimal-configuration highTIL error per window index
highTIL_opt_val_calibration_configs = highTIL_opt_val_calibration_configs[['TUNE_IDX','WINDOW_IDX']].merge(highTIL_compiled_val_calibration,how='left').rename(columns={'TUNE_IDX':'OPT_TUNE_IDX'})
highTIL_opt_val_calibration_configs['OPT_ERROR'] = (1-highTIL_opt_val_calibration_configs['VALUE']).abs()

# Calculate calibration slope error in compiled dataframes
TILBasic_compiled_val_calibration['ERROR'] = (1-TILBasic_compiled_val_calibration['VALUE']).abs()
highTIL_compiled_val_calibration['ERROR'] = (1-highTIL_compiled_val_calibration['VALUE']).abs()

# For TILBasic models, focus on macro-averaged calibration slopes
TILBasic_compiled_val_calibration = TILBasic_compiled_val_calibration[TILBasic_compiled_val_calibration.THRESHOLD=='Average'].reset_index(drop=True)

# Merge optimal-configuration error values onto compiled dataframes
TILBasic_compiled_val_calibration = TILBasic_compiled_val_calibration.merge(TILBasic_opt_val_calibration_configs[['WINDOW_IDX','METRIC','THRESHOLD','RESAMPLE_IDX','OPT_TUNE_IDX','OPT_ERROR']],how='left')
highTIL_compiled_val_calibration = highTIL_compiled_val_calibration.merge(highTIL_opt_val_calibration_configs[['WINDOW_IDX','METRIC','THRESHOLD','RESAMPLE_IDX','OPT_TUNE_IDX','OPT_ERROR']],how='left')

# For each TILBasic WINDOW_IDX/TUNE_IDX combination, calculate the number of times the error is better than the optimal configuration error
TILBasic_compiled_val_calibration['BETTER_THAN_OPT'] = (TILBasic_compiled_val_calibration.ERROR <= TILBasic_compiled_val_calibration.OPT_ERROR)
TILBasic_sig_worse_configs = TILBasic_compiled_val_calibration.groupby(['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC','OPT_TUNE_IDX'],as_index=False).BETTER_THAN_OPT.aggregate({'BETTER':'sum','RESAMPLES':'count'}).reset_index(drop=True)
TILBasic_sig_worse_configs['p'] = TILBasic_sig_worse_configs.BETTER/TILBasic_sig_worse_configs.RESAMPLES

# For each highTIL WINDOW_IDX/TUNE_IDX combination, calculate the number of times the error is better than the optimal configuration error
highTIL_compiled_val_calibration['BETTER_THAN_OPT'] = (highTIL_compiled_val_calibration.ERROR <= highTIL_compiled_val_calibration.OPT_ERROR)
highTIL_sig_worse_configs = highTIL_compiled_val_calibration.groupby(['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC','OPT_TUNE_IDX'],as_index=False).BETTER_THAN_OPT.aggregate({'BETTER':'sum','RESAMPLES':'count'}).reset_index(drop=True)
highTIL_sig_worse_configs['p'] = highTIL_sig_worse_configs.BETTER/highTIL_sig_worse_configs.RESAMPLES

# For each WINDOW_IDX, concatenate tuning indices of significantly miscalibrated configurations
TILBasic_sig_worse_configs = TILBasic_sig_worse_configs[TILBasic_sig_worse_configs.p<.05].groupby(['WINDOW_IDX'],as_index=False).TUNE_IDX.aggregate(list)
highTIL_sig_worse_configs = highTIL_sig_worse_configs[highTIL_sig_worse_configs.p<.05].groupby(['WINDOW_IDX'],as_index=False).TUNE_IDX.aggregate(list)

# Merge information of TILBasic tuning indices which are significantly calibrated
TILBasic_sig_worse_configs = TILBasic_sig_worse_configs.merge(TILBasic_val_calibrated_TIs,how='left')
TILBasic_sig_worse_configs.TUNE_IDX[~TILBasic_sig_worse_configs.CALIB_TUNE_IDX.isna()] = TILBasic_sig_worse_configs[~TILBasic_sig_worse_configs.CALIB_TUNE_IDX.isna()].apply(lambda x:list(set(x['TUNE_IDX'])-set(x['CALIB_TUNE_IDX'])),axis=1)

# Merge information of highTIL tuning indices which are significantly calibrated
highTIL_sig_worse_configs = highTIL_sig_worse_configs.merge(highTIL_val_calibrated_TIs,how='left')
highTIL_sig_worse_configs.TUNE_IDX[~highTIL_sig_worse_configs.CALIB_TUNE_IDX.isna()] = highTIL_sig_worse_configs[~highTIL_sig_worse_configs.CALIB_TUNE_IDX.isna()].apply(lambda x:list(set(x['TUNE_IDX'])-set(x['CALIB_TUNE_IDX'])),axis=1)

# Drop out TILBasic configurations that are consistently out of range and/or significantly underperforming
TILBasic_flattened_TIs = [item for sublist in TILBasic_sig_worse_configs.TUNE_IDX[TILBasic_sig_worse_configs.WINDOW_IDX<=6] for item in sublist] 
TILBasic_counts_of_removal = OrderedDict(Counter(TILBasic_flattened_TIs).most_common())
TILBasic_tune_idx_to_remove = [k for (k,v) in TILBasic_counts_of_removal.items() if v >= 2]
TILBasic_tune_idx_to_keep = [k for k in TILBasic_compiled_val_calibration.TUNE_IDX.unique() if k not in TILBasic_tune_idx_to_remove]

# Drop out highTIL configurations that are consistently out of range and/or significantly underperforming
highTIL_flattened_TIs = [item for sublist in highTIL_sig_worse_configs.TUNE_IDX[highTIL_sig_worse_configs.WINDOW_IDX<=6] for item in sublist] 
highTIL_counts_of_removal = OrderedDict(Counter(highTIL_flattened_TIs).most_common())
highTIL_tune_idx_to_remove = [k for (k,v) in highTIL_counts_of_removal.items() if v >= 1]
highTIL_tune_idx_to_keep = [k for k in highTIL_compiled_val_calibration.TUNE_IDX.unique() if k not in highTIL_tune_idx_to_remove]

## Identify configurations that are significantly less discriminating than the optimal tuning index
# Load optimal tuning configurations for each window index based on validation set performance
TILBasic_opt_val_ORC_configs = pd.read_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_optimal_val_set_discrimination_configurations.csv'))
highTIL_opt_val_AUC_configs = pd.read_csv(os.path.join(model_perf_dir,'TomorrowHighIntensityTherapy_optimal_val_set_discrimination_configurations.csv'))

# Extract optimal-configuration TILBasic ORC and highTIL AUC per window index
TILBasic_opt_val_ORC_configs = TILBasic_opt_val_ORC_configs[['TUNE_IDX','WINDOW_IDX']].merge(TILBasic_compiled_val_orc,how='left').rename(columns={'TUNE_IDX':'OPT_TUNE_IDX','VALUE':'OPT_VALUE'})
highTIL_opt_val_AUC_configs = highTIL_opt_val_AUC_configs[['TUNE_IDX','WINDOW_IDX']].merge(highTIL_compiled_val_auc,how='left').rename(columns={'TUNE_IDX':'OPT_TUNE_IDX','VALUE':'OPT_VALUE'})

# Merge optimal-configuration discrmination values onto compiled dataframes
TILBasic_compiled_val_orc = TILBasic_compiled_val_orc.merge(TILBasic_opt_val_ORC_configs[['WINDOW_IDX','METRIC','THRESHOLD','RESAMPLE_IDX','OPT_TUNE_IDX','OPT_VALUE']],how='left')
highTIL_compiled_val_auc = highTIL_compiled_val_auc.merge(highTIL_opt_val_AUC_configs[['WINDOW_IDX','METRIC','THRESHOLD','RESAMPLE_IDX','OPT_TUNE_IDX','OPT_VALUE']],how='left')

# For each TILBasic WINDOW_IDX/TUNE_IDX combination, calculate the number of times the ORC is better than the optimal configuration ORC
TILBasic_compiled_val_orc['BETTER_THAN_OPT'] = (TILBasic_compiled_val_orc.VALUE >= TILBasic_compiled_val_orc.OPT_VALUE)
TILBasic_sig_less_discrim_configs = TILBasic_compiled_val_orc.groupby(['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC','OPT_TUNE_IDX'],as_index=False).BETTER_THAN_OPT.aggregate({'BETTER':'sum','RESAMPLES':'count'}).reset_index(drop=True)
TILBasic_sig_less_discrim_configs['p'] = TILBasic_sig_less_discrim_configs.BETTER/TILBasic_sig_less_discrim_configs.RESAMPLES

# For each highTIL WINDOW_IDX/TUNE_IDX combination, calculate the number of times the error is better than the optimal configuration error
highTIL_compiled_val_auc['BETTER_THAN_OPT'] = (highTIL_compiled_val_auc.VALUE >= highTIL_compiled_val_auc.OPT_VALUE)
highTIL_sig_less_discrim_configs = highTIL_compiled_val_auc.groupby(['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC','OPT_TUNE_IDX'],as_index=False).BETTER_THAN_OPT.aggregate({'BETTER':'sum','RESAMPLES':'count'}).reset_index(drop=True)
highTIL_sig_less_discrim_configs['p'] = highTIL_sig_less_discrim_configs.BETTER/highTIL_sig_less_discrim_configs.RESAMPLES

# For each WINDOW_IDX, concatenate tuning indices of significantly under-discriminating configurations
TILBasic_sig_less_discrim_configs = TILBasic_sig_less_discrim_configs[TILBasic_sig_less_discrim_configs.p<.05].groupby(['WINDOW_IDX'],as_index=False).TUNE_IDX.aggregate(list)
highTIL_sig_less_discrim_configs = highTIL_sig_less_discrim_configs[highTIL_sig_less_discrim_configs.p<.05].groupby(['WINDOW_IDX'],as_index=False).TUNE_IDX.aggregate(list)

# Drop out TILBasic configurations that are consistently under-discriminating
TILBasic_flattened_ORC_TIs = [item for sublist in TILBasic_sig_less_discrim_configs.TUNE_IDX[TILBasic_sig_less_discrim_configs.WINDOW_IDX<=6] for item in sublist] 
TILBasic_ORC_counts_of_removal = OrderedDict(Counter(TILBasic_flattened_ORC_TIs).most_common())
TILBasic_ORC_tune_idx_to_remove = [k for (k,v) in TILBasic_ORC_counts_of_removal.items() if v >= 5]
TILBasic_ORC_tune_idx_to_keep = [k for k in TILBasic_compiled_val_orc.TUNE_IDX.unique() if k not in TILBasic_ORC_tune_idx_to_remove]

# Drop out highTIL configurations that are consistently under-discriminating
highTIL_flattened_AUC_TIs = [item for sublist in highTIL_sig_less_discrim_configs.TUNE_IDX[highTIL_sig_less_discrim_configs.WINDOW_IDX<=6] for item in sublist] 
highTIL_AUC_counts_of_removal = OrderedDict(Counter(highTIL_flattened_AUC_TIs).most_common())
highTIL_AUC_tune_idx_to_remove = [k for (k,v) in highTIL_AUC_counts_of_removal.items() if v >= 3]
highTIL_AUC_tune_idx_to_keep = [k for k in highTIL_compiled_val_auc.TUNE_IDX.unique() if k not in highTIL_AUC_tune_idx_to_remove]

## Determine which configurations remain after consideration of calibration and discrimination
# Find the TILBasic configurations which remain after calibration and discrimination check
TILBasic_final_tune_idx_to_keep = list(set(TILBasic_ORC_tune_idx_to_keep) & set(TILBasic_tune_idx_to_keep))
TILBasic_filt_tuning_grid = tuning_grid[tuning_grid.TUNE_IDX.isin(TILBasic_final_tune_idx_to_keep)].reset_index(drop=True)

# Find the highTIL configurations which remain after calibration and discrimination check
highTIL_final_tune_idx_to_keep = list(set(highTIL_AUC_tune_idx_to_keep) & set(highTIL_tune_idx_to_keep))
highTIL_filt_tuning_grid = tuning_grid[tuning_grid.TUNE_IDX.isin(highTIL_final_tune_idx_to_keep)].reset_index(drop=True)

# Merge TILBasic and highTIL post-dropout tuning grids and save
post_dropout_tuning_grid = pd.concat([TILBasic_filt_tuning_grid,highTIL_filt_tuning_grid],ignore_index=True).sort_values(by=['REPEAT','FOLD','TUNE_IDX'],ignore_index=True)
post_dropout_tuning_grid.to_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'),index=False)

# Create separate dataframe designating tuning configurations to be dropped
dropped_tuning_grid = tuning_grid[~tuning_grid.TUNE_IDX.isin(post_dropout_tuning_grid.TUNE_IDX)].reset_index(drop=True)

### IV. Delete folders of underperforming configurations
## From the tuning grid of underperforming configurations, create a list of folders to delete
delete_folders = [os.path.join(model_dir,'repeat'+str(dropped_tuning_grid.REPEAT[curr_row]).zfill(2),'fold'+str(dropped_tuning_grid.FOLD[curr_row]).zfill(1),'tune'+str(dropped_tuning_grid.TUNE_IDX[curr_row]).zfill(4)) for curr_row in range(dropped_tuning_grid.shape[0])]

## Delete folders
for curr_folder in tqdm(delete_folders,"Deleting directories corresponding to underperforming tuning configurations"):
    try:
        shutil.rmtree(curr_folder)
    except:
        pass

### V. Visualise hyperparameter optimisation results
## Load validation set performance dataframes
# TILBasic ORCs
uncalib_TILBasic_val_set_ORCs = pd.read_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_val_uncalibrated_ORCs.csv'))

# High-intensity TIL AUCs
uncalib_highTIL_val_set_AUCs = pd.read_csv(os.path.join(model_perf_dir,'TomorrowHighIntensityTherapy_val_uncalibrated_AUCs.csv'))

# TILBasic calibration metrics
uncalib_TILBasic_val_set_thresh_calibration = pd.read_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_val_uncalibrated_calibration_metrics.csv'))

# High-intensity TIL calibration metrics
uncalib_highTIL_val_set_calibration = pd.read_csv(os.path.join(model_perf_dir,'TomorrowHighIntensityTherapy_val_uncalibrated_calibration_metrics.csv'))

## Prepare validation set results
# Calculate averaged TILBasic ORCs across window indices
ave_uncalib_val_set_ORCs = uncalib_TILBasic_val_set_ORCs.groupby(['TUNE_IDX'],as_index=False).VALUE.mean().rename(columns={'VALUE':'ORC'}).sort_values(by='ORC',ascending=False).reset_index(drop=True)

# Merge hyperparameter information to TILBasic ORC scores
ORC_val_grid = tuning_grid[['TUNE_IDX','WINDOW_LIMIT','RNN_TYPE','LATENT_DIM','HIDDEN_DIM','MIN_BASE_TOKEN_REPRESENATION','MAX_TOKENS_PER_BASE_TOKEN']].drop_duplicates(ignore_index=True).merge(ave_uncalib_val_set_ORCs,how='right')

# Calculate averaged highTIL AUCs across window indices
ave_uncalib_val_set_AUCs = uncalib_highTIL_val_set_AUCs.groupby(['TUNE_IDX'],as_index=False).VALUE.mean().rename(columns={'VALUE':'AUC'}).sort_values(by='AUC',ascending=False).reset_index(drop=True)

# Merge hyperparameter information to highTIL AUC scores
AUC_val_grid = tuning_grid[['TUNE_IDX','WINDOW_LIMIT','RNN_TYPE','LATENT_DIM','HIDDEN_DIM','MIN_BASE_TOKEN_REPRESENATION','MAX_TOKENS_PER_BASE_TOKEN']].drop_duplicates(ignore_index=True).merge(ave_uncalib_val_set_AUCs,how='right')

# Calculate averaged TILBasic calibration metrics across window indices
ave_uncalib_val_set_thresh_calibration = uncalib_TILBasic_val_set_thresh_calibration[(uncalib_TILBasic_val_set_thresh_calibration.THRESHOLD=='Average')&(uncalib_TILBasic_val_set_thresh_calibration.METRIC=='CALIB_SLOPE')].groupby(['TUNE_IDX','WINDOW_IDX'],as_index=False).VALUE.mean().rename(columns={'VALUE':'CALIB_SLOPE'})
ave_uncalib_val_set_thresh_calibration['ERROR'] = (ave_uncalib_val_set_thresh_calibration.CALIB_SLOPE - 1).abs()
ave_uncalib_val_set_thresh_calibration = ave_uncalib_val_set_thresh_calibration.groupby(['TUNE_IDX'],as_index=False).ERROR.mean().sort_values(by='ERROR',ascending=True).reset_index(drop=True)

# Merge hyperparameter information to TILBasic calibration metrics
thresh_calibration_val_grid = tuning_grid[['TUNE_IDX','WINDOW_LIMIT','RNN_TYPE','LATENT_DIM','HIDDEN_DIM','MIN_BASE_TOKEN_REPRESENATION','MAX_TOKENS_PER_BASE_TOKEN']].drop_duplicates(ignore_index=True).merge(ave_uncalib_val_set_thresh_calibration,how='right')

# Calculate averaged highTIL calibration metrics across window indices
ave_uncalib_val_set_binary_calibration = uncalib_highTIL_val_set_calibration[(uncalib_highTIL_val_set_calibration.METRIC=='CALIB_SLOPE')].groupby(['TUNE_IDX','WINDOW_IDX'],as_index=False).VALUE.mean().rename(columns={'VALUE':'CALIB_SLOPE'})
ave_uncalib_val_set_binary_calibration['ERROR'] = (ave_uncalib_val_set_binary_calibration.CALIB_SLOPE - 1).abs()
ave_uncalib_val_set_binary_calibration = ave_uncalib_val_set_binary_calibration.groupby(['TUNE_IDX'],as_index=False).ERROR.mean().sort_values(by='ERROR',ascending=True).reset_index(drop=True)

# Merge hyperparameter information to highTIL calibration metrics
binary_calibration_val_grid = tuning_grid[['TUNE_IDX','WINDOW_LIMIT','RNN_TYPE','LATENT_DIM','HIDDEN_DIM','MIN_BASE_TOKEN_REPRESENATION','MAX_TOKENS_PER_BASE_TOKEN']].drop_duplicates(ignore_index=True).merge(ave_uncalib_val_set_binary_calibration,how='right')

## Generate and save high-dimensional hyperparameter parallel plots
# TILBasic ORC
TILBasic_ORC_hiplot = hip.Experiment.from_dataframe(ORC_val_grid)
TILBasic_ORC_hiplot.colorby = 'ORC'
TILBasic_ORC_hiplot.to_html(os.path.join(model_perf_dir,'ORC_hiplot.html'))

# TILBasic calibration metrics
TILBasic_thresh_hiplot = hip.Experiment.from_dataframe(thresh_calibration_val_grid)
TILBasic_thresh_hiplot.colorby = 'ERROR'
TILBasic_thresh_hiplot.to_html(os.path.join(model_perf_dir,'thresh_calibration_hiplot.html'))

# highTIL AUC
highTIL_AUC_hiplot = hip.Experiment.from_dataframe(AUC_val_grid)
highTIL_AUC_hiplot.colorby = 'AUC'
highTIL_AUC_hiplot.to_html(os.path.join(model_perf_dir,'AUC_hiplot.html'))

# highTIL calibration metrics
highTIL_thresh_hiplot = hip.Experiment.from_dataframe(binary_calibration_val_grid)
highTIL_thresh_hiplot.colorby = 'ERROR'
highTIL_thresh_hiplot.to_html(os.path.join(model_perf_dir,'binary_calibration_hiplot.html'))