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

## Calculate effect of calibration methods on calibration slope
# Extract calibration slopes of uncalibrated instances
uncalib_calibration_slopes_df = calibration_slopes_df[calibration_slopes_df.CALIBRATION=='None'].rename(columns={'ERROR':'UNCALIB_ERROR'}).reset_index(drop=True)

# Merge uncalibrated calibration slope errors to those of post-calibration combinations
calibration_slopes_df = calibration_slopes_df[calibration_slopes_df.CALIBRATION!='None'].reset_index(drop=True).merge(uncalib_calibration_slopes_df[['TUNE_IDX','OPTIMIZATION','WINDOW_IDX','THRESHOLD','SET','METRIC','UNCALIB_ERROR']],how='left')

# Calculate change in calibration error caused by calibration method
calibration_slopes_df['CHANGE_ERROR'] = calibration_slopes_df['ERROR'] - calibration_slopes_df['UNCALIB_ERROR']



# Extract uncalibrated metric values
uncalibrated_metrics = ave_calibration_metrics_df[ave_calibration_metrics_df.CALIBRATION=='None'].reset_index(drop=True)






# Average metrics over folds

# Save average calibration metrics
ave_cal_metrics.to_pickle(os.path.join(calibration_dir,'average_metrics.pkl'))

