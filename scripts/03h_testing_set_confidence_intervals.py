#### Master Script 03h: Compile testing set performance results for statistical inference of TILTomorrow models ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile and save bootstrapped testing set performance dataframes
# III. Calculate 95% confidence intervals on test set performance metrics

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

# StatsModel methods
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from statsmodels.nonparametric.smoothers_lowess import lowess

# Custom methods
from functions.analysis import prepare_df, calc_ORC, calc_AUC, calc_Somers_D, calc_thresh_calibration, calc_binary_calibration

## Define parameters for model training
# Set version code
VERSION = 'v1-0'

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Set number of resamples for bootstrapping-based testing set performance
NUM_RESAMP = 1000

# Window indices at which to calculate performance metrics
PERF_WINDOW_INDICES = [1,2,3,4,5,6,9,13,20]

## Define and create relevant directories
# Define model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)

# Define model performance directory based on version code
model_perf_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_performance',VERSION)

# Define subdirectory to store testing set bootstrapping results
test_bs_dir = os.path.join(model_perf_dir,'testing_set_bootstrapping')

## Load fundamental information for model training
# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

# Load the optimised tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))

### II. Compile and save bootstrapped testing set performance dataframes
## Find and characterise all testing set performance files
# Search for all performance files
perf_files = []
for path in Path(test_bs_dir).rglob('TomorrowTILBasic_test_*'):
    perf_files.append(str(path.resolve()))

# Characterise the performance files found
perf_file_info_df = pd.DataFrame({'FILE':perf_files,
                                  'VERSION':[re.search('_performance/(.*)/testing_set_', curr_file).group(1) for curr_file in perf_files],
                                  'OUTCOME_LABEL':[re.search('_bootstrapping/(.*)_test_', curr_file).group(1) for curr_file in perf_files],
                                  'METRIC':[re.search('test_calibrated_(.*)_rs_', curr_file).group(1) for curr_file in perf_files],
                                  'RESAMPLE_IDX':[int(re.search('_rs_(.*).pkl', curr_file).group(1)) for curr_file in perf_files],
                                 }).sort_values(by=['METRIC','RESAMPLE_IDX']).reset_index(drop=True)

# Separate ORC and calibration file dataframes
orc_file_info_df = perf_file_info_df[perf_file_info_df.METRIC == 'ORCs'].reset_index(drop=True)
somers_d_file_info_df = perf_file_info_df[perf_file_info_df.METRIC == 'Somers_D'].reset_index(drop=True)
thresh_calibration_file_info_df = perf_file_info_df[(perf_file_info_df.METRIC == 'calibration_metrics')].reset_index(drop=True)
calibration_curves_file_info_df = perf_file_info_df[(perf_file_info_df.METRIC == 'calibration_curves')].reset_index(drop=True)

## Load and compile testing set performance dataframes into single files
# Load testing set discrimination and calibration performance dataframes
TILBasic_compiled_test_orc = pd.concat([pd.read_pickle(f) for f in tqdm(orc_file_info_df.FILE,'Load and compile testing set ORC values')],ignore_index=True)
TILBasic_compiled_test_somers_d = pd.concat([pd.read_pickle(f) for f in tqdm(somers_d_file_info_df.FILE,'Load and compile testing set Somers D values')],ignore_index=True)
TILBasic_compiled_test_calibration = pd.concat([pd.read_pickle(f) for f in tqdm(thresh_calibration_file_info_df.FILE,'Load and compile testing set threshold-level calibration metrics')],ignore_index=True)
compiled_test_calibration_curves = pd.concat([pd.read_pickle(f) for f in tqdm(calibration_curves_file_info_df.FILE,'Load and compile testing set threshold-level calibration curves')],ignore_index=True)

# Concatenate dataframes
compiled_test_bootstrapping_metrics = pd.concat([TILBasic_compiled_test_orc,TILBasic_compiled_test_somers_d,TILBasic_compiled_test_calibration],ignore_index=True)

# Replace NaN threshold values with 'None'
compiled_test_bootstrapping_metrics.THRESHOLD = compiled_test_bootstrapping_metrics.THRESHOLD.fillna('None')

# Save compiled testing set performance metrics
compiled_test_bootstrapping_metrics.to_pickle(os.path.join(model_perf_dir,'test_bootstrapping_calibrated_metrics.pkl'))

# Save compiled testing set calibration curves
compiled_test_calibration_curves.to_pickle(os.path.join(model_perf_dir,'test_bootstrapping_calibration_curves.pkl'))

# ## Delete individual files once compiled dataframe has been saved
# # Iterate through performance metric files and delete
# _ = [os.remove(f) for f in tqdm(perf_file_info_df.FILE,'Clearing testing bootstrapping metric files after collection')]

### III. Calculate 95% confidence intervals on test set performance metrics
## Load and prepare compiled testing set bootstrapping metrics
# Compiled calibrated testing set performance metrics 
compiled_test_bootstrapping_metrics = pd.read_pickle(os.path.join(model_perf_dir,'test_bootstrapping_calibrated_metrics.pkl'))

# Compiled calibrated testing set calibration curves
compiled_test_calibration_curves = pd.read_pickle(os.path.join(model_perf_dir,'test_bootstrapping_calibration_curves.pkl'))

## Calculate 95% confidence intervals
# Calibrated testing set performance metrics 
test_CI_metrics = compiled_test_bootstrapping_metrics.groupby(['TUNE_IDX','METRIC','WINDOW_IDX','THRESHOLD'],as_index=False)['VALUE'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'median':lambda x: np.median(x.dropna()),'hi':lambda x: np.quantile(x.dropna(),.975),'mean':lambda x: np.mean(x.dropna()),'std':lambda x: np.std(x.dropna()),'resamples':'count'}).reset_index(drop=True)

# Calibrated testing set calibration curves
test_CI_calib_curves = compiled_test_calibration_curves.groupby(['TUNE_IDX','WINDOW_IDX','THRESHOLD','PREDPROB'],as_index=False)['TRUEPROB'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'median':lambda x: np.median(x.dropna()),'hi':lambda x: np.quantile(x.dropna(),.975),'mean':lambda x: np.mean(x.dropna()),'std':lambda x: np.std(x.dropna()),'resamples':'count'}).reset_index(drop=True)

## Save confidence intervals of both calibration and discrimination metrics
test_CI_metrics.to_csv(os.path.join(model_perf_dir,'test_set_metrics_CI.csv'),index=False)
test_CI_calib_curves.to_csv(os.path.join(model_perf_dir,'test_set_calibration_curves_CI.csv'),index=False)