#### Master Script 04c: Compile testing set performance results for statistical inference of sensitivity analysis TILTomorrow models ####
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

# Define subdirectory to store testing set bootstrapping results for sensitivity analysis
sens_bs_dir = os.path.join(model_perf_dir,'sensitivity_bootstrapping')

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
for path in Path(sens_bs_dir).rglob('sens_analysis_TomorrowTILBasic_test_calibrated_*'):
    perf_files.append(str(path.resolve()))

# Characterise the performance files found
perf_file_info_df = pd.DataFrame({'FILE':perf_files,
                                  'VERSION':[re.search('_performance/(.*)/sensitivity_', curr_file).group(1) for curr_file in perf_files],
                                  'OUTCOME_LABEL':[re.search('sens_analysis_(.*)_test_calibrated_', curr_file).group(1) for curr_file in perf_files],
                                  'METRIC':[re.search('test_calibrated_(.*)_rs_', curr_file).group(1) for curr_file in perf_files],
                                  'RESAMPLE_IDX':[int(re.search('_rs_(.*).pkl', curr_file).group(1)) for curr_file in perf_files],
                                 }).sort_values(by=['METRIC','RESAMPLE_IDX']).reset_index(drop=True)

# Separate scalar metric and calibration curve file dataframes
metric_file_info_df = perf_file_info_df[perf_file_info_df.METRIC == 'metrics'].reset_index(drop=True)
calibration_curves_file_info_df = perf_file_info_df[(perf_file_info_df.METRIC == 'calibration_curves')].reset_index(drop=True)

## Load and compile testing set performance dataframes into single files
# Load testing set discrimination and calibration performance dataframes
compiled_test_bootstrapping_metrics = pd.concat([pd.read_pickle(f) for f in tqdm(metric_file_info_df.FILE,'Load and compile testing set scalar metrics for sensitivity analysis')],ignore_index=True)
compiled_test_calibration_curves = pd.concat([pd.read_pickle(f) for f in tqdm(calibration_curves_file_info_df.FILE,'Load and compile testing set threshold-level calibration curves for sensitivity analysis')],ignore_index=True)

# Save compiled testing set performance metrics
compiled_test_bootstrapping_metrics.to_pickle(os.path.join(model_perf_dir,'sens_analysis_bootstrapping_calibrated_metrics.pkl'))

# Save compiled testing set calibration curves
compiled_test_calibration_curves.to_pickle(os.path.join(model_perf_dir,'sens_analysis_bootstrapping_calibration_curves.pkl'))

## Find and characterise all testing set performance files at points of transition
# Search for all performance files
trans_perf_files = []
for path in Path(sens_bs_dir).rglob('trans_sens_analysis_TomorrowTILBasic_test_calibrated_*'):
    trans_perf_files.append(str(path.resolve()))

# Characterise the performance files found
trans_perf_file_info_df = pd.DataFrame({'FILE':trans_perf_files,
                                        'VERSION':[re.search('_performance/(.*)/sensitivity_', curr_file).group(1) for curr_file in trans_perf_files],
                                        'OUTCOME_LABEL':[re.search('sens_analysis_(.*)_test_calibrated_', curr_file).group(1) for curr_file in trans_perf_files],
                                        'METRIC':[re.search('test_calibrated_(.*)_rs_', curr_file).group(1) for curr_file in trans_perf_files],
                                        'RESAMPLE_IDX':[int(re.search('_rs_(.*).pkl', curr_file).group(1)) for curr_file in trans_perf_files],
                                       }).sort_values(by=['METRIC','RESAMPLE_IDX']).reset_index(drop=True)

# Separate scalar metric and calibration curve file dataframes
trans_metric_file_info_df = trans_perf_file_info_df[trans_perf_file_info_df.METRIC == 'metrics'].reset_index(drop=True)
trans_calibration_curves_file_info_df = trans_perf_file_info_df[(trans_perf_file_info_df.METRIC == 'calibration_curves')].reset_index(drop=True)

## Load and compile testing set performance dataframes into single files
# Load testing set discrimination and calibration performance dataframes
trans_compiled_test_bootstrapping_metrics = pd.concat([pd.read_pickle(f) for f in tqdm(trans_metric_file_info_df.FILE,'Load and compile testing set scalar metrics for sensitivity analysis at points of transition')],ignore_index=True)
trans_compiled_test_calibration_curves = pd.concat([pd.read_pickle(f) for f in tqdm(trans_calibration_curves_file_info_df.FILE,'Load and compile testing set threshold-level calibration curves for sensitivity analysis at points of transition')],ignore_index=True)

# Save compiled testing set performance metrics
trans_compiled_test_bootstrapping_metrics.to_pickle(os.path.join(model_perf_dir,'trans_sens_analysis_bootstrapping_calibrated_metrics.pkl'))

# Save compiled testing set calibration curves
trans_compiled_test_calibration_curves.to_pickle(os.path.join(model_perf_dir,'trans_sens_analysis_bootstrapping_calibration_curves.pkl'))

## Find and characterise all no-information performance files
# Search for all performance files
no_info_files = []
for path in Path(sens_bs_dir).rglob('no_information_TomorrowTILBasic_metrics_rs_*'):
    no_info_files.append(str(path.resolve()))

# Characterise the performance files found
no_info_file_info_df = pd.DataFrame({'FILE':no_info_files,
                                     'VERSION':[re.search('_performance/(.*)/sensitivity_', curr_file).group(1) for curr_file in no_info_files],
                                     'OUTCOME_LABEL':[re.search('no_information_(.*)_metrics_', curr_file).group(1) for curr_file in no_info_files],
                                     'RESAMPLE_IDX':[int(re.search('_rs_(.*).pkl', curr_file).group(1)) for curr_file in no_info_files],
                                    }).sort_values(by=['RESAMPLE_IDX']).reset_index(drop=True)

## Load and compile testing set performance dataframes into single files
# Load testing set discrimination and calibration performance dataframes
compiled_no_info_bootstrapping_metrics = pd.concat([pd.read_pickle(f) for f in tqdm(no_info_file_info_df.FILE,'Load and compile no-information scalar metrics for sensitivity analysis')],ignore_index=True)

# Save compiled testing set performance metrics
compiled_no_info_bootstrapping_metrics.to_pickle(os.path.join(model_perf_dir,'no_information_bootstrapping_metrics.pkl'))

## Find and characterise all no-information performance files at points of transition
# Search for all performance files
trans_no_info_files = []
for path in Path(sens_bs_dir).rglob('trans_no_information_TomorrowTILBasic_metrics_rs_*'):
    trans_no_info_files.append(str(path.resolve()))

# Characterise the performance files found
trans_no_info_file_info_df = pd.DataFrame({'FILE':trans_no_info_files,
                                           'VERSION':[re.search('_performance/(.*)/sensitivity_', curr_file).group(1) for curr_file in trans_no_info_files],
                                           'OUTCOME_LABEL':[re.search('trans_no_information_(.*)_metrics_', curr_file).group(1) for curr_file in trans_no_info_files],
                                           'RESAMPLE_IDX':[int(re.search('_rs_(.*).pkl', curr_file).group(1)) for curr_file in trans_no_info_files],
                                          }).sort_values(by=['RESAMPLE_IDX']).reset_index(drop=True)

## Load and compile testing set performance dataframes into single files
# Load testing set discrimination and calibration performance dataframes
trans_compiled_no_info_bootstrapping_metrics = pd.concat([pd.read_pickle(f) for f in tqdm(trans_no_info_file_info_df.FILE,'Load and compile no-information scalar metrics for sensitivity analysis at points of transition')],ignore_index=True)

# Save compiled testing set performance metrics
trans_compiled_no_info_bootstrapping_metrics.to_pickle(os.path.join(model_perf_dir,'trans_no_information_bootstrapping_metrics.pkl'))


# ## Delete individual files once compiled dataframe has been saved
# # Iterate through performance metric files and delete
# _ = [os.remove(f) for f in tqdm(perf_file_info_df.FILE,'Clearing testing bootstrapping metric files after collection')]

### III. Calculate 95% confidence intervals on test set performance metrics
## Load and prepare compiled testing set bootstrapping metrics
# Compiled calibrated testing set performance metrics 
compiled_test_bootstrapping_metrics = pd.read_pickle(os.path.join(model_perf_dir,'sens_analysis_bootstrapping_calibrated_metrics.pkl'))

# Compiled calibrated testing set calibration curves
compiled_test_calibration_curves = pd.read_pickle(os.path.join(model_perf_dir,'sens_analysis_bootstrapping_calibration_curves.pkl'))

# Compiled calibrated testing set performance metrics at points of transition
trans_compiled_test_bootstrapping_metrics = pd.read_pickle(os.path.join(model_perf_dir,'trans_sens_analysis_bootstrapping_calibrated_metrics.pkl'))

# Compiled calibrated testing set calibration curves at points of transition
trans_compiled_test_calibration_curves = pd.read_pickle(os.path.join(model_perf_dir,'trans_sens_analysis_bootstrapping_calibration_curves.pkl'))

# Compiled no-information performance metrics 
compiled_no_info_bootstrapping_metrics = pd.read_pickle(os.path.join(model_perf_dir,'no_information_bootstrapping_metrics.pkl'))

# Compiled no-information performance metrics at points of transition
trans_compiled_no_info_bootstrapping_metrics = pd.read_pickle(os.path.join(model_perf_dir,'trans_no_information_bootstrapping_metrics.pkl'))

## Calculate 95% confidence intervals
# Calibrated testing set performance metrics 
test_CI_metrics = compiled_test_bootstrapping_metrics.groupby(['TUNE_IDX','SENS_IDX','METRIC','WINDOW_IDX','THRESHOLD'],as_index=False)['VALUE'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'median':lambda x: np.median(x.dropna()),'hi':lambda x: np.quantile(x.dropna(),.975),'mean':lambda x: np.mean(x.dropna()),'std':lambda x: np.std(x.dropna()),'resamples':'count'}).reset_index(drop=True)

# Calibrated testing set performance metric differences
test_CI_metrics_diff = compiled_test_bootstrapping_metrics.groupby(['TUNE_IDX','SENS_IDX','METRIC','WINDOW_IDX','THRESHOLD'],as_index=False)['SENS_DIFFERENCE'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'median':lambda x: np.median(x.dropna()),'hi':lambda x: np.quantile(x.dropna(),.975),'mean':lambda x: np.mean(x.dropna()),'std':lambda x: np.std(x.dropna()),'resamples':'count'}).reset_index(drop=True)

# Calibrated testing set calibration curves
test_CI_calib_curves = compiled_test_calibration_curves.groupby(['TUNE_IDX','SENS_IDX','WINDOW_IDX','THRESHOLD','PREDPROB'],as_index=False)['TRUEPROB'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'median':lambda x: np.median(x.dropna()),'hi':lambda x: np.quantile(x.dropna(),.975),'mean':lambda x: np.mean(x.dropna()),'std':lambda x: np.std(x.dropna()),'resamples':'count'}).reset_index(drop=True)

# Calibrated testing set performance metrics at points of transition
trans_test_CI_metrics = trans_compiled_test_bootstrapping_metrics.groupby(['TUNE_IDX','SENS_IDX','METRIC','WINDOW_IDX','THRESHOLD'],as_index=False)['VALUE'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'median':lambda x: np.median(x.dropna()),'hi':lambda x: np.quantile(x.dropna(),.975),'mean':lambda x: np.mean(x.dropna()),'std':lambda x: np.std(x.dropna()),'resamples':'count'}).reset_index(drop=True)

# Calibrated testing set performance metric differences at points of transition
trans_test_CI_metrics_diff = trans_compiled_test_bootstrapping_metrics.groupby(['TUNE_IDX','SENS_IDX','METRIC','WINDOW_IDX','THRESHOLD'],as_index=False)['SENS_DIFFERENCE'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'median':lambda x: np.median(x.dropna()),'hi':lambda x: np.quantile(x.dropna(),.975),'mean':lambda x: np.mean(x.dropna()),'std':lambda x: np.std(x.dropna()),'resamples':'count'}).reset_index(drop=True)

# Calibrated testing set calibration curves
trans_test_CI_calib_curves = trans_compiled_test_calibration_curves.groupby(['TUNE_IDX','SENS_IDX','WINDOW_IDX','THRESHOLD','PREDPROB'],as_index=False)['TRUEPROB'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'median':lambda x: np.median(x.dropna()),'hi':lambda x: np.quantile(x.dropna(),.975),'mean':lambda x: np.mean(x.dropna()),'std':lambda x: np.std(x.dropna()),'resamples':'count'}).reset_index(drop=True)

# Calibrated testing set performance metrics 
no_info_CI_metrics = compiled_no_info_bootstrapping_metrics.groupby(['TUNE_IDX','METRIC','WINDOW_IDX','THRESHOLD'],as_index=False)['VALUE'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'median':lambda x: np.median(x.dropna()),'hi':lambda x: np.quantile(x.dropna(),.975),'mean':lambda x: np.mean(x.dropna()),'std':lambda x: np.std(x.dropna()),'resamples':'count'}).reset_index(drop=True)

# Calibrated testing set performance metric differences
no_info_CI_metrics_diff = compiled_no_info_bootstrapping_metrics.groupby(['TUNE_IDX','METRIC','WINDOW_IDX','THRESHOLD'],as_index=False)['SENS_DIFFERENCE'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'median':lambda x: np.median(x.dropna()),'hi':lambda x: np.quantile(x.dropna(),.975),'mean':lambda x: np.mean(x.dropna()),'std':lambda x: np.std(x.dropna()),'resamples':'count'}).reset_index(drop=True)

# Calibrated testing set performance metrics at points of transition
trans_no_info_CI_metrics = trans_compiled_no_info_bootstrapping_metrics.groupby(['TUNE_IDX','METRIC','WINDOW_IDX','THRESHOLD'],as_index=False)['VALUE'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'median':lambda x: np.median(x.dropna()),'hi':lambda x: np.quantile(x.dropna(),.975),'mean':lambda x: np.mean(x.dropna()),'std':lambda x: np.std(x.dropna()),'resamples':'count'}).reset_index(drop=True)

# Calibrated testing set performance metric differences at points of transition
trans_no_info_CI_metrics_diff = trans_compiled_no_info_bootstrapping_metrics.groupby(['TUNE_IDX','METRIC','WINDOW_IDX','THRESHOLD'],as_index=False)['SENS_DIFFERENCE'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'median':lambda x: np.median(x.dropna()),'hi':lambda x: np.quantile(x.dropna(),.975),'mean':lambda x: np.mean(x.dropna()),'std':lambda x: np.std(x.dropna()),'resamples':'count'}).reset_index(drop=True)

## Save confidence intervals of both calibration and discrimination metrics
# Metric values
test_CI_metrics.to_csv(os.path.join(model_perf_dir,'sens_analysis_metrics_CI.csv'),index=False)

# Difference values
test_CI_metrics_diff.to_csv(os.path.join(model_perf_dir,'sens_analysis_metrics_diff_CI.csv'),index=False)

# Calibration curves
test_CI_calib_curves.to_csv(os.path.join(model_perf_dir,'sens_analysis_calibration_curves_CI.csv'),index=False)

# Metric values at points of transition
trans_test_CI_metrics.to_csv(os.path.join(model_perf_dir,'trans_sens_analysis_metrics_CI.csv'),index=False)

# Difference values at points of transition
trans_test_CI_metrics_diff.to_csv(os.path.join(model_perf_dir,'trans_sens_analysis_metrics_diff_CI.csv'),index=False)

# Calibration curves at points of transition
trans_test_CI_calib_curves.to_csv(os.path.join(model_perf_dir,'trans_sens_analysis_calibration_curves_CI.csv'),index=False)

# Metric values for no-information outputs
no_info_CI_metrics.to_csv(os.path.join(model_perf_dir,'no_information_metrics_CI.csv'),index=False)

# Difference values for no-information outputs
no_info_CI_metrics_diff.to_csv(os.path.join(model_perf_dir,'no_information_metrics_diff_CI.csv'),index=False)

# Metric values for no-information outputs at points of transition
trans_no_info_CI_metrics.to_csv(os.path.join(model_perf_dir,'trans_no_information_metrics_CI.csv'),index=False)

# Difference values for no-information outputs at points of transition
trans_no_info_CI_metrics_diff.to_csv(os.path.join(model_perf_dir,'trans_no_information_metrics_diff_CI.csv'),index=False)