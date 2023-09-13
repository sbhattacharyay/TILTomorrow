#### Master Script 3b: Extract outputs from TILTomorrow models and prepare for bootstrapping-based dropout ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile and save validation and testing set outputs across partitions
# III. Determine top-performing tuning configurations based on validation set calibration and discrimination
# IV. Create bootstrapping resamples for dropping out poorly calibrated configurations

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

# PyTorch, PyTorch.Text, and Lightning-PyTorch methods
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Custom methods
from models.dynamic_TTM import TILTomorrow_model
from classes.datasets import DYN_ALL_VARIABLE_SET
from functions.model_building import collate_batch, calc_uncalib_outputs, load_model_outputs
from functions.analysis import prepare_df, calc_ORC, calc_AUC, calc_Somers_D, calc_thresh_calibration, calc_binary_calibration

## Define parameters for model training
# Set version code
VERSION = 'v1-0'

# Variable to set whether output files should be cleaned
CLEAN_OUTPUT_FILES = False

# Window indices at which to calculate performance metrics
PERF_WINDOW_INDICES = [1,2,3,4,5,6,9,13,20]

# Number of resamples for validation set bootstrapping
NUM_RESAMP = 1000

## Define and create relevant directories
# Define directory in which tokens are stored for each partition
tokens_dir = '/home/sb2406/rds/hpc-work/tokens'

# Define model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)

# Define and create model performance directory based on version code
model_perf_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_performance',VERSION)
os.makedirs(model_perf_dir,exist_ok=True)

## Load fundamental information for model training
# Load cross-validation splits of study population
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Isolate partitions
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

# Load the optimised tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

# Load prepared token dictionary
full_token_keys = pd.read_excel(os.path.join(tokens_dir,'TILTomorrow_full_token_keys_'+VERSION+'.xlsx'))
full_token_keys.Token = full_token_keys.Token.fillna('')
full_token_keys.BaseToken = full_token_keys.BaseToken.fillna('')

### II. Compile and save validation and testing set outputs across partitions
## Locate and load all model output files
# Search for all output files
pred_files = []
for path in Path(model_dir).rglob('*_predictions.csv'):
    pred_files.append(str(path.resolve()))

# Characterise the output files found
pred_file_info_df = pd.DataFrame({'FILE':pred_files,
                                  'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in pred_files],
                                  'FOLD':[int(re.search('/fold(.*)/tune', curr_file).group(1)) for curr_file in pred_files],
                                  'TUNE_IDX':[int(re.search('/tune(.*)/', curr_file).group(1)) for curr_file in pred_files],
                                  'VERSION':[re.search('_outputs/(.*)/repeat', curr_file).group(1) for curr_file in pred_files],
                                  'CALIBRATION':[re.search('tune(.*)_predictions.csv', curr_file).group(1) for curr_file in pred_files],
                                  'SET':[re.search('calibrated_(.*)_predictions.csv', curr_file).group(1) for curr_file in pred_files]
                                 }).sort_values(by=['REPEAT','FOLD','TUNE_IDX','SET']).reset_index(drop=True)
pred_file_info_df['CALIBRATION'] = pred_file_info_df['CALIBRATION'].str.rsplit(pat='/', n=1).apply(lambda x: x[1])
pred_file_info_df['CALIBRATION'] = pred_file_info_df['CALIBRATION'].str.rsplit(pat='_', n=1).apply(lambda x: x[0])

# Merge outcome label to model output dataframe
pred_file_info_df = pred_file_info_df.merge(tuning_grid[['TUNE_IDX','OUTCOME_LABEL']].drop_duplicates(ignore_index=True),how='left')

# Load and compile uncalibrated TomorrowTILBasic validation set outputs
uncalib_TILBasic_val_outputs = load_model_outputs(pred_file_info_df[(pred_file_info_df.CALIBRATION=='uncalibrated')&
                                                                    (pred_file_info_df.SET=='val')&
                                                                    (pred_file_info_df.OUTCOME_LABEL=='TomorrowTILBasic')].reset_index(drop=True),
                                                  True,
                                                  'Loading uncalibrated TomorrowTILBasic validation set outputs').sort_values(by=['REPEAT','FOLD','TUNE_IDX','GUPI']).reset_index(drop=True)

# Load and compile uncalibrated TomorrowTILBasic testing set outputs
uncalib_TILBasic_test_outputs = load_model_outputs(pred_file_info_df[(pred_file_info_df.CALIBRATION=='uncalibrated')&
                                                                     (pred_file_info_df.SET=='test')&
                                                                     (pred_file_info_df.OUTCOME_LABEL=='TomorrowTILBasic')].reset_index(drop=True),
                                                   True,
                                                   'Loading uncalibrated TomorrowTILBasic testing set outputs').sort_values(by=['REPEAT','FOLD','TUNE_IDX','GUPI']).reset_index(drop=True)

# Load and compile uncalibrated TomorrowHighIntensityTherapy validation set outputs
uncalib_highTIL_val_outputs = load_model_outputs(pred_file_info_df[(pred_file_info_df.CALIBRATION=='uncalibrated')&
                                                                   (pred_file_info_df.SET=='val')&
                                                                   (pred_file_info_df.OUTCOME_LABEL=='TomorrowHighIntensityTherapy')].reset_index(drop=True),
                                                 True,
                                                 'Loading uncalibrated TomorrowHighIntensityTherapy validation set outputs').sort_values(by=['REPEAT','FOLD','TUNE_IDX','GUPI']).reset_index(drop=True)

# Load and compile uncalibrated TomorrowHighIntensityTherapy testing set outputs
uncalib_highTIL_test_outputs = load_model_outputs(pred_file_info_df[(pred_file_info_df.CALIBRATION=='uncalibrated')&
                                                                    (pred_file_info_df.SET=='test')&
                                                                    (pred_file_info_df.OUTCOME_LABEL=='TomorrowHighIntensityTherapy')].reset_index(drop=True),
                                                  True,
                                                  'Loading uncalibrated TomorrowHighIntensityTherapy testing set outputs').sort_values(by=['REPEAT','FOLD','TUNE_IDX','GUPI']).reset_index(drop=True)

## Save validation and testing model output files
# Save compiled uncalibrated TomorrowTILBasic validation set outputs
uncalib_TILBasic_val_outputs.to_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_val_uncalibrated_outputs.pkl'))

# Save compiled uncalibrated TomorrowTILBasic testing set outputs
uncalib_TILBasic_test_outputs.to_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_test_uncalibrated_outputs.pkl'))

# Save compiled uncalibrated TomorrowHighIntensityTherapy validation set outputs
uncalib_highTIL_val_outputs.to_pickle(os.path.join(model_dir,'TomorrowHighIntensityTherapy_compiled_val_uncalibrated_outputs.pkl'))

# Save compiled uncalibrated TomorrowHighIntensityTherapy testing set outputs
uncalib_highTIL_test_outputs.to_pickle(os.path.join(model_dir,'TomorrowHighIntensityTherapy_compiled_test_uncalibrated_outputs.pkl'))

## If `CLEAN_OUTPUT_FILES` set to True, then remove individual output files after compiling
if CLEAN_OUTPUT_FILES:
    
    # Iterate and delete files
    _ = [os.remove(f) for f in tqdm(ckpts_to_drop.file,'Deleting validation and testing set model output files after compilation')]

### III. Determine top-performing tuning configurations based on validation set calibration and discrimination
## Load and prepare compiled validation set outputs
# Load compiled uncalibrated TomorrowTILBasic validation set outputs
uncalib_TILBasic_val_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_val_uncalibrated_outputs.pkl'))

# Load compiled uncalibrated TomorrowHighIntensityTherapy validation set outputs
uncalib_highTIL_val_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowHighIntensityTherapy_compiled_val_uncalibrated_outputs.pkl'))

# Calculate intermediate values for TomorrowTILBasic validation set outputs
prob_cols = [col for col in uncalib_TILBasic_val_outputs if col.startswith('Pr(TILBasic=')]
logit_cols = [col for col in uncalib_TILBasic_val_outputs if col.startswith('z_TILBasic=')]
prob_matrix = uncalib_TILBasic_val_outputs[prob_cols]
prob_matrix.columns = list(range(prob_matrix.shape[1]))
index_vector = np.array(list(range(prob_matrix.shape[1])), ndmin=2).T
uncalib_TILBasic_val_outputs['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
uncalib_TILBasic_val_outputs['PredLabel'] = prob_matrix.idxmax(axis=1)

# Calculate intermediate values for TomorrowHighIntensityTherapy validation set outputs
uncalib_highTIL_val_outputs['ExpectedValue'] = uncalib_highTIL_val_outputs['Pr(HighTIL=1)']
uncalib_highTIL_val_outputs['PredLabel'] = (uncalib_highTIL_val_outputs['Pr(HighTIL=1)'] >= 0.5).astype(int)

# Prepare validation set output dataframes for performance calculation
filt_TILBasic_val_outputs = prepare_df(uncalib_TILBasic_val_outputs,PERF_WINDOW_INDICES)
filt_highTIL_val_outputs = prepare_df(uncalib_highTIL_val_outputs,PERF_WINDOW_INDICES)

## Calculate performance metrics on validation set outputs
# Calculate ORCs of TIL-Basic model on validation set outputs
uncalib_TILBasic_val_set_ORCs = calc_ORC(filt_TILBasic_val_outputs,PERF_WINDOW_INDICES,True,'Calculating validation set ORC')

# Calculate AUCs of high TIL intensity model on validation set outputs
uncalib_highTIL_val_set_AUCs = calc_AUC(filt_highTIL_val_outputs,PERF_WINDOW_INDICES,True,'Calculating validation set AUC')

# Calculate Somers' D of TIL-Basic model on validation set outputs
uncalib_TILBasic_val_set_Somers_D = calc_Somers_D(filt_TILBasic_val_outputs,PERF_WINDOW_INDICES,True,'Calculating validation set Somers D')

# Calculate Somers' D of high TIL intensity model on validation set outputs
uncalib_highTIL_val_set_Somers_D = calc_Somers_D(filt_highTIL_val_outputs,PERF_WINDOW_INDICES,True,'Calculating validation set Somers D')

# Calculate threshold-level calibration metrics of TIL-Basic model on validation set outputs
uncalib_TILBasic_val_set_thresh_calibration = calc_thresh_calibration(filt_TILBasic_val_outputs,PERF_WINDOW_INDICES,True,'Calculating validation set threshold-level calibration metrics')

# Add macro-averages to threshold-level calibration metrics
macro_average_thresh_calibration = uncalib_TILBasic_val_set_thresh_calibration.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(macro_average_thresh_calibration.shape[0])])
uncalib_TILBasic_val_set_thresh_calibration = pd.concat([uncalib_TILBasic_val_set_thresh_calibration,macro_average_thresh_calibration],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)

# Calculate binary calibration metrics of high TIL intensity model on validation set outputs
uncalib_highTIL_val_set_calibration = calc_binary_calibration(filt_highTIL_val_outputs,PERF_WINDOW_INDICES,True,'Calculating validation set binary calibration metrics')

## Save performance metrics from validation outputs
# Save ORCs of TIL-Basic model from validation set outputs
uncalib_TILBasic_val_set_ORCs.to_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_val_uncalibrated_ORCs.csv'),index=False)

# Save AUCs of high TIL intensity model from validation set outputs
uncalib_highTIL_val_set_AUCs.to_csv(os.path.join(model_perf_dir,'TomorrowHighIntensityTherapy_val_uncalibrated_AUCs.csv'),index=False)

# Save Somers' D of TIL-Basic model from validation set outputs
uncalib_TILBasic_val_set_Somers_D.to_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_val_uncalibrated_Somers_D.csv'),index=False)

# Save Somers' D of high TIL intensity model from validation set outputs
uncalib_highTIL_val_set_Somers_D.to_csv(os.path.join(model_perf_dir,'TomorrowHighIntensityTherapy_val_uncalibrated_Somers_D.csv'),index=False)

# Save threshold-level calibration metrics of TIL-Basic model from validation set outputs
uncalib_TILBasic_val_set_thresh_calibration.to_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_val_uncalibrated_calibration_metrics.csv'),index=False)

# Save binary calibration metrics of high TIL intensity model from validation set outputs
uncalib_highTIL_val_set_calibration.to_csv(os.path.join(model_perf_dir,'TomorrowHighIntensityTherapy_val_uncalibrated_calibration_metrics.csv'),index=False)

## Determine optimally performing tuning configurations based on average performance metrics
# Calculate average ORC for each tuning index
ave_uncalib_val_set_ORCs = uncalib_TILBasic_val_set_ORCs.groupby(['TUNE_IDX','WINDOW_IDX'],as_index=False).VALUE.mean().rename(columns={'VALUE':'ORC'}).sort_values(by='ORC',ascending=False).reset_index(drop=True)

# For each `WINDOW_IDX`, identify the optimal tuning index
TILBasic_opt_val_ORC_configs = ave_uncalib_val_set_ORCs.loc[ave_uncalib_val_set_ORCs.groupby('WINDOW_IDX').ORC.idxmax()].reset_index(drop=True)
TILBasic_opt_val_ORC_configs.to_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_optimal_val_set_discrimination_configurations.csv'),index=False)

# Calculate average AUC for each tuning index
ave_uncalib_val_set_AUCs = uncalib_highTIL_val_set_AUCs.groupby(['TUNE_IDX','WINDOW_IDX'],as_index=False).VALUE.mean().rename(columns={'VALUE':'AUC'}).sort_values(by='AUC',ascending=False).reset_index(drop=True)

# For each `WINDOW_IDX`, identify the optimal tuning index
highTIL_opt_val_AUC_configs = ave_uncalib_val_set_AUCs.loc[ave_uncalib_val_set_AUCs.groupby('WINDOW_IDX').AUC.idxmax()].reset_index(drop=True)
highTIL_opt_val_AUC_configs.to_csv(os.path.join(model_perf_dir,'TomorrowHighIntensityTherapy_optimal_val_set_discrimination_configurations.csv'),index=False)

# Calculate average threshold-level calibration slopes for each tuning index
ave_uncalib_val_set_thresh_calibration = uncalib_TILBasic_val_set_thresh_calibration[(uncalib_TILBasic_val_set_thresh_calibration.THRESHOLD=='Average')&(uncalib_TILBasic_val_set_thresh_calibration.METRIC=='CALIB_SLOPE')].groupby(['TUNE_IDX','WINDOW_IDX'],as_index=False).VALUE.mean().rename(columns={'VALUE':'CALIB_SLOPE'})
ave_uncalib_val_set_thresh_calibration['ERROR'] = (ave_uncalib_val_set_thresh_calibration.CALIB_SLOPE - 1).abs()
ave_uncalib_val_set_thresh_calibration = ave_uncalib_val_set_thresh_calibration.sort_values(by='ERROR',ascending=True).reset_index(drop=True)

# For each `WINDOW_IDX`, identify the optimal tuning index
TILBasic_opt_val_calibration_configs = ave_uncalib_val_set_thresh_calibration.loc[ave_uncalib_val_set_thresh_calibration.groupby('WINDOW_IDX').ERROR.idxmin()].reset_index(drop=True)
TILBasic_opt_val_calibration_configs.to_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_optimal_val_set_calibration_configurations.csv'),index=False)

# Calculate average binary calibration slope for each tuning index
ave_uncalib_val_set_binary_calibration = uncalib_highTIL_val_set_calibration[(uncalib_highTIL_val_set_calibration.METRIC=='CALIB_SLOPE')].groupby(['TUNE_IDX','WINDOW_IDX'],as_index=False).VALUE.mean().rename(columns={'VALUE':'CALIB_SLOPE'})
ave_uncalib_val_set_binary_calibration['ERROR'] = (ave_uncalib_val_set_binary_calibration.CALIB_SLOPE - 1).abs()
ave_uncalib_val_set_binary_calibration = ave_uncalib_val_set_binary_calibration.sort_values(by='ERROR',ascending=True).reset_index(drop=True)

# For each `WINDOW_IDX`, identify the optimal tuning index
highTIL_opt_val_calibration_configs = ave_uncalib_val_set_binary_calibration.loc[ave_uncalib_val_set_binary_calibration.groupby('WINDOW_IDX').ERROR.idxmin()].reset_index(drop=True)
highTIL_opt_val_calibration_configs.to_csv(os.path.join(model_perf_dir,'TomorrowHighIntensityTherapy_optimal_val_set_calibration_configurations.csv'),index=False)

### IV. Create bootstrapping resamples for dropping out poorly calibrated configurations
## Load and prepare validation set outputs
# Load compiled validation set outputs
uncalib_TILBasic_val_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_val_uncalibrated_outputs.pkl'))
uncalib_highTIL_val_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowHighIntensityTherapy_compiled_val_uncalibrated_outputs.pkl'))

# Preare compiled validation set outputs to desired performance window indices
filt_TILBasic_val_outputs = prepare_df(uncalib_TILBasic_val_outputs,PERF_WINDOW_INDICES)
filt_highTIL_val_outputs = prepare_df(uncalib_highTIL_val_outputs,PERF_WINDOW_INDICES)

## Create bootstrapping resamples
# Create array of unique validation set GUPIs
uniq_GUPIs = filt_TILBasic_val_outputs.GUPI.unique()

# Make stratified resamples for bootstrapping metrics
bs_rs_GUPIs = [resample(uniq_GUPIs,replace=True,n_samples=len(uniq_GUPIs)) for _ in range(NUM_RESAMP)]
bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':bs_rs_GUPIs})

# Save bootstrapping resample dataframe
bs_resamples.to_pickle(os.path.join(model_perf_dir,'val_dropout_bs_resamples.pkl'))