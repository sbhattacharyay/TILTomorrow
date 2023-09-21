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
# V. Create bootstrapping resamples for calculating testing set performance

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

import hiplot as hip

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

# Define subdirectory to store validation set bootstrapping results
val_bs_dir = os.path.join(model_perf_dir,'validation_set_bootstrapping')

## Load fundamental information for model training
# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

# Load the optimised tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

### II. Compile and save bootstrapped validation set performance dataframes
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

# Iterate through performance metric files and delete
_ = [os.remove(f) for f in tqdm(perf_file_info_df.FILE,'Clearing validation bootstrapping metric files after collection')]

### III. Dropout configurations based on validation set calibration and discrimination information
## Identify configurations with calibration slope 1 in confidence interval
# Calculate confidence intervals for each tuning configuration
TILBasic_val_CI_calibration_slope = TILBasic_compiled_val_calibration[TILBasic_compiled_val_calibration.METRIC=='CALIB_SLOPE'].groupby(['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC'],as_index=False)['VALUE'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'hi':lambda x: np.quantile(x.dropna(),.975),'resamples':'count'}).reset_index(drop=True)
highTIL_val_CI_calibration_slope = highTIL_compiled_val_calibration[highTIL_compiled_val_calibration.METRIC=='CALIB_SLOPE'].groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False)['VALUE'].aggregate({'lo':lambda x: np.quantile(x.dropna(),.025),'hi':lambda x: np.quantile(x.dropna(),.975),'resamples':'count'}).reset_index(drop=True)

# Mark TUNE_IDX/WINDOW_IDX combinations which are significantly calibrated
TILBasic_val_CI_calibration_slope['CALIBRATED'] = ((TILBasic_val_CI_calibration_slope['lo']<=1)&(TILBasic_val_CI_calibration_slope['hi']>=1))
highTIL_val_CI_calibration_slope['CALIBRATED'] = ((highTIL_val_CI_calibration_slope['lo']<=1)&(highTIL_val_CI_calibration_slope['hi']>=1))

# Concatenate tuning indices of significantly calibrated configurations for each window index
val_calibrated_TIs = TILBasic_val_CI_calibration_slope[(TILBasic_val_CI_calibration_slope.CALIBRATED)&(TILBasic_val_CI_calibration_slope.THRESHOLD=='Average')].groupby(['WINDOW_IDX'],as_index=False).TUNE_IDX.aggregate(list).rename(columns={'TUNE_IDX':'CALIB_TUNE_IDX'})
highTIL_val_calibrated_TIs = highTIL_val_CI_calibration_slope[highTIL_val_CI_calibration_slope.CALIBRATED].groupby(['WINDOW_IDX'],as_index=False).TUNE_IDX.aggregate(list).rename(columns={'TUNE_IDX':'CALIB_TUNE_IDX'})



uncalib_TILBasic_val_set_ORCs = pd.read_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_val_uncalibrated_ORCs.csv'))
uncalib_highTIL_val_set_AUCs = pd.read_csv(os.path.join(model_perf_dir,'TomorrowHighIntensityTherapy_val_uncalibrated_AUCs.csv'))

uncalib_TILBasic_val_set_thresh_calibration = pd.read_csv(os.path.join(model_perf_dir,'TomorrowTILBasic_val_uncalibrated_calibration_metrics.csv'))
uncalib_highTIL_val_set_calibration = pd.read_csv(os.path.join(model_perf_dir,'TomorrowHighIntensityTherapy_val_uncalibrated_calibration_metrics.csv'))

ave_uncalib_val_set_AUCs = uncalib_highTIL_val_set_AUCs.groupby(['TUNE_IDX'],as_index=False).VALUE.mean().rename(columns={'VALUE':'AUC'}).sort_values(by='AUC',ascending=False).reset_index(drop=True)
highTIL_cream_of_crop = ave_uncalib_val_set_AUCs[ave_uncalib_val_set_AUCs.AUC>=0.85].reset_index(drop=True)

chupi = highTIL_val_CI_calibration_slope[highTIL_val_CI_calibration_slope.TUNE_IDX.isin(highTIL_cream_of_crop.TUNE_IDX)].groupby(['TUNE_IDX'],as_index=False).CALIBRATED.sum().merge(highTIL_cream_of_crop).sort_values(by='AUC',ascending=False).reset_index(drop=True)

ave_uncalib_val_set_ORCs = uncalib_TILBasic_val_set_ORCs.groupby(['TUNE_IDX'],as_index=False).VALUE.mean().rename(columns={'VALUE':'ORC'}).sort_values(by='ORC',ascending=False).reset_index(drop=True)
TILBasic_cream_of_crop = ave_uncalib_val_set_ORCs[ave_uncalib_val_set_ORCs.ORC>=0.8].reset_index(drop=True)

flupi = TILBasic_val_CI_calibration_slope[TILBasic_val_CI_calibration_slope.TUNE_IDX.isin(TILBasic_cream_of_crop.TUNE_IDX)&(TILBasic_val_CI_calibration_slope.THRESHOLD=='Average')].groupby(['TUNE_IDX'],as_index=False).CALIBRATED.sum().merge(TILBasic_cream_of_crop).sort_values(by='ORC',ascending=False).reset_index(drop=True)

AUC_val_grid = tuning_grid[['TUNE_IDX','WINDOW_LIMIT','RNN_TYPE','LATENT_DIM','HIDDEN_DIM','MIN_BASE_TOKEN_REPRESENATION','MAX_TOKENS_PER_BASE_TOKEN']].drop_duplicates(ignore_index=True).merge(ave_uncalib_val_set_AUCs,how='right')
ORC_val_grid = tuning_grid[['TUNE_IDX','WINDOW_LIMIT','RNN_TYPE','LATENT_DIM','HIDDEN_DIM','MIN_BASE_TOKEN_REPRESENATION','MAX_TOKENS_PER_BASE_TOKEN']].drop_duplicates(ignore_index=True).merge(ave_uncalib_val_set_ORCs,how='right')

ave_uncalib_val_set_thresh_calibration = uncalib_TILBasic_val_set_thresh_calibration[(uncalib_TILBasic_val_set_thresh_calibration.THRESHOLD=='Average')&(uncalib_TILBasic_val_set_thresh_calibration.METRIC=='CALIB_SLOPE')].groupby(['TUNE_IDX','WINDOW_IDX'],as_index=False).VALUE.mean().rename(columns={'VALUE':'CALIB_SLOPE'})
ave_uncalib_val_set_thresh_calibration['ERROR'] = (ave_uncalib_val_set_thresh_calibration.CALIB_SLOPE - 1).abs()
ave_uncalib_val_set_thresh_calibration = ave_uncalib_val_set_thresh_calibration.groupby(['TUNE_IDX'],as_index=False).ERROR.mean().sort_values(by='ERROR',ascending=True).reset_index(drop=True)
thresh_calibration_val_grid = tuning_grid[['TUNE_IDX','WINDOW_LIMIT','RNN_TYPE','LATENT_DIM','HIDDEN_DIM','MIN_BASE_TOKEN_REPRESENATION','MAX_TOKENS_PER_BASE_TOKEN']].drop_duplicates(ignore_index=True).merge(ave_uncalib_val_set_thresh_calibration,how='right')

ave_uncalib_val_set_binary_calibration = uncalib_highTIL_val_set_calibration[(uncalib_highTIL_val_set_calibration.METRIC=='CALIB_SLOPE')].groupby(['TUNE_IDX','WINDOW_IDX'],as_index=False).VALUE.mean().rename(columns={'VALUE':'CALIB_SLOPE'})
ave_uncalib_val_set_binary_calibration['ERROR'] = (ave_uncalib_val_set_binary_calibration.CALIB_SLOPE - 1).abs()
ave_uncalib_val_set_binary_calibration = ave_uncalib_val_set_binary_calibration.groupby(['TUNE_IDX'],as_index=False).ERROR.mean().sort_values(by='ERROR',ascending=True).reset_index(drop=True)
binary_calibration_val_grid = tuning_grid[['TUNE_IDX','WINDOW_LIMIT','RNN_TYPE','LATENT_DIM','HIDDEN_DIM','MIN_BASE_TOKEN_REPRESENATION','MAX_TOKENS_PER_BASE_TOKEN']].drop_duplicates(ignore_index=True).merge(ave_uncalib_val_set_binary_calibration,how='right')

########### Facebook HiPlot
holla = hip.Experiment.from_dataframe(AUC_val_grid)
holla.colorby = 'AUC'
holla.to_html('../TILTomorrow_model_performance/v1-0/AUC_hiplot.html')

scholla = hip.Experiment.from_dataframe(ORC_val_grid)
scholla.colorby = 'ORC'
scholla.to_html('../TILTomorrow_model_performance/v1-0/ORC_hiplot.html')

chupa = hip.Experiment.from_dataframe(thresh_calibration_val_grid)
chupa.colorby = 'ERROR'
chupa.to_html('../TILTomorrow_model_performance/v1-0/thresh_calibration_hiplot.html')

scupa = hip.Experiment.from_dataframe(binary_calibration_val_grid)
scupa.colorby = 'ERROR'
scupa.to_html('../TILTomorrow_model_performance/v1-0/binary_calibration_hiplot.html')



# Load compiled validation set outputs
uncalib_TILBasic_test_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_test_uncalibrated_outputs.pkl'))
uncalib_highTIL_test_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowHighIntensityTherapy_compiled_test_uncalibrated_outputs.pkl'))

# Calculate intermediate values for TomorrowTILBasic validation set outputs
prob_cols = [col for col in uncalib_TILBasic_test_outputs if col.startswith('Pr(TILBasic=')]
logit_cols = [col for col in uncalib_TILBasic_test_outputs if col.startswith('z_TILBasic=')]
prob_matrix = uncalib_TILBasic_test_outputs[prob_cols]
prob_matrix.columns = list(range(prob_matrix.shape[1]))
index_vector = np.array(list(range(prob_matrix.shape[1])), ndmin=2).T
uncalib_TILBasic_test_outputs['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
uncalib_TILBasic_test_outputs['PredLabel'] = prob_matrix.idxmax(axis=1)

# Calculate intermediate values for TomorrowHighIntensityTherapy validation set outputs
uncalib_highTIL_test_outputs['ExpectedValue'] = uncalib_highTIL_test_outputs['Pr(HighTIL=1)']
uncalib_highTIL_test_outputs['PredLabel'] = (uncalib_highTIL_test_outputs['Pr(HighTIL=1)'] >= 0.5).astype(int)

# Prepare validation set output dataframes for performance calculation
filt_TILBasic_test_outputs = prepare_df(uncalib_TILBasic_test_outputs,PERF_WINDOW_INDICES)
filt_highTIL_test_outputs = prepare_df(uncalib_highTIL_test_outputs,PERF_WINDOW_INDICES)

# Filter further to specific tuning indices
filt_TILBasic_test_outputs = filt_TILBasic_test_outputs[filt_TILBasic_test_outputs.TUNE_IDX.isin([277,171])].reset_index(drop=True)
filt_highTIL_test_outputs = filt_highTIL_test_outputs[filt_highTIL_test_outputs.TUNE_IDX.isin([220,240,136])].reset_index(drop=True)

bs_rs_GUPIs = [resample(filt_TILBasic_test_outputs.GUPI.unique(),replace=True,n_samples=filt_TILBasic_test_outputs.GUPI.nunique()) for _ in range(NUM_RESAMP)]
bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':bs_rs_GUPIs})

ORCs = []
AUCs = []
TILBasic_SomersD = []
highTIL_SomersD = []
TILBasic_calibration = []
highTIL_calibration = []

for curr_rs_idx in tqdm(bs_resamples.RESAMPLE_IDX,'Testing set performance calculation'):
    
    # Extract current bootstrapping resample parameters
    curr_GUPIs = bs_resamples.GUPIs[(bs_resamples.RESAMPLE_IDX==curr_rs_idx).idxmax()]
    
    curr_ORC = calc_ORC(filt_TILBasic_test_outputs[filt_TILBasic_test_outputs.GUPI.isin(curr_GUPIs)].reset_index(drop=True),PERF_WINDOW_INDICES,False)
    curr_AUC = calc_AUC(filt_highTIL_test_outputs[filt_highTIL_test_outputs.GUPI.isin(curr_GUPIs)].reset_index(drop=True),PERF_WINDOW_INDICES,False)
    curr_TILBasic_SomersD = calc_Somers_D(filt_TILBasic_test_outputs[filt_TILBasic_test_outputs.GUPI.isin(curr_GUPIs)].reset_index(drop=True),PERF_WINDOW_INDICES,False)
    curr_highTIL_SomersD = calc_Somers_D(filt_highTIL_test_outputs[filt_highTIL_test_outputs.GUPI.isin(curr_GUPIs)].reset_index(drop=True),PERF_WINDOW_INDICES,False)
    curr_TILBasic_calibration = calc_thresh_calibration(filt_TILBasic_test_outputs[filt_TILBasic_test_outputs.GUPI.isin(curr_GUPIs)].reset_index(drop=True),PERF_WINDOW_INDICES,False)
    curr_highTIL_calibration = calc_binary_calibration(filt_highTIL_test_outputs[filt_highTIL_test_outputs.GUPI.isin(curr_GUPIs)].reset_index(drop=True),PERF_WINDOW_INDICES,False)

    # Add macro-averages to threshold-level calibration metrics
    macro_average_thresh_calibration = curr_TILBasic_calibration.groupby(['TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
    macro_average_thresh_calibration.insert(2,'THRESHOLD',['Average' for idx in range(macro_average_thresh_calibration.shape[0])])
    curr_TILBasic_calibration = pd.concat([curr_TILBasic_calibration,macro_average_thresh_calibration],ignore_index=True).sort_values(by=['TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)

    ORCs.append(curr_ORC)
    AUCs.append(curr_AUC)
    TILBasic_SomersD.append(curr_TILBasic_SomersD)
    highTIL_SomersD.append(curr_highTIL_SomersD)
    TILBasic_calibration.append(curr_TILBasic_calibration)
    highTIL_calibration.append(curr_highTIL_calibration)

ORCs = pd.concat(ORCs,ignore_index=True)
AUCs = pd.concat(AUCs,ignore_index=True)
TILBasic_SomersD = pd.concat(TILBasic_SomersD,ignore_index=True)
highTIL_SomersD = pd.concat(highTIL_SomersD,ignore_index=True)
TILBasic_calibration = pd.concat(TILBasic_calibration,ignore_index=True)
highTIL_calibration = pd.concat(highTIL_calibration,ignore_index=True)

ORCs.to_csv('../TILTomorrow_model_performance/v1-0/test_set_TILBasic_ORCs.csv',index=False)
AUCs.to_csv('../TILTomorrow_model_performance/v1-0/test_set_highTIL_AUCs.csv',index=False)
TILBasic_SomersD.to_csv('../TILTomorrow_model_performance/v1-0/test_set_TILBasic_SomersD.csv',index=False)
highTIL_SomersD.to_csv('../TILTomorrow_model_performance/v1-0/test_set_highTIL_SomersD.csv',index=False)
TILBasic_calibration.to_csv('../TILTomorrow_model_performance/v1-0/test_set_TILBasic_calibration.csv',index=False)
highTIL_calibration.to_csv('../TILTomorrow_model_performance/v1-0/test_set_highTIL_calibration.csv',index=False)






## Identify configurations that are significantly worse than the optimal tuning index
# Load optimal tuning configurations for each window index based on validation set performance
opt_val_calibration_configs = pd.read_csv(os.path.join(model_perf_dir,'optimal_val_set_calibration_configurations.csv')).drop(columns=['CALIB_SLOPE','ERROR']).rename(columns={'TUNE_IDX':'OPT_TUNE_IDX'})

# Add optimal tuning index information to compiled validation set calibration dataframe
compiled_val_calibration = compiled_val_calibration.merge(opt_val_calibration_configs,how='left')

# Identify the optimal error for each WINDOW_IDX/RESAMPLE_IDX combination and merge to compiled dataframe
bs_val_opt_errors = compiled_val_calibration[compiled_val_calibration.TUNE_IDX == compiled_val_calibration.OPT_TUNE_IDX].drop(columns=['THRESHOLD','METRIC','VALUE','TUNE_IDX']).rename(columns={'ERROR':'OPT_ERROR'}).reset_index(drop=True)
compiled_val_calibration = compiled_val_calibration.merge(bs_val_opt_errors,how='left')

# For each WINDOW_IDX/TUNE_IDX combination, calculate the number of times the error is better than the optimal configuration error
compiled_val_calibration['BETTER_THAN_OPT'] = (compiled_val_calibration.ERROR <= compiled_val_calibration.OPT_ERROR)
sig_worse_configs = compiled_val_calibration.groupby(['TUNE_IDX','WINDOW_IDX','THRESHOLD','METRIC','OPT_TUNE_IDX'],as_index=False).BETTER_THAN_OPT.aggregate({'BETTER':'sum','RESAMPLES':'count'}).reset_index(drop=True)
sig_worse_configs['p'] = sig_worse_configs.BETTER/sig_worse_configs.RESAMPLES

# For each WINDOW_IDX, concatenate tuning indices of significantly miscalibrated configurations
sig_worse_configs = sig_worse_configs[sig_worse_configs.p<.05].groupby(['WINDOW_IDX'],as_index=False).TUNE_IDX.aggregate(list)

# Merge information of tuning indices which are significantly calibrated
sig_worse_configs = sig_worse_configs.merge(val_calibrated_TIs,how='left')
sig_worse_configs.TUNE_IDX[~sig_worse_configs.CALIB_TUNE_IDX.isna()] = sig_worse_configs[~sig_worse_configs.CALIB_TUNE_IDX.isna()].apply(lambda x:list(set(x['TUNE_IDX'])-set(x['CALIB_TUNE_IDX'])),axis=1)

## Drop out configurations that are consistently out of range and/or significantly underperforming
flattened_TIs = [item for sublist in sig_worse_configs.TUNE_IDX for item in sublist] 
counts_of_removal = OrderedDict(Counter(flattened_TIs).most_common())
tune_idx_to_remove = [k for (k,v) in counts_of_removal.items() if v >= 80]
tune_idx_to_keep = [k for (k,v) in counts_of_removal.items() if v < 80]

## Identify configurations that are significantly less discriminating than the optimal tuning index
# Load validation set ORCs
validation_set_ORCs = pd.read_csv(os.path.join(model_perf_dir,'val_set_ORCs.csv'))

# For each `WINDOW_IDX`, identify the optimal tuning index based on discrimination
opt_val_discrimination_configs = validation_set_ORCs.loc[validation_set_ORCs.groupby('WINDOW_IDX').VALUE.idxmax()].reset_index(drop=True).drop(columns=['METRIC','VALUE']).rename(columns={'TUNE_IDX':'OPT_TUNE_IDX'})

# Add optimal tuning index information to compiled validation set discrimination dataframe
compiled_val_orc = compiled_val_orc.merge(opt_val_discrimination_configs,how='left')

# Identify the optimal ORC for each WINDOW_IDX/RESAMPLE_IDX combination and merge to compiled dataframe
bs_val_opt_orc = compiled_val_orc[compiled_val_orc.TUNE_IDX == compiled_val_orc.OPT_TUNE_IDX].drop(columns=['METRIC','TUNE_IDX']).rename(columns={'VALUE':'OPT_VALUE'}).reset_index(drop=True)
compiled_val_orc = compiled_val_orc.merge(bs_val_opt_orc,how='left')

# For each WINDOW_IDX/TUNE_IDX combination, calculate the number of times the ORC is better than the optimal configuration ORC
compiled_val_orc['BETTER_THAN_OPT'] = (compiled_val_orc.VALUE >= compiled_val_orc.OPT_VALUE)
sig_less_discrim_configs = compiled_val_orc.groupby(['TUNE_IDX','WINDOW_IDX','METRIC','OPT_TUNE_IDX'],as_index=False).BETTER_THAN_OPT.aggregate({'BETTER':'sum','RESAMPLES':'count'}).reset_index(drop=True)
sig_less_discrim_configs['p'] = sig_less_discrim_configs.BETTER/sig_less_discrim_configs.RESAMPLES

# For each WINDOW_IDX, concatenate tuning indices of significantly under-discriminating configurations
sig_less_discrim_configs = sig_less_discrim_configs[sig_less_discrim_configs.p<.05].groupby(['WINDOW_IDX'],as_index=False).TUNE_IDX.aggregate(list)

## Determine which configurations remain after consideration of calibration and discrimination
# Drop out configurations that are consistently under-discriminating
flattened_ORC_TIs = [item for sublist in sig_less_discrim_configs.TUNE_IDX for item in sublist] 
ORC_counts_of_removal = OrderedDict(Counter(flattened_ORC_TIs).most_common())
ORC_tune_idx_to_remove = [k for (k,v) in ORC_counts_of_removal.items() if v >= 80]
ORC_tune_idx_to_keep = [k for (k,v) in ORC_counts_of_removal.items() if v < 80]

# Find the configurations which remain after calibration and discrimination check
final_tune_idx_to_keep = list(set(ORC_tune_idx_to_keep) & set(tune_idx_to_keep))
filt_tuning_grid = tuning_grid[tuning_grid.TUNE_IDX.isin(final_tune_idx_to_keep)].reset_index(drop=True)
filt_tuning_grid.to_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'),index=False)
dropped_tuning_grid = tuning_grid[~tuning_grid.TUNE_IDX.isin(final_tune_idx_to_keep)].reset_index(drop=True)

### IV. Delete folders of underperforming configurations
## From the tuning grid of underperforming configurations, create a list of folders to delete
delete_folders = [os.path.join(model_dir,'fold'+str(dropped_tuning_grid.FOLD[curr_row]).zfill(1),'tune'+str(dropped_tuning_grid.TUNE_IDX[curr_row]).zfill(4)) for curr_row in range(dropped_tuning_grid.shape[0])]

## Delete folders
for curr_folder in tqdm(delete_folders,"Deleting directories corresponding to underperforming tuning configurations"):
    try:
        shutil.rmtree(curr_folder)
    except:
        pass
    
### V. Create bootstrapping resamples for calculating testing set performance
## Load testing set predictions and optimal configurations
# Load the post-dropout tuning grid
filt_tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))

# Load compiled testing set
test_predictions_df = pd.read_pickle(os.path.join(model_dir,'compiled_test_predictions.pkl'))

# Filter out tuning indices that remain after dropout
test_predictions_df = test_predictions_df[test_predictions_df.TUNE_IDX.isin(filt_tuning_grid.TUNE_IDX)].reset_index(drop=True)

## Create bootstrapping resamples
# Create array of unique testing set GUPIs
uniq_GUPIs = test_predictions_df.GUPI.unique()

# Filter out GUPI-GOSE combinations that are in the testing set
test_GUPI_GOSE = study_GUPI_GOSE[study_GUPI_GOSE.GUPI.isin(uniq_GUPIs)].reset_index(drop=True)

# Make stratified resamples for bootstrapping metrics
bs_rs_GUPIs = [resample(test_GUPI_GOSE.GUPI.values,replace=True,n_samples=test_GUPI_GOSE.shape[0],stratify=test_GUPI_GOSE.GOSE.values) for _ in range(NUM_RESAMP)]
bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resamples 
bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':bs_rs_GUPIs})

# Save bootstrapping resample dataframe
bs_resamples.to_pickle(os.path.join(model_perf_dir,'test_perf_bs_resamples.pkl'))