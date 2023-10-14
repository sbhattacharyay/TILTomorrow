#### Master Script 05d: Compile TimeSHAP values calculated in parallel ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile TimeSHAP values and clean directory
# III. Prepare TimeSHAP values for plotting
# IV. Prepare event TimeSHAP values for plotting
# V. Identify candidate patients for illustrative plotting
# VI. Examine tokens with differing effects across thresholds

### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import copy
import shutil
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
from ast import literal_eval
import matplotlib.pyplot as plt
from collections import Counter
from argparse import ArgumentParser
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# PyTorch, PyTorch.Text, and Lightning-PyTorch methods
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Import TimeSHAP methods
import timeshap.explainer as tsx
import timeshap.plot as tsp
from timeshap.wrappers import TorchModelWrapper
from timeshap.utils import get_avg_score_with_avg_event

# SciKit-Learn methods
from sklearn.utils import resample

# Custom methods
from classes.datasets import DYN_ALL_VARIABLE_SET
from models.dynamic_TTM import TILTomorrow_model, timeshap_TILTomorrow_model
from functions.model_building import collate_batch, df_to_multihot_matrix

## Define parameters for model training
# Set version code
VERSION = 'v2-0'

# Set threshold at which to calculate TimeSHAP values
SHAP_THRESHOLD = 'TILBasic>3'

# Set window indices at which to calculate TimeSHAP values
SHAP_WINDOW_INDICES = [1,2,3,4,5,6,7]

# Set number of maximum array tasks in HPC job
MAX_ARRAY_TASKS = 10000

## Define and create relevant directories
# Define model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)

# Define directory in which tokens are stored
tokens_dir = os.path.join('/home/sb2406/rds/hpc-work','tokens')

# Define a directory for the storage of model interpretation values
interp_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_interpretations',VERSION)

# Define a directory for the storage of TimeSHAP values
shap_dir = os.path.join(interp_dir,'timeSHAP')

# Define a subdirectory for the storage of TimeSHAP values
sub_shap_dir = os.path.join(shap_dir,'parallel_results')

# Define a subdirectory for the storage of missed TimeSHAP timepoints
missed_timepoint_dir = os.path.join(shap_dir,'missed_timepoints')

## Load fundamental information for model training
# Load the current version tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))

# Load cross-validation split information to extract testing resamples
cv_splits = pd.read_csv('../cross_validation_splits.csv')
test_splits = cv_splits[cv_splits.SET == 'test'].reset_index(drop=True)
uniq_GUPIs = test_splits.GUPI.unique()

# Load sensitivty analysis grid
sens_analysis_grid = pd.read_csv(os.path.join(model_dir,'sens_analysis_grid.csv'))
sens_analysis_grid = sens_analysis_grid[['SENS_IDX','DROPOUT_VARS']].drop_duplicates(ignore_index=True)

# Read model checkpoint information dataframes
full_ckpt_info = pd.read_pickle(os.path.join(shap_dir,'full_ckpt_info.pkl'))
sens_ckpt_info = pd.read_pickle(os.path.join(shap_dir,'sens_ckpt_info.pkl'))

# Decode dropout variables in sensitivity analysis model checkpoint dataframe
sens_ckpt_info = sens_ckpt_info.merge(sens_analysis_grid).drop(columns='SENS_IDX')

# Compile model checkpoint information dataframes
ckpt_info = pd.concat([full_ckpt_info,sens_ckpt_info],ignore_index=True)
ckpt_info.DROPOUT_VARS = ckpt_info.DROPOUT_VARS.fillna('none')

# Load partitioned significant clinical timepoints for allocated TimeSHAP calculation
timeshap_partitions = pd.read_pickle(os.path.join(shap_dir,'timeSHAP_partitions.pkl'))

# Load prepared token dictionary
full_token_keys = pd.read_excel(os.path.join(tokens_dir,'TILTomorrow_full_token_keys_'+VERSION+'.xlsx'))
full_token_keys.Token = full_token_keys.Token.fillna('')
full_token_keys.BaseToken = full_token_keys.BaseToken.fillna('')

### II. Compile TimeSHAP values and clean directory
## Find completed TimeSHAP configurations and log remaining configurations, if any
# Identify TimeSHAP dataframe files in parallel storage directory
tsx_files = []
for path in Path(os.path.join(sub_shap_dir)).rglob('*_timeSHAP_values_partition_idx_*'):
    tsx_files.append(str(path.resolve()))

# Characterise found TimeSHAP dataframe files
tsx_info_df = pd.DataFrame({'FILE':tsx_files,
                            'VERSION':[re.search('model_interpretations/(.*)/timeSHAP', curr_file).group(1) for curr_file in tsx_files],
                            'BASELINE':[re.search('parallel_results/(.*)_thresh_', curr_file).group(1) for curr_file in tsx_files],
                            'TYPE':[re.search('TILBasic_(.*)_timeSHAP_values', curr_file).group(1) for curr_file in tsx_files],
                            'PARTITION_IDX':[int(re.search('partition_idx_(.*).pkl', curr_file).group(1)) for curr_file in tsx_files]
                           }).sort_values(by=['PARTITION_IDX','TYPE','BASELINE']).reset_index(drop=True)

# Identify TimeSHAP significant timepoints that were missed based on stored files
missed_timepoint_files = []
for path in Path(os.path.join(missed_timepoint_dir)).rglob('*_missed_timepoints_partition_idx_*'):
    missed_timepoint_files.append(str(path.resolve()))

# Characterise found missing timepoint dataframe files
missed_info_df = pd.DataFrame({'FILE':missed_timepoint_files,
                               'VERSION':[re.search('model_interpretations/(.*)/timeSHAP', curr_file).group(1) for curr_file in missed_timepoint_files],
                               'BASELINE':[re.search('missed_timepoints/(.*)_missed_', curr_file).group(1) for curr_file in missed_timepoint_files],
                               'PARTITION_IDX':[int(re.search('partition_idx_(.*).pkl', curr_file).group(1)) for curr_file in missed_timepoint_files]
                              }).sort_values(by=['PARTITION_IDX','BASELINE']).reset_index(drop=True)

# Determine partition indices that have not yet been accounted for
full_range = list(range(1,MAX_ARRAY_TASKS+1))
remaining_partition_indices = np.sort(list(set(full_range)-set(tsx_info_df.PARTITION_IDX)-set(missed_info_df.PARTITION_IDX))).tolist()

# Create partitions for TimeSHAP configurations that are unaccounted for
remaining_timeshap_partitions = timeshap_partitions[timeshap_partitions.PARTITION_IDX.isin(remaining_partition_indices)].reset_index(drop=True)

# Save remaining partitions
if not remaining_timeshap_partitions.empty:
    remaining_timeshap_partitions.to_pickle(os.path.join(shap_dir,'remaining_timeSHAP_partitions.pkl'))

## Load, compile, and save missed significant timepoints
# Load missed significant timepoints
compiled_missed_timepoints = pd.concat([pd.read_pickle(f) for f in tqdm(missed_info_df.FILE,'Loading missed TimeSHAP timepoint files')],ignore_index=True).drop_duplicates(ignore_index=True)

# Impute missing threshold values with 'ExpectedValue'
compiled_missed_timepoints['Threshold'] = SHAP_THRESHOLD
    
# Save compiled missed timepoints dataframe into TimeSHAP directory
compiled_missed_timepoints.to_pickle(os.path.join(shap_dir,'missed_timepoints.pkl'))

## Load, compile, and save TimeSHAP values
# Load and concatenate completed feature TimeSHAP dataframes in parallel
compiled_feature_timeSHAP_values = pd.concat([pd.read_pickle(f) for f in tqdm(tsx_info_df.FILE[tsx_info_df.TYPE=='features'],'Loading feature TimeSHAP files')],ignore_index=True)

# Save compiled feature TimeSHAP values dataframe into TimeSHAP directory
compiled_feature_timeSHAP_values.to_pickle(os.path.join(shap_dir,'feature_timeSHAP_values.pkl'))

# Load and concatenate completed event TimeSHAP dataframes in parallel
compiled_event_timeSHAP_values = pd.concat([pd.read_pickle(f) for f in tqdm(tsx_info_df.FILE[tsx_info_df.TYPE=='event'],'Loading event TimeSHAP files')],ignore_index=True)

# Save compiled event TimeSHAP values dataframe into TimeSHAP directory
compiled_event_timeSHAP_values.to_pickle(os.path.join(shap_dir,'event_timeSHAP_values.pkl'))

# ## After compiling and saving values, delete individual files
# # Delete missed timepoint files
# shutil.rmtree(missed_timepoint_dir)

# # Delete TimeSHAP value files
# shutil.rmtree(sub_shap_dir)

### III. Prepare TimeSHAP values for plotting
## Prepare TimeSHAP value dataframe
# Load compiled TimeSHAP values dataframe from TimeSHAP directory
compiled_feature_timeSHAP_values = pd.read_pickle(os.path.join(shap_dir,'feature_timeSHAP_values.pkl')).rename(columns={'Feature':'Token'}).drop(columns=['Random seed','NSamples'])

# Add version number to compiled TimeSHAP value dataframe
compiled_feature_timeSHAP_values['VERSION'] = VERSION

# Average SHAP values per GUPI-WindowIdx combination
summarised_feature_timeSHAP_values = compiled_feature_timeSHAP_values.groupby(['TUNE_IDX','DROPOUT_VARS','BaselineFeatures','GUPI','Threshold','Token','WindowIdx'],as_index=False)['SHAP'].mean()

## Determine "most important" features for visualisation
# Calculate summary statistics of SHAP per Token
token_level_timeSHAP_summaries = summarised_feature_timeSHAP_values.groupby(['TUNE_IDX','DROPOUT_VARS','BaselineFeatures','Threshold','Token'],as_index=False)['SHAP'].aggregate({'median':np.median,'mean':np.mean,'std':np.std,'instances':'count'}).sort_values('median').reset_index(drop=True)

# # Filter out `Tokens` with less than 2 unique patients
# token_level_timeSHAP_summaries = token_level_timeSHAP_summaries[token_level_timeSHAP_summaries.instances >= 2].reset_index(drop=True)

# Calculate overall summary statistics of SHAP per BaseToken
basetoken_timeSHAP_summaries = token_level_timeSHAP_summaries[token_level_timeSHAP_summaries.instances >= 2].groupby(['TUNE_IDX','BaselineFeatures','Baseline','Missing','Threshold','Type','BaseToken'],as_index=False)['median'].aggregate({'std':np.std,'min':np.min,'q1':lambda x: np.quantile(x,.25),'median':np.median,'q3':lambda x: np.quantile(x,.75),'max':np.max,'mean':np.mean, 'variable_values':'count'}).sort_values('std',ascending=False).reset_index(drop=True)

# Calculate total instances per `BaseToken` and merge information to dataframe
basetoken_timeSHAP_summaries = basetoken_timeSHAP_summaries.merge(summarised_feature_timeSHAP_values.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Missing','Threshold','Type','BaseToken'],as_index=False)['GUPI'].aggregate({'total_instances':lambda x: len(np.unique(x))}),how='left',on=['TUNE_IDX','BaselineFeatures','Baseline','Missing','Threshold','Type','BaseToken'])

# # Remove `BaseTokens` with limited patient representation
# basetoken_timeSHAP_summaries = basetoken_timeSHAP_summaries[(basetoken_timeSHAP_summaries.BaseToken.isin(['HighestDailyDose','PmMedicationName']))|(basetoken_timeSHAP_summaries.variable_values<=100)].reset_index(drop=True)

## Extract most impactful missing value tokens
# Filter `BaseToken` summaries to focus on missing value tokens
missing_basetoken_timeSHAP_summaries = basetoken_timeSHAP_summaries[basetoken_timeSHAP_summaries.Missing].sort_values('min').reset_index(drop=True)

# For each TUNE_IDX-Threshold combination, select the bottom 10 `BaseTokens` based on min median token SHAP values
missing_top_min_timeSHAP_basetokens = missing_basetoken_timeSHAP_summaries.loc[missing_basetoken_timeSHAP_summaries.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold'])['min'].head(10).index].reset_index(drop=True)
missing_top_min_timeSHAP_basetokens['RankIdx'] = missing_top_min_timeSHAP_basetokens.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold'])['min'].rank('dense', ascending=False) + 10

# For each TUNE_IDX-Threshold combination, select the top 10 `BaseTokens` based on max median token SHAP values that are not in bottom 10
missing_filt_set = missing_basetoken_timeSHAP_summaries.merge(missing_top_min_timeSHAP_basetokens[['TUNE_IDX','BaselineFeatures','Baseline','Threshold','BaseToken']], on=['TUNE_IDX','BaselineFeatures','Baseline','Threshold','BaseToken'],how='left', indicator=True)
missing_filt_set = missing_filt_set[missing_filt_set['_merge'] == 'left_only'].sort_values('max',ascending=False).drop(columns='_merge').reset_index(drop=True)
missing_top_max_timeSHAP_basetokens = missing_filt_set.loc[missing_filt_set.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold'])['max'].head(10).index].reset_index(drop=True)
missing_top_max_timeSHAP_basetokens['RankIdx'] = missing_top_max_timeSHAP_basetokens.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold'])['max'].rank('dense', ascending=False)

# Combine and filter
missing_min_max_timeSHAP_basetokens = pd.concat([missing_top_max_timeSHAP_basetokens,missing_top_min_timeSHAP_basetokens],ignore_index=True)
missing_filtered_min_max_timeSHAP_values = summarised_feature_timeSHAP_values.merge(missing_min_max_timeSHAP_basetokens[['TUNE_IDX','BaselineFeatures','Baseline','Missing','Threshold', 'BaseToken','RankIdx']],how='inner',on=['TUNE_IDX','BaselineFeatures','Baseline','Missing','Threshold', 'BaseToken']).reset_index(drop=True)
missing_unique_values_per_base_token = missing_filtered_min_max_timeSHAP_values.groupby('BaseToken',as_index=False).Token.aggregate({'unique_values':lambda x: len(np.unique(x))})
missing_filtered_min_max_timeSHAP_values = missing_filtered_min_max_timeSHAP_values.merge(missing_unique_values_per_base_token,how='left')
missing_filtered_min_max_timeSHAP_values['TokenRankIdx'] = missing_filtered_min_max_timeSHAP_values.groupby(['BaseToken'])['Token'].rank('dense', ascending=True)

# Save dataframe as CSV for plotting
missing_filtered_min_max_timeSHAP_values.to_csv(os.path.join(shap_dir,'filtered_plotting_missing_timeSHAP_values.csv'),index=False)

## Extract most impactful tokens per category
# Filter `BaseToken` summaries to remove missing value tokens
nonmissing_basetoken_timeSHAP_summaries = basetoken_timeSHAP_summaries[basetoken_timeSHAP_summaries.Missing==False].sort_values('min').reset_index(drop=True)

# For each TUNE_IDX-Threshold-Type combination, select the bottom 10 `BaseTokens` based on min median token SHAP values
types_top_min_timeSHAP_basetokens = nonmissing_basetoken_timeSHAP_summaries.loc[nonmissing_basetoken_timeSHAP_summaries.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold','Type'])['min'].head(10).index].reset_index(drop=True)
types_top_min_timeSHAP_basetokens['RankIdx'] = types_top_min_timeSHAP_basetokens.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold','Type'])['min'].rank('dense', ascending=False) + 10

# For each TUNE_IDX-Threshold-Type combination, select the top 10 `BaseTokens` based on max median token SHAP values that are not in bottom 10
nonmissing_filt_set = nonmissing_basetoken_timeSHAP_summaries.merge(types_top_min_timeSHAP_basetokens[['TUNE_IDX','BaselineFeatures','Baseline','Threshold','BaseToken','Type']], on=['TUNE_IDX','BaselineFeatures','Baseline','Threshold','BaseToken','Type'],how='left', indicator=True)
nonmissing_filt_set = nonmissing_filt_set[nonmissing_filt_set['_merge'] == 'left_only'].sort_values('max',ascending=False).drop(columns='_merge').reset_index(drop=True)
types_top_max_timeSHAP_basetokens = nonmissing_filt_set.loc[nonmissing_filt_set.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold','Type'])['max'].head(10).index].reset_index(drop=True)
types_top_max_timeSHAP_basetokens['RankIdx'] = types_top_max_timeSHAP_basetokens.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold','Type'])['max'].rank('dense', ascending=False)

# Combine and filter
types_min_max_timeSHAP_basetokens = pd.concat([types_top_max_timeSHAP_basetokens,types_top_min_timeSHAP_basetokens],ignore_index=True)
types_filtered_min_max_timeSHAP_values = summarised_feature_timeSHAP_values.merge(types_min_max_timeSHAP_basetokens[['TUNE_IDX','BaselineFeatures','Baseline','Missing','Threshold','BaseToken','Type','RankIdx']],how='inner',on=['TUNE_IDX','BaselineFeatures','Baseline','Missing','Threshold','BaseToken','Type']).reset_index(drop=True)
types_unique_values_per_base_token = types_filtered_min_max_timeSHAP_values.groupby('BaseToken',as_index=False).Token.aggregate({'unique_values':lambda x: len(np.unique(x))})
types_filtered_min_max_timeSHAP_values = types_filtered_min_max_timeSHAP_values.merge(types_unique_values_per_base_token,how='left')
types_filtered_min_max_timeSHAP_values['TokenRankIdx'] = types_filtered_min_max_timeSHAP_values.groupby(['BaseToken'])['Token'].rank('dense', ascending=True)

# Save dataframe as CSV for plotting
types_filtered_min_max_timeSHAP_values.to_csv(os.path.join(shap_dir,'filtered_plotting_types_timeSHAP_values.csv'),index=False)

## Extract most impactful tokens overall
# Filter `BaseToken` summaries to remove missing value tokens
nonmissing_basetoken_timeSHAP_summaries = basetoken_timeSHAP_summaries[basetoken_timeSHAP_summaries.Missing==False].sort_values('min').reset_index(drop=True)

# For each TUNE_IDX-Threshold combination, select the top 20 `BaseTokens` based on variance across values
top_variance_timeSHAP_basetokens = nonmissing_basetoken_timeSHAP_summaries.loc[nonmissing_basetoken_timeSHAP_summaries.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold'])['std'].head(20).index].reset_index(drop=True)
top_variance_timeSHAP_basetokens['RankIdx'] = top_variance_timeSHAP_basetokens.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold'])['std'].rank('dense', ascending=False)
filtered_top_variance_timeSHAP_values = summarised_feature_timeSHAP_values.merge(top_variance_timeSHAP_basetokens[['TUNE_IDX','BaselineFeatures','Baseline','Threshold', 'BaseToken','RankIdx']],how='inner',on=['TUNE_IDX','BaselineFeatures','Baseline','Threshold','BaseToken']).reset_index(drop=True)

# For each TUNE_IDX-Threshold combination, select the bottom 10 `BaseTokens` based on min median token SHAP values
nonmissing_basetoken_timeSHAP_summaries = nonmissing_basetoken_timeSHAP_summaries.sort_values('min').reset_index(drop=True)
top_min_timeSHAP_basetokens = nonmissing_basetoken_timeSHAP_summaries.loc[nonmissing_basetoken_timeSHAP_summaries.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold'])['min'].head(10).index].reset_index(drop=True)
top_min_timeSHAP_basetokens['RankIdx'] = top_min_timeSHAP_basetokens.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold'])['min'].rank('dense', ascending=False) + 10

# For each TUNE_IDX-Threshold combination, select the top 10 `BaseTokens` based on max median token SHAP values that are not in bottom 10
filt_set = nonmissing_basetoken_timeSHAP_summaries.merge(top_min_timeSHAP_basetokens[['TUNE_IDX','BaselineFeatures','Baseline','Threshold','BaseToken']], on=['TUNE_IDX','BaselineFeatures','Baseline','Threshold','BaseToken'],how='left', indicator=True)
filt_set = filt_set[filt_set['_merge'] == 'left_only'].sort_values('max',ascending=False).drop(columns='_merge').reset_index(drop=True)
top_max_timeSHAP_basetokens = filt_set.loc[filt_set.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold'])['max'].head(10).index].reset_index(drop=True)
top_max_timeSHAP_basetokens['RankIdx'] = top_max_timeSHAP_basetokens.groupby(['TUNE_IDX','BaselineFeatures','Baseline','Threshold'])['max'].rank('dense', ascending=False)

## Isolate TimeSHAP values corresponding to "most important" and save
# Combine and filter
min_max_timeSHAP_basetokens = pd.concat([top_max_timeSHAP_basetokens,top_min_timeSHAP_basetokens],ignore_index=True)
filtered_min_max_timeSHAP_values = summarised_feature_timeSHAP_values.merge(min_max_timeSHAP_basetokens[['TUNE_IDX','BaselineFeatures','Baseline','Threshold','Missing','BaseToken','RankIdx']],how='inner',on=['TUNE_IDX','BaselineFeatures','Baseline','Threshold','Missing','BaseToken']).reset_index(drop=True)
unique_values_per_base_token = filtered_min_max_timeSHAP_values.groupby('BaseToken',as_index=False).Token.aggregate({'unique_values':lambda x: len(np.unique(x))})
filtered_min_max_timeSHAP_values = filtered_min_max_timeSHAP_values.merge(unique_values_per_base_token,how='left')
filtered_min_max_timeSHAP_values['TokenRankIdx'] = filtered_min_max_timeSHAP_values.groupby(['BaseToken'])['Token'].rank('dense', ascending=True)

# Save dataframe as CSV for plotting
filtered_min_max_timeSHAP_values.to_csv(os.path.join(shap_dir,'filtered_plotting_timeSHAP_values.csv'),index=False)

# ## Extract most impactful tokens overall per version-WLST cross
# # Filter `BaseToken` summaries to remove missing value tokens
# nonmissing_version_basetoken_timeSHAP_summaries = version_basetoken_timeSHAP_summaries[version_basetoken_timeSHAP_summaries.Missing==False].sort_values('min').reset_index(drop=True)

# # For each TUNE_IDX-Threshold-Version-WLST combination, select the top 20 `BaseTokens` based on variance across values
# version_top_variance_timeSHAP_basetokens = nonmissing_version_basetoken_timeSHAP_summaries.loc[nonmissing_version_basetoken_timeSHAP_summaries.groupby(['TUNE_IDX','Baseline','Threshold','VERSION','WLST'])['std'].head(20).index].reset_index(drop=True)
# version_top_variance_timeSHAP_basetokens['RankIdx'] = version_top_variance_timeSHAP_basetokens.groupby(['TUNE_IDX','Baseline','Threshold','VERSION','WLST'])['std'].rank('dense', ascending=False)
# version_filtered_top_variance_timeSHAP_values = version_summarised_feature_timeSHAP_values.merge(version_top_variance_timeSHAP_basetokens[['VERSION','WLST','TUNE_IDX','Baseline','Threshold', 'BaseToken','RankIdx']],how='inner',on=['VERSION','WLST','TUNE_IDX','Baseline','Threshold','BaseToken']).reset_index(drop=True)

# # For each TUNE_IDX-Threshold-Version-WLST combination, select the bottom 10 `BaseTokens` based on min median token SHAP values
# nonmissing_version_basetoken_timeSHAP_summaries = nonmissing_version_basetoken_timeSHAP_summaries.sort_values('min').reset_index(drop=True)
# version_top_min_timeSHAP_basetokens = nonmissing_version_basetoken_timeSHAP_summaries.loc[nonmissing_version_basetoken_timeSHAP_summaries.groupby(['VERSION','WLST','TUNE_IDX','Baseline','Threshold'])['min'].head(10).index].reset_index(drop=True)
# version_top_min_timeSHAP_basetokens['RankIdx'] = version_top_min_timeSHAP_basetokens.groupby(['VERSION','WLST','TUNE_IDX','Baseline','Threshold'])['min'].rank('dense', ascending=False) + 10

# # For each TUNE_IDX-Threshold combination, select the top 10 `BaseTokens` based on max median token SHAP values that are not in bottom 10
# version_filt_set = nonmissing_version_basetoken_timeSHAP_summaries.merge(version_top_min_timeSHAP_basetokens[['VERSION','WLST','TUNE_IDX','Baseline','Threshold','BaseToken']], on=['VERSION','WLST','TUNE_IDX','Baseline','Threshold','BaseToken'],how='left', indicator=True)
# version_filt_set = version_filt_set[version_filt_set['_merge'] == 'left_only'].sort_values('max',ascending=False).drop(columns='_merge').reset_index(drop=True)
# version_top_max_timeSHAP_basetokens = version_filt_set.loc[version_filt_set.groupby(['VERSION','WLST','TUNE_IDX','Baseline','Threshold'])['max'].head(10).index].reset_index(drop=True)
# version_top_max_timeSHAP_basetokens['RankIdx'] = version_top_max_timeSHAP_basetokens.groupby(['VERSION','WLST','TUNE_IDX','Baseline','Threshold'])['max'].rank('dense', ascending=False)

# ## Isolate TimeSHAP values corresponding to "most important" and save
# # Combine and filter
# version_min_max_timeSHAP_basetokens = pd.concat([version_top_max_timeSHAP_basetokens,version_top_min_timeSHAP_basetokens],ignore_index=True)
# version_filtered_min_max_timeSHAP_values = version_summarised_feature_timeSHAP_values.merge(version_min_max_timeSHAP_basetokens[['VERSION','WLST','TUNE_IDX','Baseline','Threshold','Missing','BaseToken','RankIdx']],how='inner',on=['VERSION','WLST','TUNE_IDX','Baseline','Threshold','Missing','BaseToken']).reset_index(drop=True)
# version_unique_values_per_base_token = version_filtered_min_max_timeSHAP_values.groupby(['VERSION','BaseToken'],as_index=False).Token.aggregate({'unique_values':lambda x: len(np.unique(x))})
# version_filtered_min_max_timeSHAP_values = version_filtered_min_max_timeSHAP_values.merge(version_unique_values_per_base_token,how='left')
# version_filtered_min_max_timeSHAP_values['TokenRankIdx'] = version_filtered_min_max_timeSHAP_values.groupby(['VERSION','BaseToken'])['Token'].rank('dense', ascending=True)

# # Save dataframe as CSV for plotting
# version_filtered_min_max_timeSHAP_values.to_csv(os.path.join(shap_dir,'filtered_WLST_plotting_timeSHAP_values.csv'),index=False)

### IV. Prepare event TimeSHAP values for plotting
## Prepare event TimeSHAP value dataframe
# Load compiled event TimeSHAP values dataframe from TimeSHAP directory
compiled_event_timeSHAP_values = pd.read_pickle(os.path.join(shap_dir,'event_timeSHAP_values.pkl')).rename(columns={'Feature':'TimePretimepoint'}).drop(columns=['Random seed','NSamples'])

# Remove "pruned events"
compiled_event_timeSHAP_values = compiled_event_timeSHAP_values[compiled_event_timeSHAP_values.TimePretimepoint != 'Pruned Events'].reset_index(drop=True)

# Reformat `TimePretimepoint` into numeric variable
compiled_event_timeSHAP_values['TimePretimepoint'] = compiled_event_timeSHAP_values.TimePretimepoint.str.replace('Event ','').astype('int')

# Store absolute SHAP values in new column
compiled_event_timeSHAP_values['absSHAP'] = compiled_event_timeSHAP_values['SHAP'].abs()

# Average SHAP values per GUPI-`TimePretimepoint` combinations
summarised_event_timeSHAP_values = compiled_event_timeSHAP_values.groupby(['TUNE_IDX','BaselineFeatures','GUPI','Threshold','TimePretimepoint'],as_index=False)['absSHAP'].mean()

# Save dataframe as CSV for plotting
summarised_event_timeSHAP_values.to_csv(os.path.join(shap_dir,'filtered_plotting_event_timeSHAP_values.csv'),index=False)

## Prepare dataframe to determine effect of `WindowIdx` on `PruneIdx`
# Drop duplicates corresponding to each individual "significant" timepoint
trans_prune_combos = compiled_event_timeSHAP_values[['TUNE_IDX','BaselineFeatures','REPEAT','FOLD','GUPI','Threshold','WindowIdx','PruneIdx']].drop_duplicates().sort_values(['TUNE_IDX','BaselineFeatures','REPEAT','FOLD','GUPI','Threshold','WindowIdx','PruneIdx'],ignore_index=True)

# Save dataframe as CSV for observation
trans_prune_combos.to_csv(os.path.join(shap_dir,'timepoint_pruning_combos.csv'),index=False)

### V. Identify candidate patients for illustrative plotting
## Characterise testing set predictions
# Load compiled testing set predictions
test_predictions_df = pd.read_csv(os.path.join(model_dir,'compiled_test_predictions.csv'))

# Extract predictions of ideal tuning configuration index
test_predictions_df = test_predictions_df[test_predictions_df.TUNE_IDX==135].reset_index(drop=True)

# Remove logit columns from dataframe
logit_cols = [col for col in test_predictions_df if col.startswith('z_GOSE=')]
test_predictions_df = test_predictions_df.drop(columns=logit_cols).reset_index(drop=True)

# Extract predicted label for each row based on top GOSE probability
prob_cols = [col for col in test_predictions_df if col.startswith('Pr(GOSE=')]
prob_matrix = test_predictions_df[prob_cols]
prob_matrix.columns = list(range(prob_matrix.shape[1]))
test_predictions_df['PredLabel'] = prob_matrix.idxmax(1)

# Calculate threshold-level probability and label
thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
for thresh in range(1,len(prob_cols)):
    cols_gt = prob_cols[thresh:]
    prob_gt = test_predictions_df[cols_gt].sum(1).values
    gt = (test_predictions_df['TrueLabel'] >= thresh).astype(int).values
    test_predictions_df['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
    test_predictions_df[thresh_labels[thresh-1]] = gt

# Remove intermediate probability columns from dataframe
test_predictions_df = test_predictions_df.drop(columns=prob_cols).reset_index(drop=True)

# Add a new column designating maximum `WindowIdx` per patient
test_predictions_df['WindowTotal'] = test_predictions_df.groupby(['GUPI','TUNE_IDX','REPEAT','FOLD']).WindowIdx.transform('max')

## Determine patients with appropriate prediction trajectories
# Convert threshold-level labels to long form dataframe
thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
long_thresh_labels = test_predictions_df[['GUPI','WindowIdx','WindowTotal','REPEAT','FOLD']+thresh_labels].melt(id_vars=['GUPI','WindowIdx','WindowTotal','REPEAT','FOLD'],var_name='Threshold',value_name='TrueThreshLabel')

# Convert threshold-level predictions to long form dataframe
thresh_prob_cols = [col for col in test_predictions_df if col.startswith('Pr(GOSE>')]
long_thresh_probs = test_predictions_df[['GUPI','WindowIdx','WindowTotal','REPEAT','FOLD']+thresh_prob_cols].melt(id_vars=['GUPI','WindowIdx','WindowTotal','REPEAT','FOLD'],var_name='Threshold',value_name='ThreshProb')
long_thresh_probs['Threshold'] = long_thresh_probs['Threshold'].str.removeprefix('Pr(').str.removesuffix(')')
long_thresh_probs['PredThreshLabel'] = (long_thresh_probs.ThreshProb>=0.5).astype('int')

# Merge the two long-form dataframes
long_thresh_preds = pd.merge(long_thresh_labels,long_thresh_probs,how='left').reset_index(drop=True)

# Isolate instances in which final prediction is correct at each threshold
final_pred_instances = long_thresh_preds[(long_thresh_preds.WindowIdx==long_thresh_preds.WindowTotal)&(long_thresh_preds.TrueThreshLabel==long_thresh_preds.PredThreshLabel)].sort_values(['REPEAT','FOLD','GUPI','Threshold']).reset_index(drop=True)
final_pred_instances['CorrectThreshCount'] = final_pred_instances.groupby(['GUPI','WindowIdx','WindowTotal','REPEAT','FOLD']).Threshold.transform('count')
final_pred_instances = final_pred_instances[final_pred_instances.CorrectThreshCount==6].reset_index(drop=True)
correct_end_pred_instances = final_pred_instances[['GUPI','WindowTotal','REPEAT','FOLD']].drop_duplicates(ignore_index=True)

# Isolate instances with prediction change, but correct prediction resolution at the end
cand_preds = long_thresh_preds.merge(correct_end_pred_instances,how='inner')
cand_preds['CorrectPred'] = (cand_preds.PredThreshLabel == cand_preds.TrueThreshLabel).astype('int')
cand_preds['PropCorrect'] = cand_preds[(cand_preds.WindowIdx>=4)&(cand_preds.WindowIdx<=84)].groupby(['GUPI','WindowTotal','REPEAT','FOLD']).CorrectPred.transform('mean')
cand_preds = cand_preds[(cand_preds.PropCorrect!=1)&(~cand_preds.PropCorrect.isna())].drop(columns='PropCorrect').reset_index(drop=True)

# Add true labels and filter patients with intermediate outcomes
cand_preds = cand_preds.merge(test_predictions_df[['GUPI','TrueLabel']].drop_duplicates(ignore_index=True),how='left')
# cand_preds = cand_preds[~cand_preds.TrueLabel.isin([0,6])].reset_index(drop=True)

## Identify suitable significant timepoints for exploration
# Filter TimeSHAP timepoint dataframe to candidate predictions
cand_tsx_partitions = timeshap_partitions.merge(cand_preds[['REPEAT','FOLD','GUPI']].drop_duplicates(ignore_index=True),how='inner')

# Add true label information to candidate timepoint dataframe
cand_tsx_partitions = cand_tsx_partitions.merge(test_predictions_df[['GUPI','TrueLabel']].drop_duplicates(ignore_index=True),how='left')

# Disregard early significant timepoints
cand_tsx_partitions = cand_tsx_partitions[cand_tsx_partitions.WindowIdx!=4].reset_index(drop=True)

# Focus on significant timepoints at the highest positive thresholds or lowest negative thresholds
cand_tsx_partitions['HighestPositiveThresh'] = cand_tsx_partitions.TrueLabel.apply(lambda x: thresh_labels[max(int(x-1),0)])
cand_tsx_partitions['LowestNegativeThresh'] = cand_tsx_partitions.TrueLabel.apply(lambda x: thresh_labels[min(int(x),5)])
cand_tsx_partitions = cand_tsx_partitions[(cand_tsx_partitions.Threshold==cand_tsx_partitions.HighestPositiveThresh)|(cand_tsx_partitions.Threshold==cand_tsx_partitions.LowestNegativeThresh)].reset_index(drop=True)

# Determine timepoints that were significant across partitions
cand_across_prx = cand_tsx_partitions.groupby(['GUPI','WindowIdx','Threshold','TrueLabel','Above'],as_index=False).Diff.aggregate({'count':'count','mean':'mean'}).sort_values('count',ascending=False)

# From manual inspection of candidate timepoints, create a dataframe of selections for visualisation
man_ispect_selections = pd.DataFrame({'GUPI':['2BWg753','5HZz257','6xrH956','2DLL573','7YeE448','9isg322','5sMQ758'],'REPEAT':[12,1,12,4,15,10,3],'WindowIdx':[43,25,46,22,60,41,33],'TrueLabel':[6,5,4,3,2,1,0]})

# Filtered candidate TimeSHAP partitions
man_tsx_partitions = cand_tsx_partitions[cand_tsx_partitions.GUPI.isin(man_ispect_selections.GUPI)].reset_index(drop=True)

## Filter and prepare testing set predictions and TimeSHAP for selected GUPI
# Filter and save testing set predictions
man_filt_preds = test_predictions_df.merge(man_tsx_partitions[['GUPI','REPEAT']].drop_duplicates(),how='inner').reset_index(drop=True)
man_filt_preds.to_csv(os.path.join(model_dir,'plotting_test_predictions.csv'),index=False)

# Retrieve feature-level TimeSHAP values corresponding to current timepoint of focus
compiled_feature_timeSHAP_values = pd.read_pickle(os.path.join(shap_dir,'timeSHAP_values.pkl')).rename(columns={'Feature':'Token'}).drop(columns=['Random seed','NSamples'])
man_timeSHAP_values = compiled_feature_timeSHAP_values[compiled_feature_timeSHAP_values.Threshold=='ExpectedValue'].merge(man_ispect_selections,how='inner').reset_index(drop=True)
    
# Calculate absolute SHAP values
man_timeSHAP_values['absSHAP'] = man_timeSHAP_values['SHAP'].abs()

# Characterise tokens
man_timeSHAP_values['Baseline'] = man_timeSHAP_values['Token'].str.startswith('Baseline')
man_timeSHAP_values['Numeric'] = man_timeSHAP_values['Token'].str.contains('_BIN')
man_timeSHAP_values['Missing'] = ((man_timeSHAP_values.Numeric)&(man_timeSHAP_values['Token'].str.endswith('_BIN_missing')))|((~man_timeSHAP_values.Numeric)&(man_timeSHAP_values['Token'].str.endswith('_NA')))
man_timeSHAP_values['BaseToken'] = ''
man_timeSHAP_values.BaseToken[man_timeSHAP_values.Numeric] = man_timeSHAP_values.Token[man_timeSHAP_values.Numeric].str.replace('\\_BIN.*','',1,regex=True)
man_timeSHAP_values.BaseToken[~man_timeSHAP_values.Numeric] = man_timeSHAP_values.Token[~man_timeSHAP_values.Numeric].str.replace('_[^_]*$','',1,regex=True)
man_timeSHAP_values.BaseToken[man_timeSHAP_values.Baseline] = man_timeSHAP_values.BaseToken[man_timeSHAP_values.Baseline].str.replace('Baseline','',1,regex=False)
man_timeSHAP_values.BaseToken = man_timeSHAP_values.BaseToken.str.replace('_','')

# Calculate the sum of missing token SHAPs by instance and then remove missing token rows
overall_SHAP_sums = man_timeSHAP_values.groupby(['GUPI','REPEAT','WindowIdx','TrueLabel','Missing','Baseline'],as_index=False).SHAP.sum()
man_timeSHAP_values = man_timeSHAP_values[~man_timeSHAP_values.Missing].reset_index(drop=True)

# Add a ranking value based on Static-Dynamic group
man_timeSHAP_values['RankIdx'] = man_timeSHAP_values.groupby(['GUPI','REPEAT','WindowIdx','TrueLabel','Baseline'],as_index=False)['absSHAP'].rank(method="dense", ascending=False)

# Summarise the non-top token SHAPs and remove
nonmissing_others_SHAP_sum = man_timeSHAP_values[man_timeSHAP_values.RankIdx > 10].groupby(['GUPI','REPEAT','FOLD','TUNE_IDX','WindowIdx','Threshold','BaselineFeatures','PARTITION_IDX','TrueLabel','Baseline','Missing'],as_index=False)['SHAP'].sum()
nonmissing_others_SHAP_sum['Token'] = 'Others'
nonmissing_others_SHAP_sum['RankIdx'] = 11
man_timeSHAP_values = man_timeSHAP_values[man_timeSHAP_values.RankIdx<=10].reset_index(drop=True)
man_timeSHAP_values = pd.concat([man_timeSHAP_values,nonmissing_others_SHAP_sum],ignore_index=True)

# Remove rows that are exactly zero
man_timeSHAP_values = man_timeSHAP_values[man_timeSHAP_values.SHAP!=0].reset_index(drop=True)

# Save top ranking SHAP values
man_timeSHAP_values.to_csv(os.path.join(shap_dir,'individual_plotting_timeSHAP_values.csv'),index=False)

# Retrieve event-level TimeSHAP values corresponding to current timepoint of focus
compiled_event_timeSHAP_values = pd.read_pickle(os.path.join(shap_dir,'event_timeSHAP_values.pkl')).rename(columns={'Feature':'TimePretimepoint'}).drop(columns=['Random seed','NSamples'])
compiled_event_timeSHAP_values['TimePretimepoint'] = compiled_event_timeSHAP_values.TimePretimepoint.str.replace('Event ','')
compiled_event_timeSHAP_values['absSHAP'] = compiled_event_timeSHAP_values['SHAP'].abs()
man_event_timeSHAP_values = compiled_event_timeSHAP_values[compiled_event_timeSHAP_values.Threshold=='ExpectedValue'].merge(man_ispect_selections,how='inner').reset_index(drop=True)

# Save event SHAP values
man_event_timeSHAP_values.to_csv(os.path.join(shap_dir,'individual_plotting_event_timeSHAP_values.csv'),index=False)

### VI. Examine tokens with differing effects across thresholds
## Extract proper timepoints based on differing TimeSHAP directionality at different thresholds
# Load compiled TimeSHAP values dataframe from TimeSHAP directory
compiled_feature_timeSHAP_values = pd.read_pickle(os.path.join(shap_dir,'timeSHAP_values.pkl'))

# Remove unnecessary columns and sort
compiled_feature_timeSHAP_values = compiled_feature_timeSHAP_values[['TUNE_IDX','GUPI','WindowIdx','Token','Threshold','SHAP']].sort_values(by=['TUNE_IDX','GUPI','WindowIdx','Token','Threshold'],ignore_index=True)

# Add a column designating TimeSHAP directionality
compiled_feature_timeSHAP_values['SHAPDirection'] = 'Neutral'
compiled_feature_timeSHAP_values.SHAPDirection[compiled_feature_timeSHAP_values.SHAP>0] = 'Positive'
compiled_feature_timeSHAP_values.SHAPDirection[compiled_feature_timeSHAP_values.SHAP<0] = 'Negative'

# For each token in each significant clinical timepoint, identify the proportion of directionality across the thresholds
token_proportion_directionality = compiled_feature_timeSHAP_values[['TUNE_IDX','GUPI','WindowIdx','Token','SHAPDirection']].groupby(['TUNE_IDX','GUPI','WindowIdx','Token'],as_index=False).value_counts(normalize=True)

# Filter out all token-timepoint instances without threshold-differing effect or with unknown token
diff_token_proportion_directionality = token_proportion_directionality[(token_proportion_directionality.proportion!=1)&(~token_proportion_directionality.Token.str.startswith('<unk>'))].reset_index(drop=True)

# Merge full token key information to dataframe of tokens with threshold-differing effects
diff_token_proportion_directionality = diff_token_proportion_directionality.merge(full_token_keys,how='left')

# For initial purpose, filter out any missing value tokens
diff_token_proportion_directionality = diff_token_proportion_directionality[diff_token_proportion_directionality.Missing==False].reset_index(drop=True)

# Pivot dataframe of tokens with threshold-differing effects to wide format
diff_token_proportion_directionality = pd.pivot_table(diff_token_proportion_directionality, values = 'proportion', index=['TUNE_IDX','GUPI','WindowIdx','Token'], columns = 'SHAPDirection').reset_index()

# For initial purpose, only keep instances in which token had both positive and negative effect across thresholds
diff_token_proportion_directionality = diff_token_proportion_directionality[(~diff_token_proportion_directionality.Negative.isna())&(~diff_token_proportion_directionality.Positive.isna())].reset_index(drop=True)

# Remerge full token key information
diff_token_proportion_directionality = diff_token_proportion_directionality.merge(full_token_keys,how='left')

## Extract and save TimeSHAP values corresponding to timepoints with threshold-differing effects
# Keep only TimeSHAP values in the threshold-differing effects dataframe
filt_thresh_diff_timeSHAP_values = compiled_feature_timeSHAP_values.merge(diff_token_proportion_directionality[['TUNE_IDX','GUPI','WindowIdx','Token']],how='inner').reset_index(drop=True)

# Save filtered dataframe as CSV
filt_thresh_diff_timeSHAP_values.to_csv(os.path.join(shap_dir,'filtered_thresh_differing_timeSHAP_values.csv'),index=False)

# ### III. Partition missed significant timepoints for second-pass parallel TimeSHAP calculation
# ## Partition evenly for parallel calculation
# # Load missed significant points of prognostic timepoint
# compiled_missed_timepoints = pd.read_pickle(os.path.join(shap_dir,'first_pass_missed_timepoints.pkl'))

# # Partition evenly along number of available array tasks
# max_array_tasks = 10000
# s = [compiled_missed_timepoints.shape[0] // max_array_tasks for _ in range(max_array_tasks)]
# s[:(compiled_missed_timepoints.shape[0] - sum(s))] = [over+1 for over in s[:(compiled_missed_timepoints.shape[0] - sum(s))]]    
# end_idx = np.cumsum(s)
# start_idx = np.insert(end_idx[:-1],0,0)
# timeshap_partitions = pd.concat([compiled_missed_timepoints.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True).assign(PARTITION_IDX=idx) for idx in range(len(start_idx))],ignore_index=True)

# # Save derived missed timepoint partitions
# cp.dump(timeshap_partitions, open(os.path.join(shap_dir,'second_pass_timeSHAP_partitions.pkl'), "wb" ))

# ### IV. Compile second-pass TimeSHAP values and clean directory
# ## Find completed second-pass TimeSHAP configurations and log remaining configurations, if any
# # Identify TimeSHAP dataframe files in parallel storage directory
# second_tsx_files = []
# for path in Path(os.path.join(second_sub_shap_dir)).rglob('timeSHAP_values_partition_idx_*'):
#     second_tsx_files.append(str(path.resolve()))

# # Characterise found second-pass TimeSHAP dataframe files
# second_tsx_info_df = pd.DataFrame({'FILE':second_tsx_files,
#                                    'PARTITION_IDX':[int(re.search('partition_idx_(.*).pkl', curr_file).group(1)) for curr_file in second_tsx_files]
#                                   }).sort_values(by=['PARTITION_IDX']).reset_index(drop=True)

# # Identify second-pass TimeSHAP significant timepoints that were missed based on stored files
# second_missed_timepoint_files = []
# for path in Path(os.path.join(second_missed_timepoint_dir)).rglob('timeSHAP_values_partition_idx_*'):
#     second_missed_timepoint_files.append(str(path.resolve()))
# for path in Path(os.path.join(second_missed_timepoint_dir)).rglob('missing_timepoints_partition_idx_*'):
#     second_missed_timepoint_files.append(str(path.resolve()))
    
# # Characterise found second-pass missing timepoint dataframe files
# second_missed_info_df = pd.DataFrame({'FILE':second_missed_timepoint_files,
#                                       'PARTITION_IDX':[int(re.search('partition_idx_(.*).pkl', curr_file).group(1)) for curr_file in second_missed_timepoint_files]
#                                      }).sort_values(by=['PARTITION_IDX']).reset_index(drop=True)

# # Determine partition indices that have not yet been accounted for
# full_range = list(range(10000))
# remaining_partition_indices = np.sort(list(set(full_range)-set(second_tsx_info_df.PARTITION_IDX)-set(second_missed_info_df.PARTITION_IDX))).tolist()

# # Create second-pass partitions for TimeSHAP configurations that are unaccounted for
# second_partition_list = pd.read_pickle(os.path.join(shap_dir,'second_pass_timeSHAP_partitions.pkl'))
# remaining_timeshap_partitions = second_partition_list[second_partition_list.PARTITION_IDX.isin(remaining_partition_indices)].reset_index(drop=True)

# # Save remaining partitions
# remaining_timeshap_partitions.to_pickle(os.path.join(shap_dir,'second_pass_remaining_timeSHAP_partitions.pkl'))

# ## In parallel, load, compile, and save second-pass missed significant timepoints
# # Partition missed timepoint files across available cores
# s = [second_missed_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
# s[:(second_missed_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(second_missed_info_df.shape[0] - sum(s))]]    
# end_idx = np.cumsum(s)
# start_idx = np.insert(end_idx[:-1],0,0)
# missed_files_per_core = [(second_missed_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling second-pass missed significant timepoints') for idx in range(len(start_idx))]

# # Load second-pass missed signficant timepoint dataframes in parallel
# with multiprocessing.Pool(NUM_CORES) as pool:
#     second_pass_compiled_missed_timepoints = pd.concat(pool.starmap(load_timeSHAP, missed_files_per_core),ignore_index=True)

# # Save compiled second-pass missed timepoints dataframe into TimeSHAP directory
# second_pass_compiled_missed_timepoints.to_pickle(os.path.join(shap_dir,'second_pass_missed_timepoints.pkl'))

# ## In parallel, load, compile, and save second-pass TimeSHAP values
# # Partition completed second-pass TimeSHAP files across available cores
# s = [second_tsx_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
# s[:(second_tsx_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(second_tsx_info_df.shape[0] - sum(s))]]    
# end_idx = np.cumsum(s)
# start_idx = np.insert(end_idx[:-1],0,0)
# tsx_files_per_core = [(second_tsx_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'Loading and compiling second-pass TimeSHAP values') for idx in range(len(start_idx))]

# # Load completed second-pass TimeSHAP dataframes in parallel
# with multiprocessing.Pool(NUM_CORES) as pool:
#     second_compiled_feature_timeSHAP_values = pd.concat(pool.starmap(load_timeSHAP, tsx_files_per_core),ignore_index=True)

# # Load compiled first-pass TimeSHAP values dataframe from TimeSHAP directory
# first_compiled_feature_timeSHAP_values = pd.read_pickle(os.path.join(shap_dir,'first_pass_timeSHAP_values.pkl')).rename(columns={'Feature':'Token'})

# # Clean first-pass TimeSHAP values dataframe before compilation
# first_compiled_feature_timeSHAP_values = first_compiled_feature_timeSHAP_values.drop(columns=['Random seed','NSamples'])
# first_compiled_feature_timeSHAP_values['PASS'] = 'First'
                                        
# # Clean second-pass TimeSHAP values dataframe before compilation
# second_compiled_feature_timeSHAP_values = second_compiled_feature_timeSHAP_values.rename(columns={'Feature':'Token'}).drop(columns=['Random seed','NSamples'])
# second_compiled_feature_timeSHAP_values['PASS'] = 'Second'

# # Compile and save TimeSHAP values dataframe into TimeSHAP directory
# compiled_feature_timeSHAP_values = pd.concat([first_compiled_feature_timeSHAP_values,second_compiled_feature_timeSHAP_values],ignore_index=True).sort_values(by=['TUNE_IDX','FOLD','Threshold','GUPI','WindowIdx','Token'],ignore_index=True)
                                        
# # Save compiled TimeSHAP values dataframe into TimeSHAP directory
# compiled_feature_timeSHAP_values.to_pickle(os.path.join(shap_dir,'compiled_feature_timeSHAP_values.pkl'))

# ## After compiling and saving values, delete individual files
# # Delete second-pass missed timepoint files
# shutil.rmtree(second_missed_timepoint_dir)

# # Delete second-pass TimeSHAP value files
# shutil.rmtree(second_sub_shap_dir)