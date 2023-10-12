#### Master Script 05a: Extract relevance layer values from trained TILTomorrow models ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Find all top-performing model checkpoint files for relevance layer extraction
# III. Iteratively extract relevance layers from top-performing tuning configurations
# IV. Characterise and summarise relevance layer tokens based on `BaseToken`

### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import copy
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

# Custom methods
from models.dynamic_TTM import TILTomorrow_model

## Define parameters
# Set version code
VERSION = 'v2-0'

## Define and create relevant directories
# Define directory in which tokens are stored for each partition
tokens_dir = '/home/sb2406/rds/hpc-work/tokens'

# Define model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)

# Define a directory for the storage of model interpretation values
interp_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_interpretations',VERSION)
os.makedirs(interp_dir,exist_ok=True)

# Define a directory for the storage of relevance layer values
relevance_dir = os.path.join(interp_dir,'relevance_layer')
os.makedirs(relevance_dir,exist_ok=True)

## Load fundamental information
# Load the current version tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))

# Load cross-validation split information to extract testing resamples
cv_splits = pd.read_csv('../cross_validation_splits.csv')
test_splits = cv_splits[cv_splits.SET == 'test'].reset_index(drop=True)
uniq_GUPIs = test_splits.GUPI.unique()

# Load sensitivty analysis grid
sens_analysis_grid = pd.read_csv(os.path.join(model_dir,'sens_analysis_grid.csv'))
sens_analysis_grid = sens_analysis_grid[['SENS_IDX','DROPOUT_VARS']].drop_duplicates(ignore_index=True)

# Load prepared token dictionary
full_token_keys = pd.read_excel(os.path.join(tokens_dir,'TILTomorrow_full_token_keys_'+VERSION+'.xlsx'))
full_token_keys.Token = full_token_keys.Token.fillna('')
full_token_keys.BaseToken = full_token_keys.BaseToken.fillna('')

### II. Find all top-performing model checkpoint files for relevance layer extraction
# Either create or load TILTomorrow checkpoint information for relevance layer extraction
if not os.path.exists(os.path.join(relevance_dir,'full_ckpt_info.pkl')):
    
    # Find all model checkpoint files in TILTomorrow output directory
    ckpt_files = []
    for path in Path(model_dir).rglob('*.ckpt'):
        ckpt_files.append(str(path.resolve()))
        
    # Separate checkpoint file list based on sensitivity analysis status
    full_ckpt_files = [f for f in ckpt_files if 'sens' not in f]
    sens_ckpt_files = [f for f in ckpt_files if 'sens' in f]

    # Categorize full model checkpoint files based on name
    full_ckpt_info = pd.DataFrame({'file':full_ckpt_files,
                                   'TUNE_IDX':[int(re.search('tune(.*)/epoch=', curr_file).group(1)) for curr_file in full_ckpt_files],
                                   'VERSION':[re.search('model_outputs/(.*)/repeat', curr_file).group(1) for curr_file in full_ckpt_files],
                                   'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in full_ckpt_files],
                                   'FOLD':[int(re.search('/fold(.*)/tune', curr_file).group(1)) for curr_file in full_ckpt_files],
                                   'VAL_METRIC':[re.search('val_metric=(.*).ckpt', curr_file).group(1) for curr_file in full_ckpt_files]
                                  }).sort_values(by=['FOLD','TUNE_IDX','VERSION']).reset_index(drop=True)
    full_ckpt_info.VAL_METRIC = full_ckpt_info.VAL_METRIC.str.split('-').str[0].astype(float)
    
    # Categorize sensitivity analysis model checkpoint files based on name
    sens_ckpt_info = pd.DataFrame({'file':sens_ckpt_files,
                                   'TUNE_IDX':[int(re.search('tune(.*)/sens', curr_file).group(1)) for curr_file in sens_ckpt_files],
                                   'SENS_IDX':[int(re.search('sens(.*)/epoch', curr_file).group(1)) for curr_file in sens_ckpt_files],
                                   'VERSION':[re.search('model_outputs/(.*)/repeat', curr_file).group(1) for curr_file in sens_ckpt_files],
                                   'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in sens_ckpt_files],
                                   'FOLD':[int(re.search('/fold(.*)/tune', curr_file).group(1)) for curr_file in sens_ckpt_files],
                                   'VAL_METRIC':[re.search('val_metric=(.*).ckpt', curr_file).group(1) for curr_file in sens_ckpt_files]
                                  }).sort_values(by=['FOLD','TUNE_IDX','SENS_IDX','VERSION']).reset_index(drop=True)
    sens_ckpt_info.VAL_METRIC = sens_ckpt_info.VAL_METRIC.str.split('-').str[0].astype(float)
    
    # Save model checkpoint information dataframes
    full_ckpt_info.to_pickle(os.path.join(relevance_dir,'full_ckpt_info.pkl'))
    sens_ckpt_info.to_pickle(os.path.join(relevance_dir,'sens_ckpt_info.pkl'))

else:
    
    # Read model checkpoint information dataframes
    full_ckpt_info = pd.read_pickle(os.path.join(relevance_dir,'full_ckpt_info.pkl'))
    sens_ckpt_info = pd.read_pickle(os.path.join(relevance_dir,'sens_ckpt_info.pkl'))
    
### III. Iteratively extract relevance layers from top-performing tuning configurations
## Prepare environment for relevance layer weight extraction
# Initiate empty list to store compiled relevance layers
compiled_relevance_layers = []

# Decode dropout variables in sensitivity analysis model checkpoint dataframe
sens_ckpt_info = sens_ckpt_info.merge(sens_analysis_grid).drop(columns='SENS_IDX')

# Compile model checkpoint information dataframes
ckpt_info = pd.concat([full_ckpt_info,sens_ckpt_info],ignore_index=True)
ckpt_info.DROPOUT_VARS = ckpt_info.DROPOUT_VARS.fillna('none')

## Iterate through identified checkpoints to aggregate relevance layers
for ckpt_row in tqdm(range(ckpt_info.shape[0]),'Iterating through checkpoints to extract relevance layers'):
    
    # Extract current file, tune index, and fold information
    curr_file = ckpt_info.file[ckpt_row]
    curr_tune_idx = ckpt_info.TUNE_IDX[ckpt_row]
    curr_dropout_vars = ckpt_info.DROPOUT_VARS[ckpt_row]
    curr_repeat = ckpt_info.REPEAT[ckpt_row]
    curr_fold = ckpt_info.FOLD[ckpt_row]
    
    # Determine training set GUPIs in current partition
    curr_train_GUPIs = cv_splits[(cv_splits.REPEAT == curr_repeat)&(cv_splits.FOLD == curr_fold)&(cv_splits.SET == 'train')].GUPI.unique()

    # Filter out current tuning directory configuration hyperparameters
    curr_tune_hp = tuning_grid[(tuning_grid.TUNE_IDX == curr_tune_idx)&(tuning_grid.REPEAT == curr_repeat)&(tuning_grid.FOLD == curr_fold)].reset_index(drop=True)
    
    # Determine token filtering hyperparameters based on tuning configuration selection
    curr_max_tokens_per_base_token = curr_tune_hp.MAX_TOKENS_PER_BASE_TOKEN.values[0]
    curr_base_token_representation = curr_tune_hp.MIN_BASE_TOKEN_REPRESENATION.values[0]
    
    # Load current token dictionary
    curr_token_dir = os.path.join(tokens_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold))
    curr_vocab = cp.load(open(os.path.join(curr_token_dir,'TILTomorrow_token_dictionary.pkl'),"rb"))
    unknown_index = curr_vocab['<unk>']
    
    # Load current pretrained model
    ttm_model = TILTomorrow_model.load_from_checkpoint(curr_file)
    ttm_model.eval()
    
    # Extract relevance layer values
    with torch.no_grad():
        relevance_layer = torch.exp(ttm_model.embedW.weight.detach().squeeze(1)).numpy()
        token_labels = curr_vocab.get_itos()
        curr_relevance_df = pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                          'DROPOUT_VARS':curr_dropout_vars,
                                          'Token':token_labels,
                                          'RELEVANCE':relevance_layer,
                                          'REPEAT':curr_repeat,
                                          'FOLD':curr_fold})
        
    # Add base token information to relevance layer dataframe
    curr_relevance_df = curr_relevance_df.merge(full_token_keys[['Token','BaseToken']],how='left')
    
    # Remove tokens corresponding to current dropout set
    if curr_dropout_vars == 'dynamic':
        banned_tokens = np.sort(full_token_keys[~full_token_keys.Baseline].Token.unique())
    elif curr_dropout_vars == 'clinician_impressions':
        banned_tokens = np.sort(full_token_keys[full_token_keys.ClinicianInput].Token.unique())
    elif curr_dropout_vars == 'treatments':
        banned_tokens = np.sort(full_token_keys[full_token_keys.ICUIntervention].Token.unique())
    elif curr_dropout_vars == 'clinician_impressions_and_treatments':
        banned_tokens = np.sort(full_token_keys[full_token_keys.ClinicianInput | full_token_keys.ICUIntervention].Token.unique())
    elif curr_dropout_vars == 'none':
        banned_tokens = []
        
    # Create a list to store banned base tokens based on hyperparameters
    banned_base_tokens = []
    
    # If there is a maximum on the number of tokens per base token, remove base tokens which violate this limit
    if curr_max_tokens_per_base_token != 'None':
        tokens_per_base_token = full_token_keys[(~full_token_keys.Missing)&(~full_token_keys.BaseToken.isin(['','<unk>','DayOfICUStay']))].groupby(['BaseToken'],as_index=False).Token.nunique()
        banned_base_tokens += tokens_per_base_token.BaseToken[tokens_per_base_token.Token > int(curr_max_tokens_per_base_token)].unique().tolist()

    # If there is a minimum on the number of patients needed per base token, remove base tokens which violate this limit
    if curr_base_token_representation != 'None':
        token_counts_per_patient = pd.read_pickle(os.path.join(curr_token_dir,'TILTomorrow_token_incidences_per_patient.pkl')).merge(full_token_keys[['Token','BaseToken']],how='left')
        token_counts_per_patient = token_counts_per_patient[token_counts_per_patient.GUPI.isin(curr_train_GUPIs)].reset_index(drop=True)
        patient_counts_per_base_token = token_counts_per_patient.groupby('BaseToken',as_index=False).GUPI.nunique()
        banned_base_tokens += patient_counts_per_base_token.BaseToken[patient_counts_per_base_token.GUPI<(float(curr_base_token_representation)*len(curr_train_GUPIs))].unique().tolist()
        
    # Remove banned tokens and base tokens from relevance layer weights
    curr_relevance_df = curr_relevance_df[(~curr_relevance_df.Token.isin(banned_tokens))&(~curr_relevance_df.BaseToken.isin(banned_base_tokens))].drop(columns='BaseToken').reset_index(drop=True)
    
    # Add compiled relevance layer to running list
    compiled_relevance_layers.append(curr_relevance_df)

## Concatenate all extracted relevance layers and calculate summary statistics for each `TUNE_IDX`-`Token` combination
compiled_relevance_layers = pd.concat(compiled_relevance_layers, ignore_index=True)
agg_relevance_layers = compiled_relevance_layers.groupby(['TUNE_IDX','DROPOUT_VARS','Token'],as_index=False)['RELEVANCE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'min':np.min,'max':np.max,'Q1':lambda x: np.quantile(x,.25),'Q3':lambda x: np.quantile(x,.75),'resamples':'count'}).reset_index(drop=True)

### IV. Characterise and summarise relevance layer tokens based on `BaseToken`
## Characterise tokens based on matches in manually constructed token directory
# Merge base token key information to aggregated relevance layer dataframes
agg_relevance_layers = agg_relevance_layers.merge(full_token_keys,how='left')

# Remove blank token from aggregated relevance layer values
agg_relevance_layers = agg_relevance_layers[~(agg_relevance_layers.Token == '')].reset_index(drop=True)

# Save aggregated relevance layer values in the current relevance directory
agg_relevance_layers.to_csv(os.path.join(relevance_dir,'agg_relevance_layers.csv'),index=False)

## Summarise token relevances for plotting
# Load aggregated relevance layer values from the current relevance directory
agg_relevance_layers = pd.read_csv(os.path.join(relevance_dir,'agg_relevance_layers.csv'))

# Remove missing token values
nonmissing_agg_relevance_layers = agg_relevance_layers[~agg_relevance_layers.Missing].reset_index(drop=True)

# Take only maximum (nonmissing) token values per variable
variable_relevances = nonmissing_agg_relevance_layers.loc[nonmissing_agg_relevance_layers.groupby(['TUNE_IDX','DROPOUT_VARS','BaseToken'])['median'].idxmax()].sort_values(by='median',ascending=False).reset_index(drop=True)

# Identify top 20 and bottom 3 baseline variables per tuning index
baseline_relevance_layers = variable_relevances[variable_relevances.Baseline].sort_values(by=['TUNE_IDX','DROPOUT_VARS','median'],ascending=[True,True,False]).reset_index(drop=True)
specific_baseline_relevances = pd.concat([baseline_relevance_layers.groupby(['TUNE_IDX','DROPOUT_VARS']).head(20),baseline_relevance_layers.groupby(['TUNE_IDX','DROPOUT_VARS']).tail(3)],ignore_index=True)

# For baseline variables not in the top 20 or bottom 3, calculate summary statistics
unspecific_baseline_relevances = baseline_relevance_layers.merge(specific_baseline_relevances,how='left', indicator=True)
unspecific_baseline_relevances = unspecific_baseline_relevances[unspecific_baseline_relevances._merge=='left_only'].drop(columns='_merge').reset_index(drop=True)[['TUNE_IDX','DROPOUT_VARS','Token']]
unspecific_baseline_relevances = compiled_relevance_layers.merge(unspecific_baseline_relevances,how='inner').groupby(['TUNE_IDX','DROPOUT_VARS'],as_index=False)['RELEVANCE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'min':np.min,'max':np.max,'Q1':lambda x: np.quantile(x,.25),'Q3':lambda x: np.quantile(x,.75),'resamples':'count'}).reset_index(drop=True)

# Concatenate baseline variable relevances and save
plot_df_baseline_variables = pd.concat([specific_baseline_relevances,unspecific_baseline_relevances],ignore_index=True).sort_values(by=['TUNE_IDX','DROPOUT_VARS','median'],ascending=[True,True,False]).reset_index(drop=True)
plot_df_baseline_variables.Token[plot_df_baseline_variables.Token.isna()] = 'Other'
plot_df_baseline_variables.BaseToken[plot_df_baseline_variables.BaseToken.isna()] = 'Other'
plot_df_baseline_variables.to_csv(os.path.join(relevance_dir,'baseline_relevances_plot_df.csv'),index=False)

# Identify top 20 and bottom 3 dynamic variables per tuning index
dynamic_relevance_layers = variable_relevances[~variable_relevances.Baseline].sort_values(by=['TUNE_IDX','DROPOUT_VARS','median'],ascending=[True,True,False]).reset_index(drop=True)
specific_dynamic_relevances = pd.concat([dynamic_relevance_layers.groupby(['TUNE_IDX','DROPOUT_VARS']).head(20),dynamic_relevance_layers.groupby(['TUNE_IDX','DROPOUT_VARS']).tail(3)],ignore_index=True)

# For dynamic variables not in the top 20 or bottom 3, calculate summary statistics
unspecific_dynamic_relevances = dynamic_relevance_layers.merge(specific_dynamic_relevances,how='left', indicator=True)
unspecific_dynamic_relevances = unspecific_dynamic_relevances[unspecific_dynamic_relevances._merge=='left_only'].drop(columns='_merge').reset_index(drop=True)[['TUNE_IDX','DROPOUT_VARS','Token']]
unspecific_dynamic_relevances = compiled_relevance_layers.merge(unspecific_dynamic_relevances,how='inner').groupby(['TUNE_IDX','DROPOUT_VARS'],as_index=False)['RELEVANCE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'min':np.min,'max':np.max,'Q1':lambda x: np.quantile(x,.25),'Q3':lambda x: np.quantile(x,.75),'resamples':'count'}).reset_index(drop=True)

# Concatenate dynamic variable relevances and save
plot_df_dynamic_variables = pd.concat([specific_dynamic_relevances,unspecific_dynamic_relevances],ignore_index=True).sort_values(by=['TUNE_IDX','DROPOUT_VARS','median'],ascending=[True,True,False]).reset_index(drop=True)
plot_df_dynamic_variables.Token[plot_df_dynamic_variables.Token.isna()] = 'Other'
plot_df_dynamic_variables.BaseToken[plot_df_dynamic_variables.BaseToken.isna()] = 'Other'
plot_df_dynamic_variables.to_csv(os.path.join(relevance_dir,'dynamic_relevances_plot_df.csv'),index=False)

# Identify top 20 and bottom 3 intervention variables per tuning index
intervention_relevance_layers = variable_relevances[variable_relevances.ICUIntervention].sort_values(by=['TUNE_IDX','DROPOUT_VARS','median'],ascending=[True,True,False]).reset_index(drop=True)
specific_intervention_relevances = pd.concat([intervention_relevance_layers.groupby(['TUNE_IDX','DROPOUT_VARS']).head(20),intervention_relevance_layers.groupby(['TUNE_IDX','DROPOUT_VARS']).tail(3)],ignore_index=True)

# For intervention variables not in the top 20 or bottom 3, calculate summary statistics
unspecific_intervention_relevances = intervention_relevance_layers.merge(specific_intervention_relevances,how='left', indicator=True)
unspecific_intervention_relevances = unspecific_intervention_relevances[unspecific_intervention_relevances._merge=='left_only'].drop(columns='_merge').reset_index(drop=True)[['TUNE_IDX','DROPOUT_VARS','Token']]
unspecific_intervention_relevances = compiled_relevance_layers.merge(unspecific_intervention_relevances,how='inner').groupby(['TUNE_IDX','DROPOUT_VARS'],as_index=False)['RELEVANCE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'min':np.min,'max':np.max,'Q1':lambda x: np.quantile(x,.25),'Q3':lambda x: np.quantile(x,.75),'resamples':'count'}).reset_index(drop=True)

# Concatenate intervention variable relevances and save
plot_df_intervention_variables = pd.concat([specific_intervention_relevances,unspecific_intervention_relevances],ignore_index=True).sort_values(by=['TUNE_IDX','DROPOUT_VARS','median'],ascending=[True,True,False]).reset_index(drop=True)
plot_df_intervention_variables.Token[plot_df_intervention_variables.Token.isna()] = 'Other'
plot_df_intervention_variables.BaseToken[plot_df_intervention_variables.BaseToken.isna()] = 'Other'
plot_df_intervention_variables.to_csv(os.path.join(relevance_dir,'intervention_relevances_plot_df.csv'),index=False)

# Identify top 20 and bottom 3 TIL variables per tuning index
TIL_relevance_layers = variable_relevances[variable_relevances.Token.str.contains('TIL')].sort_values(by=['TUNE_IDX','DROPOUT_VARS','median'],ascending=[True,True,False]).reset_index(drop=True)
specific_TIL_relevances = pd.concat([TIL_relevance_layers.groupby(['TUNE_IDX','DROPOUT_VARS']).head(20),TIL_relevance_layers.groupby(['TUNE_IDX','DROPOUT_VARS']).tail(3)],ignore_index=True)

# For TIL variables not in the top 20 or bottom 3, calculate summary statistics
unspecific_TIL_relevances = TIL_relevance_layers.merge(specific_TIL_relevances,how='left', indicator=True)
unspecific_TIL_relevances = unspecific_TIL_relevances[unspecific_TIL_relevances._merge=='left_only'].drop(columns='_merge').reset_index(drop=True)[['TUNE_IDX','DROPOUT_VARS','Token']]
unspecific_TIL_relevances = compiled_relevance_layers.merge(unspecific_TIL_relevances,how='inner').groupby(['TUNE_IDX','DROPOUT_VARS'],as_index=False)['RELEVANCE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'min':np.min,'max':np.max,'Q1':lambda x: np.quantile(x,.25),'Q3':lambda x: np.quantile(x,.75),'resamples':'count'}).reset_index(drop=True)

# Concatenate TIL variable relevances and save
plot_df_TIL_variables = pd.concat([specific_TIL_relevances,unspecific_TIL_relevances],ignore_index=True).sort_values(by=['TUNE_IDX','DROPOUT_VARS','median'],ascending=[True,True,False]).reset_index(drop=True)
plot_df_TIL_variables.Token[plot_df_TIL_variables.Token.isna()] = 'Other'
plot_df_TIL_variables.BaseToken[plot_df_TIL_variables.BaseToken.isna()] = 'Other'
plot_df_TIL_variables.to_csv(os.path.join(relevance_dir,'TIL_relevances_plot_df.csv'),index=False)