#### Master Script 05c: Calculating TimeSHAP for TILTomorrow models in parallel ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate testing set TimeSHAP values based on provided TimeSHAP partition row index

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

# Import TimeSHAP methods
import timeshap.explainer as tsx
import timeshap.plot as tsp
from timeshap.wrappers import TorchModelWrapper
from timeshap.utils import get_avg_score_with_avg_event

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

## Define and create relevant directories
# Define model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)

# Define directory in which tokens are stored
tokens_dir = os.path.join('/home/sb2406/rds/hpc-work','tokens')

# Define a directory for the storage of model interpretation values
interp_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_interpretations',VERSION)

# Define a directory for the storage of TimeSHAP values
shap_dir = os.path.join(interp_dir,'timeSHAP')

# Create a subdirectory for the storage of TimeSHAP values
sub_shap_dir = os.path.join(shap_dir,'parallel_results')
os.makedirs(sub_shap_dir,exist_ok=True)

# Create a subdirectory for the storage of missed TimeSHAP timepoints
missed_timepoints_dir = os.path.join(shap_dir,'missed_timepoints')
os.makedirs(missed_timepoints_dir,exist_ok=True)

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

# Load prepared token dictionary
full_token_keys = pd.read_excel(os.path.join(tokens_dir,'TILTomorrow_full_token_keys_'+VERSION+'.xlsx'))
full_token_keys.Token = full_token_keys.Token.fillna('')
full_token_keys.BaseToken = full_token_keys.BaseToken.fillna('')

# Load partitioned significant clinical timepoints for allocated TimeSHAP calculation
timeshap_partitions = pd.read_pickle(os.path.join(shap_dir,'timeSHAP_partitions.pkl'))

### II. Calculate testing set TimeSHAP values based on provided TimeSHAP partition row index
# Argument-induced bootstrapping functions
def main(array_task_id):

    ## Calculate the "average-event" for TimeSHAP
    # Extract current significant clinical timepoints based on `array_task_id`
    curr_timepoints = timeshap_partitions[timeshap_partitions.PARTITION_IDX==array_task_id].reset_index(drop=True)
    
    # Identify unique CV partitions in current batch to load training set outputs
    unique_cv_partitons = curr_timepoints[['REPEAT','FOLD','TUNE_IDX','DROPOUT_VARS']].drop_duplicates().reset_index(drop=True)
    
    # Create empty lists to store average events and zero events
    avg_event_lists = []
    zero_event_lists = []
    
    # Create empty list to store current timepoint testing set outputs
    curr_testing_sets = []
    
    # Iterate through unique CV partitions to load and calculate average event
    for curr_cv_row in tqdm(range(unique_cv_partitons.shape[0]),'Iterating through unique cross-validation partitions to calculate TimeSHAP'):
        
        # Extract current repeat, fold, and tuning index
        curr_repeat = unique_cv_partitons.REPEAT[curr_cv_row]
        curr_fold = unique_cv_partitons.FOLD[curr_cv_row]
        curr_tune_idx = unique_cv_partitons.TUNE_IDX[curr_cv_row]
        curr_dropout_vars = unique_cv_partitons.DROPOUT_VARS[curr_cv_row]

        # Define current fold token subdirectory
        token_fold_dir = os.path.join(tokens_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold))
            
        # Load current token-indexed training set
        training_set = pd.read_pickle(os.path.join(token_fold_dir,'TILTomorrow_training_indices.pkl'))
            
        # Filter training set outputs based on `WindowIdx`
        training_set = training_set[training_set.WindowIdx.isin(SHAP_WINDOW_INDICES)].reset_index(drop=True)

        # Load current token-indexed testing set
        testing_set = pd.read_pickle(os.path.join(token_fold_dir,'TILTomorrow_testing_indices.pkl'))

        # Filter testing set outputs based on `WindowIdx`
        testing_set = testing_set[testing_set.WindowIdx.isin(SHAP_WINDOW_INDICES)].reset_index(drop=True)

        # Load current token dictionary
        curr_vocab = cp.load(open(os.path.join(token_fold_dir,'TILTomorrow_token_dictionary.pkl'),"rb"))
        unknown_index = curr_vocab['<unk>']
        
        # Create dataframe version of vocabulary
        curr_vocab_df = pd.DataFrame({'VocabIndex':list(range(len(curr_vocab))),'Token':curr_vocab.get_itos()})

        # Merge token dictionary information onto current vocabulary
        curr_vocab_df = curr_vocab_df.merge(full_token_keys,how='left')

        # Extract relevant current configuration hyperparameters
        curr_physician_impressions = tuning_grid.PHYS_IMPRESSION_TOKENS[(tuning_grid.TUNE_IDX==curr_tune_idx)].values[0]
        curr_outcome_label = tuning_grid.OUTCOME_LABEL[(tuning_grid.TUNE_IDX==curr_tune_idx)].values[0]
        curr_rnn_type = tuning_grid.RNN_TYPE[(tuning_grid.TUNE_IDX==curr_tune_idx)].values[0]
        curr_max_tokens_per_base_token = tuning_grid.MAX_TOKENS_PER_BASE_TOKEN[(tuning_grid.TUNE_IDX==curr_tune_idx)].values[0]
        curr_base_token_representation = tuning_grid.MIN_BASE_TOKEN_REPRESENATION[(tuning_grid.TUNE_IDX==curr_tune_idx)].values[0]
    
        # Create copies of training, validation, and testing sets for configuration-specific formatting
        format_training_set = training_set.copy()
        format_testing_set = testing_set.copy()
        
        # Format tokens based on physician-impression decision
        if curr_physician_impressions:
            format_training_set.VocabIndex = format_training_set.VocabIndex + format_training_set.VocabPhysImpressionIndex
            format_testing_set.VocabIndex = format_testing_set.VocabIndex + format_testing_set.VocabPhysImpressionIndex

        # Create a list to store banned token indices for current parametric combination
        banned_indices = []

        # If there is a maximum on the number of tokens per base token, remove base tokens which violate this limit
        if curr_max_tokens_per_base_token != 'None':
            tokens_per_base_token = full_token_keys[(~full_token_keys.Missing)&(~full_token_keys.BaseToken.isin(['','<unk>','DayOfICUStay']))].groupby(['BaseToken'],as_index=False).Token.nunique()
            base_tokens_to_mask = tokens_per_base_token.BaseToken[tokens_per_base_token.Token > int(curr_max_tokens_per_base_token)].unique()
            banned_indices += curr_vocab_df[curr_vocab_df.BaseToken.isin(base_tokens_to_mask)].VocabIndex.unique().tolist()

        # If there is a minimum on the number of patients needed per base token, remove base tokens which violate this limit
        if curr_base_token_representation != 'None':
            token_counts_per_patient = pd.read_pickle(os.path.join(token_fold_dir,'TILTomorrow_token_incidences_per_patient.pkl')).merge(full_token_keys[['Token','BaseToken']],how='left')
            token_counts_per_patient = token_counts_per_patient[token_counts_per_patient.GUPI.isin(training_set.GUPI.unique())].reset_index(drop=True)
            patient_counts_per_base_token = token_counts_per_patient.groupby('BaseToken',as_index=False).GUPI.nunique()
            base_tokens_to_mask = patient_counts_per_base_token.BaseToken[patient_counts_per_base_token.GUPI<(float(curr_base_token_representation)*training_set.GUPI.nunique())].unique()
            mask_indices += curr_vocab_df[curr_vocab_df.BaseToken.isin(base_tokens_to_mask)].VocabIndex.unique().tolist()





        # Ensure indices are unique
        format_training_set.VocabIndex = format_training_set.VocabIndex.apply(lambda x: np.unique(x).tolist())
        format_testing_set.VocabIndex = format_testing_set.VocabIndex.apply(lambda x: np.unique(x).tolist())
        
        # Calculate maximum number of unknowns in testing set
        format_testing_set['SeqLength'] = format_testing_set.VocabIndex.apply(len)
        format_testing_set['Unknowns'] = format_testing_set.VocabIndex.apply(lambda x: x.count(unknown_index))      
        
        # Calculate number of columns to add
        cols_to_add = max(format_testing_set['Unknowns'].max(),1) - 1
        
        # Define token labels from current vocab
        token_labels = curr_vocab.get_itos() + [curr_vocab.get_itos()[unknown_index]+'_'+str(i+1).zfill(3) for i in range(cols_to_add)]
        token_labels[unknown_index] = token_labels[unknown_index]+'_000'
        
        # Convert training set dataframe to multihot matrix
        training_multihot = df_to_multihot_matrix(format_training_set, len(curr_vocab), unknown_index, cols_to_add)
        
        # Define average-token dataframe from training set for "average event"
        training_token_frequencies = training_multihot.sum(0)/training_multihot.shape[0]
        average_event = pd.DataFrame(np.expand_dims((training_token_frequencies>0.5).astype(int),0), index=np.arange(1), columns=token_labels)

        # Add CV partition and tuning index to average event dataframe and append to empty list
        average_event.insert(0,'REPEAT',curr_repeat)
        average_event.insert(1,'FOLD',curr_fold)
        average_event.insert(2,'TUNE_IDX',curr_tune_idx)
        avg_event_lists.append(average_event)
        
        # Define zero-token dataframe for second-pass "average event"
        zero_event = pd.DataFrame(0, index=np.arange(1), columns=token_labels)
        
        # Add CV partition and tuning index to average event dataframe and append to empty list
        zero_event.insert(0,'REPEAT',curr_repeat)
        zero_event.insert(1,'FOLD',curr_fold)
        zero_event.insert(2,'TUNE_IDX',curr_tune_idx)
        zero_event_lists.append(zero_event)
        
        # Add cross-validation partition and tuning configuration information to testing set dataframe
        format_testing_set['REPEAT'] = curr_repeat
        format_testing_set['FOLD']= curr_fold
        format_testing_set['TUNE_IDX'] = curr_tune_idx
        
        # Filter testing set and store
        curr_testing_sets.append(format_testing_set[format_testing_set.GUPI.isin(curr_timepoints[(curr_timepoints.REPEAT==curr_repeat)&(curr_timepoints.FOLD==curr_fold)&(curr_timepoints.TUNE_IDX==curr_tune_idx)].GUPI.unique())].reset_index(drop=True))
        
    # Concatenate average- and zero-event lists for storage
    avg_event_lists = pd.concat(avg_event_lists,ignore_index=True)
    zero_event_lists = pd.concat(zero_event_lists,ignore_index=True)
    curr_testing_sets = pd.concat(curr_testing_sets,ignore_index=True)

    ## Calculate TimeSHAP values for output contributions to TILBasic thresholds
    # Define list of possible TILBasic thresholds
    thresh_labels = ['TILBasic>0','TILBasic>1','TILBasic>2','TILBasic>3']
    
    # Initialize empty list to compile TimeSHAP dataframes
    avg_compiled_threshold_TILBasic_ts = []
    zero_compiled_threshold_TILBasic_ts = []

    # Initialize empty list to compile TimeSHAP event dataframes
    avg_compiled_threshold_TILBasic_event_ts = []
    zero_compiled_threshold_TILBasic_event_ts = []

    # Initialize empty list to compile missed timepoints
    avg_compiled_threshold_TILBasic_missed = []
    zero_compiled_threshold_TILBasic_missed = []

    # Iterate through unique combinations and calculate TimeSHAP    
    for curr_trans_row in tqdm(range(curr_timepoints.shape[0]),'Iterating through unique combinations to calculate threshold TILBasic TimeSHAP'):
        
        # Extract current repeat, fold, GUPI, tuning index, and window index
        curr_repeat = curr_timepoints.REPEAT[curr_trans_row]
        curr_fold = curr_timepoints.FOLD[curr_trans_row]
        curr_GUPI = curr_timepoints.GUPI[curr_trans_row]
        curr_tune_idx = curr_timepoints.TUNE_IDX[curr_trans_row]
        curr_wi = curr_timepoints.WindowIdx[curr_trans_row]
        curr_thresh_idx = thresh_labels.index(SHAP_THRESHOLD)
        
        # Extract average- and zero-events based on current combination parameters
        curr_avg_event = avg_event_lists[(avg_event_lists.REPEAT==curr_repeat)&(avg_event_lists.FOLD==curr_fold)&(avg_event_lists.TUNE_IDX==curr_tune_idx)].drop(columns=['REPEAT','FOLD','TUNE_IDX']).reset_index(drop=True)
        curr_zero_event = zero_event_lists[(zero_event_lists.REPEAT==curr_repeat)&(zero_event_lists.FOLD==curr_fold)&(zero_event_lists.TUNE_IDX==curr_tune_idx)].drop(columns=['REPEAT','FOLD','TUNE_IDX']).reset_index(drop=True)
        
        # Extract testing set outputs based on current combination parameters
        filt_testing_set = curr_testing_sets[(curr_testing_sets.GUPI==curr_GUPI)&(curr_testing_sets.REPEAT==curr_repeat)&(curr_testing_sets.FOLD==curr_fold)&(curr_testing_sets.TUNE_IDX==curr_tune_idx)].reset_index(drop=True)
        
        # Define current fold token subdirectory
        token_fold_dir = os.path.join(tokens_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold))

        # Load current token dictionary
        curr_vocab = cp.load(open(os.path.join(token_fold_dir,'TILTomorrow_token_dictionary.pkl'),"rb"))
        unknown_index = curr_vocab['<unk>']

        # Convert filtered testing set dataframe to multihot matrix
        testing_multihot = df_to_multihot_matrix(filt_testing_set, len(curr_vocab), unknown_index, curr_avg_event.shape[1]-len(curr_vocab))

        # Filter testing multihot matrix up to the window index of focus
        filt_testing_multihot = np.expand_dims(testing_multihot[:curr_wi,:],axis=0)
                        
        # Extract current file and required hyperparameter information
        curr_file = ckpt_info.file[(ckpt_info.REPEAT==curr_repeat)&(ckpt_info.FOLD==curr_fold)&(ckpt_info.TUNE_IDX==curr_tune_idx)].values[0]
        curr_rnn_type = tuning_grid[tuning_grid.TUNE_IDX==curr_tune_idx].RNN_TYPE.values[0]
            
        # Load current pretrained model
        ttm_model = TILTomorrow_model.load_from_checkpoint(curr_file)
        ttm_model.eval()
        
        # Initialize custom TimeSHAP model for threshold value effect calculation
        ts_TILBasic_model = timeshap_TILTomorrow_model(ttm_model,curr_rnn_type,curr_thresh_idx,unknown_index,curr_avg_event.shape[1]-len(curr_vocab))
        wrapped_ttm_model = TorchModelWrapper(ts_TILBasic_model)
        f_hs = lambda x, y=None: wrapped_ttm_model.predict_last_hs(x, y)

        # First, try to calculate threshold TILBasic TimeSHAP values with average event baseline
        try:
            # Prune timepoints based on tolerance of 0.025
            _,prun_idx = tsx.local_pruning(f_hs, filt_testing_multihot, {'tol': 0.025}, curr_avg_event, entity_uuid=None, entity_col=None, verbose=True)
            
            # Calculate local feature-level TimeSHAP values after pruning
            feature_dict = {'rs': 2023, 'nsamples': 3200, 'feature_names': curr_avg_event.columns.to_list()}
            ts_feature_data = tsx.local_feat(f_hs, filt_testing_multihot, feature_dict, entity_uuid=None, entity_col=None, baseline=curr_avg_event, pruned_idx=filt_testing_multihot.shape[1]+prun_idx)
            
            # Calculate local event-level TimeSHAP values after pruning
            event_dict = {'rs': 2023, 'nsamples': 32000}
            ts_event_data = tsx.local_event(f_hs, filt_testing_multihot, event_dict, entity_uuid=None, entity_col=None, baseline=curr_avg_event, pruned_idx=filt_testing_multihot.shape[1]+prun_idx)
            
            # Find features that exist within unpruned region
            existing_features = np.asarray(curr_avg_event.columns.to_list())[filt_testing_multihot[:,filt_testing_multihot.shape[1]+prun_idx:,:].sum(1).squeeze(0) > 0]

            # Filter feature-level TimeSHAP values to existing features
            ts_feature_data = ts_feature_data[ts_feature_data.Feature.isin(existing_features)].reset_index(drop=True)
            
            # Add metadata to TimeSHAP feature dataframe
            ts_feature_data['REPEAT'] = curr_repeat
            ts_feature_data['FOLD'] = curr_fold
            ts_feature_data['TUNE_IDX'] = curr_tune_idx
            ts_feature_data['Threshold'] = SHAP_THRESHOLD
            ts_feature_data['GUPI'] = curr_GUPI
            ts_feature_data['WindowIdx'] = curr_wi
            ts_feature_data['BaselineFeatures'] = 'Average'
            ts_feature_data['PruneIdx'] = prun_idx
            
            # Add metadata to TimeSHAP event dataframe
            ts_event_data['REPEAT'] = curr_repeat
            ts_event_data['FOLD'] = curr_fold
            ts_event_data['TUNE_IDX'] = curr_tune_idx
            ts_event_data['Threshold'] = SHAP_THRESHOLD
            ts_event_data['GUPI'] = curr_GUPI
            ts_event_data['WindowIdx'] = curr_wi
            ts_event_data['BaselineFeatures'] = 'Average'
            ts_event_data['PruneIdx'] = prun_idx
            
            #Append current TimeSHAP feature dataframe to compilation list
            avg_compiled_threshold_TILBasic_ts.append(ts_feature_data)
            
            #Append current TimeSHAP event dataframe to compilation list
            avg_compiled_threshold_TILBasic_event_ts.append(ts_event_data)

        except:
            # Identify significant timepoints for which TimeSHAP cannot be calculated
            curr_missed_timepoint = curr_timepoints.iloc[[curr_trans_row]].reset_index(drop=True)

            # Append to running list of missing timepoints
            avg_compiled_threshold_TILBasic_missed.append(curr_missed_timepoint)

        # Second, try to calculate threshold TILBasic TimeSHAP values with zero event baseline
        try:                 
            # Prune timepoints based on tolerance of 0.025
            _,prun_idx = tsx.local_pruning(f_hs, filt_testing_multihot, {'tol': 0.025}, curr_zero_event, entity_uuid=None, entity_col=None, verbose=True)

            # Calculate local feature-level TimeSHAP values after pruning
            feature_dict = {'rs': 2023, 'nsamples': 3200, 'feature_names': curr_zero_event.columns.to_list()}
            ts_feature_data = tsx.local_feat(f_hs, filt_testing_multihot, feature_dict, entity_uuid=None, entity_col=None, baseline=curr_zero_event, pruned_idx=filt_testing_multihot.shape[1]+prun_idx)

            # Calculate local event-level TimeSHAP values after pruning
            event_dict = {'rs': 2023, 'nsamples': 32000}
            ts_event_data = tsx.local_event(f_hs, filt_testing_multihot, event_dict, entity_uuid=None, entity_col=None, baseline=curr_zero_event, pruned_idx=filt_testing_multihot.shape[1]+prun_idx)

            # Find features that exist within unpruned region
            existing_features = np.asarray(curr_zero_event.columns.to_list())[filt_testing_multihot[:,filt_testing_multihot.shape[1]+prun_idx:,:].sum(1).squeeze(0) > 0]

            # Filter feature-level TimeSHAP values to existing features
            ts_feature_data = ts_feature_data[ts_feature_data.Feature.isin(existing_features)].reset_index(drop=True)

            # Add metadata to TimeSHAP feature dataframe
            ts_feature_data['REPEAT'] = curr_repeat
            ts_feature_data['FOLD'] = curr_fold
            ts_feature_data['TUNE_IDX'] = curr_tune_idx
            ts_feature_data['Threshold'] = SHAP_THRESHOLD
            ts_feature_data['GUPI'] = curr_GUPI
            ts_feature_data['WindowIdx'] = curr_wi
            ts_feature_data['BaselineFeatures'] = 'Zero'
            ts_feature_data['PruneIdx'] = prun_idx

            # Add metadata to TimeSHAP event dataframe
            ts_event_data['REPEAT'] = curr_repeat
            ts_event_data['FOLD'] = curr_fold
            ts_event_data['TUNE_IDX'] = curr_tune_idx
            ts_event_data['Threshold'] = SHAP_THRESHOLD
            ts_event_data['GUPI'] = curr_GUPI
            ts_event_data['WindowIdx'] = curr_wi
            ts_event_data['BaselineFeatures'] = 'Zero'
            ts_event_data['PruneIdx'] = prun_idx

            #Append current TimeSHAP dataframe to compilation list
            zero_compiled_threshold_TILBasic_ts.append(ts_feature_data)

            #Append current TimeSHAP dataframe to compilation list
            zero_compiled_threshold_TILBasic_event_ts.append(ts_event_data)

        except:    
            # Identify significant timepoints for which TimeSHAP cannot be calculated
            curr_missed_timepoint = curr_timepoints.iloc[[curr_trans_row]].reset_index(drop=True)

            # Append to running list of missing timepoints
            zero_compiled_threshold_TILBasic_missed.append(curr_missed_timepoint)
    
    # Based on availability, save compiled TimeSHAP values
    if avg_compiled_threshold_TILBasic_ts:
        
        # Compile list of TimeSHAP dataframes
        avg_compiled_threshold_TILBasic_ts = pd.concat(avg_compiled_threshold_TILBasic_ts,ignore_index=True)

        # Rename `Shapley Value` column
        avg_compiled_threshold_TILBasic_ts = avg_compiled_threshold_TILBasic_ts.rename(columns={'Shapley Value':'SHAP'})

        # Save compiled TimeSHAP values into SHAP subdirectory
        avg_compiled_threshold_TILBasic_ts.to_pickle(os.path.join(sub_shap_dir,'avg_thresh_TILBasic_features_timeSHAP_values_partition_idx_'+str(array_task_id).zfill(4)+'.pkl'))
        
    # Based on availability, save compiled TimeSHAP values
    if zero_compiled_threshold_TILBasic_ts:
        
        # Compile list of TimeSHAP dataframes
        zero_compiled_threshold_TILBasic_ts = pd.concat(zero_compiled_threshold_TILBasic_ts,ignore_index=True)

        # Rename `Shapley Value` column
        zero_compiled_threshold_TILBasic_ts = zero_compiled_threshold_TILBasic_ts.rename(columns={'Shapley Value':'SHAP'})

        # Save compiled TimeSHAP values into SHAP subdirectory
        zero_compiled_threshold_TILBasic_ts.to_pickle(os.path.join(sub_shap_dir,'zero_thresh_TILBasic_features_timeSHAP_values_partition_idx_'+str(array_task_id).zfill(4)+'.pkl'))
        
    # Based on availability, save compiled TimeSHAP values
    if avg_compiled_threshold_TILBasic_event_ts:
        
        # Compile list of TimeSHAP dataframes
        avg_compiled_threshold_TILBasic_event_ts = pd.concat(avg_compiled_threshold_TILBasic_event_ts,ignore_index=True)

        # Rename `Shapley Value` column
        avg_compiled_threshold_TILBasic_event_ts = avg_compiled_threshold_TILBasic_event_ts.rename(columns={'Shapley Value':'SHAP'})

        # Save compiled TimeSHAP values into SHAP subdirectory
        avg_compiled_threshold_TILBasic_event_ts.to_pickle(os.path.join(sub_shap_dir,'avg_thresh_TILBasic_event_timeSHAP_values_partition_idx_'+str(array_task_id).zfill(4)+'.pkl'))
        
    # Based on availability, save compiled TimeSHAP values
    if zero_compiled_threshold_TILBasic_event_ts:
        
        # Compile list of TimeSHAP dataframes
        zero_compiled_threshold_TILBasic_event_ts = pd.concat(zero_compiled_threshold_TILBasic_event_ts,ignore_index=True)

        # Rename `Shapley Value` column
        zero_compiled_threshold_TILBasic_event_ts = zero_compiled_threshold_TILBasic_event_ts.rename(columns={'Shapley Value':'SHAP'})

        # Save compiled TimeSHAP values into SHAP subdirectory
        zero_compiled_threshold_TILBasic_event_ts.to_pickle(os.path.join(sub_shap_dir,'zero_thresh_TILBasic_event_timeSHAP_values_partition_idx_'+str(array_task_id).zfill(4)+'.pkl'))
        
    # Based on availability, save compiled TimeSHAP values
    if avg_compiled_threshold_TILBasic_missed:
        
        # Compile list of TimeSHAP dataframes
        avg_compiled_threshold_TILBasic_missed = pd.concat(avg_compiled_threshold_TILBasic_missed,ignore_index=True)
        
        # Save compiled TimeSHAP values into SHAP subdirectory
        avg_compiled_threshold_TILBasic_missed.to_pickle(os.path.join(missed_timepoints_dir,'avg_missed_timepoints_partition_idx_'+str(array_task_id).zfill(4)+'.pkl'))
        
    # Based on availability, save compiled TimeSHAP values
    if zero_compiled_threshold_TILBasic_missed:
        
        # Compile list of TimeSHAP dataframes
        zero_compiled_threshold_TILBasic_missed = pd.concat(zero_compiled_threshold_TILBasic_missed,ignore_index=True)
        
        # Save compiled TimeSHAP values into SHAP subdirectory
        zero_compiled_threshold_TILBasic_missed.to_pickle(os.path.join(missed_timepoints_dir,'zero_missed_timepoints_partition_idx_'+str(array_task_id).zfill(4)+'.pkl'))
        
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])+1
    main(array_task_id)