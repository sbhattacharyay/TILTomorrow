#### Master Script 04b: Prepare environment to calculate TimeSHAP for TILTomorrow models ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Identify transition points in testing set outputs for TimeSHAP focus
# III. Partition significant transition points for parallel TimeSHAP calculation
# IV. Calculate average training set outputs per tuning configuration
# V. Determine distribution of signficant transitions over time and entropy
# VI. Summarise average output at each threshold over time

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
from functions.model_building import collate_batch, format_shap, format_tokens, format_time_tokens, df_to_multihot_matrix

## Define parameters for model training
# Set version code
VERSION = 'v1-0'

# Set threshold at which to calculate TimeSHAP values
SHAP_THRESHOLD = 'TILBasic>3'

## Define and create relevant directories
# Define model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)

# Define directory in which tokens are stored
tokens_dir = os.path.join('/home/sb2406/rds/hpc-work','tokens')

# Define a directory for the storage of model interpretation values
interp_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_interpretations',VERSION)

# Define a directory for the storage of TimeSHAP values
shap_dir = os.path.join(interp_dir,'timeSHAP')
os.makedirs(shap_dir,exist_ok=True)

## Load fundamental information for model training
# Load the current version tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))

# Load cross-validation split information to extract testing resamples
cv_splits = pd.read_csv('../cross_validation_splits.csv')
test_splits = cv_splits[cv_splits.SET == 'test'].reset_index(drop=True)
uniq_GUPIs = test_splits.GUPI.unique()

### II. Identify transition points in testing set outputs for TimeSHAP focus
# Load and filter testing set outputs
test_outputs_df = pd.read_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_test_calibrated_outputs.pkl'))
test_outputs_df = test_outputs_df[(test_outputs_df.TUNE_IDX.isin(tuning_grid.TUNE_IDX))].reset_index(drop=True)

# Remove logit columns
logit_cols = [col for col in test_outputs_df if col.startswith('z_TILBasic=')]
test_outputs_df = test_outputs_df.drop(columns=logit_cols).reset_index(drop=True)

# Calculate threshold-based output probabilities
prob_cols = [col for col in test_outputs_df if col.startswith('Pr(TILBasic=')]
thresh_labels = ['TILBasic>0','TILBasic>1','TILBasic>2','TILBasic>3']
for thresh in range(1,len(prob_cols)):
    cols_gt = prob_cols[thresh:]
    prob_gt = test_outputs_df[cols_gt].sum(1).values
    gt = (test_outputs_df['TrueLabel'] >= thresh).astype(int).values
    test_outputs_df['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
    test_outputs_df[thresh_labels[thresh-1]] = gt

# Remove TILBasic probability columns
test_outputs_df = test_outputs_df.drop(columns=prob_cols).reset_index(drop=True)

## Iterate through highest-intensity threshold and identify significant points of transition per patient
# First iterate through each GUPI and tuning index to identify points of prognostic change in correct direction during region of analysis
diff_values = []
below_thresh_preds = test_outputs_df[(test_outputs_df[SHAP_THRESHOLD] == 0)&(test_outputs_df.TrueLabel.notna())].reset_index(drop=True)
for curr_below_GUPI in tqdm(below_thresh_preds.GUPI.unique(),'Iterating through patients below threshold: '+SHAP_THRESHOLD):
    curr_GUPI_preds = below_thresh_preds[below_thresh_preds.GUPI==curr_below_GUPI].reset_index(drop=True)
    for curr_tune_idx in curr_GUPI_preds.TUNE_IDX.unique():
        curr_TI_preds = curr_GUPI_preds[curr_GUPI_preds.TUNE_IDX==curr_tune_idx][['REPEAT','FOLD','GUPI','TUNE_IDX','WindowIdx','Pr('+SHAP_THRESHOLD+')']].reset_index(drop=True)
        curr_TI_preds['Diff'] = curr_TI_preds['Pr('+SHAP_THRESHOLD+')'].diff()
        curr_TI_preds = curr_TI_preds[curr_TI_preds.Diff < 0].drop(columns=['Pr('+SHAP_THRESHOLD+')']).reset_index(drop=True)
        curr_TI_preds['Threshold'] = SHAP_THRESHOLD
        diff_values.append(curr_TI_preds)

above_thresh_preds = test_outputs_df[(test_outputs_df[SHAP_THRESHOLD] == 1)&(test_outputs_df.TrueLabel.notna())].reset_index(drop=True)
for curr_above_GUPI in tqdm(above_thresh_preds.GUPI.unique(),'Iterating through patients above threshold: '+SHAP_THRESHOLD):
    curr_GUPI_preds = above_thresh_preds[above_thresh_preds.GUPI==curr_above_GUPI].reset_index(drop=True)
    for curr_tune_idx in curr_GUPI_preds.TUNE_IDX.unique():
        curr_TI_preds = curr_GUPI_preds[curr_GUPI_preds.TUNE_IDX==curr_tune_idx][['REPEAT','FOLD','GUPI','TUNE_IDX','WindowIdx','Pr('+SHAP_THRESHOLD+')']].reset_index(drop=True)
        curr_TI_preds['Diff'] = curr_TI_preds['Pr('+SHAP_THRESHOLD+')'].diff()
        curr_TI_preds = curr_TI_preds[curr_TI_preds.Diff > 0].drop(columns=['Pr('+SHAP_THRESHOLD+')']).reset_index(drop=True)
        curr_TI_preds['Threshold'] = SHAP_THRESHOLD
        diff_values.append(curr_TI_preds)
diff_values = pd.concat(diff_values,ignore_index=True)

# Add a marker to designate cases above and below threshold
diff_values['Above'] = diff_values['Diff'] > 0

# Save calculated points of prognostic transition
diff_values.to_pickle(os.path.join(shap_dir,'all_transition_points.pkl'))

# Load calculated points of prognostic transition
diff_values = pd.read_pickle(os.path.join(shap_dir,'all_transition_points.pkl'))

# Filter to first week of ICU stay
diff_values = diff_values[diff_values.WindowIdx<=7].reset_index(drop=True)

### III. Partition significant transition points for parallel TimeSHAP calculation
## Partition evenly for parallel calculation
# Isolate unique partition-GUPI-tuning configuration-window index combinations
unique_transitions = diff_values[['REPEAT','FOLD','GUPI','TUNE_IDX','WindowIdx']].drop_duplicates().sort_values(by=['REPEAT','FOLD','TUNE_IDX','GUPI','WindowIdx']).reset_index(drop=True)

# Partition evenly along number of available array tasks
max_array_tasks = 10000
s = [unique_transitions.shape[0] // max_array_tasks for _ in range(max_array_tasks)]
s[:(unique_transitions.shape[0] - sum(s))] = [over+1 for over in s[:(unique_transitions.shape[0] - sum(s))]]    
end_idx = np.cumsum(s)
start_idx = np.insert(end_idx[:-1],0,0)
timeshap_partitions = [unique_transitions.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True) for idx in range(len(start_idx))]

# Merge all significant transitions into list of partitions
timeshap_partitions = [diff_values.merge(tp,how='inner') for tp in timeshap_partitions]

# Save derived partitions
cp.dump(timeshap_partitions, open(os.path.join(shap_dir,'timeSHAP_partitions.pkl'), "wb" ))

### IV. Calculate average training set outputs per tuning configuration
## Extract checkpoints for all top-performing tuning configurations
# Either create or load TILTomorrow checkpoint information for TimeSHAP calculation
if not os.path.exists(os.path.join(shap_dir,'ckpt_info.pkl')):
    
    # Find all model checkpoint files in APM output directory
    ckpt_files = []
    for path in Path(model_dir).rglob('*.ckpt'):
        ckpt_files.append(str(path.resolve()))

    # Categorize model checkpoint files based on name
    ckpt_info = pd.DataFrame({'file':ckpt_files,
                              'TUNE_IDX':[int(re.search('tune(.*)/epoch=', curr_file).group(1)) for curr_file in ckpt_files],
#                               'VERSION':[re.search('model_outputs/(.*)/fold', curr_file).group(1) for curr_file in ckpt_files],
                              'VERSION':[re.search('model_outputs/(.*)/repeat', curr_file).group(1) for curr_file in ckpt_files],
                              'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in ckpt_files],
                              'FOLD':[int(re.search('/fold(.*)/tune', curr_file).group(1)) for curr_file in ckpt_files],
#                               'VAL_ORC':[re.search('val_ORC=(.*).ckpt', curr_file).group(1) for curr_file in ckpt_files]
                              'VAL_LOSS':[re.search('val_loss=(.*).ckpt', curr_file).group(1) for curr_file in ckpt_files]
                             }).sort_values(by=['REPEAT','FOLD','TUNE_IDX','VERSION']).reset_index(drop=True)
    ckpt_info.VAL_LOSS = ckpt_info.VAL_LOSS.str.split('-').str[0].astype(float)
    
    # Isolate iterations that minimize loss   
    ckpt_info = ckpt_info.loc[ckpt_info.groupby(['TUNE_IDX','VERSION','REPEAT','FOLD']).VAL_LOSS.idxmin()].reset_index(drop=True)

    # Save model checkpoint information dataframe
    ckpt_info.to_pickle(os.path.join(shap_dir,'ckpt_info.pkl'))
    
else:
    
    # Read model checkpoint information dataframe
    ckpt_info = pd.read_pickle(os.path.join(shap_dir,'ckpt_info.pkl'))

# Filter checkpoints of top-performing model
ckpt_info = ckpt_info[ckpt_info.TUNE_IDX==135].reset_index(drop=True)

## Calculate and summarise training set outputs for each checkpoint
# Define variable to store summarised training set outputs
summ_train_preds = []

# Iterate through folds
for curr_fold in tqdm(ckpt_info.FOLD.unique(),'Iterating through folds to summarise training set outputs'):
    
    # Define current fold token subdirectory
    token_fold_dir = os.path.join(tokens_dir,'fold'+str(curr_fold))
    
    # Load current token-indexed training set
    training_set = pd.read_pickle(os.path.join(token_fold_dir,'training_indices.pkl'))
    
    # Filter training set outputs based on `WindowIdx`
    training_set = training_set[training_set.WindowIdx <= 84].reset_index(drop=True)
    
    # Iterate through tuning indices
    for curr_tune_idx in tqdm(ckpt_info[ckpt_info.FOLD==curr_fold].TUNE_IDX.unique(),'Iterating through tuning indices in fold '+str(curr_fold)+' to summarise training set outputs'):
        
        # Extract current file and required hyperparameter information
        curr_file = ckpt_info.file[(ckpt_info.FOLD==curr_fold)&(ckpt_info.TUNE_IDX==curr_tune_idx)].values[0]
        curr_time_tokens = tuning_grid.TIME_TOKENS[(tuning_grid.TUNE_IDX==curr_tune_idx)&(tuning_grid.FOLD==curr_fold)].values[0]
        
        # Format time tokens of index sets based on current tuning configuration
        format_training_set,time_tokens_mask = format_time_tokens(training_set.copy(),curr_time_tokens,True)
        
        # Add GOSE scores to training set
        format_training_set = pd.merge(format_training_set,study_GUPIs,how='left',on='GUPI')
        
        # Create PyTorch Dataset object
        train_Dataset = DYN_ALL_VARIABLE_SET(format_training_set,tuning_grid.OUTPUT_ACTIVATION[(tuning_grid.TUNE_IDX==curr_tune_idx)&(tuning_grid.FOLD==curr_fold)].values[0])
        
        # Create PyTorch DataLoader objects
        curr_train_DL = DataLoader(train_Dataset,
                                   batch_size=len(train_Dataset),
                                   shuffle=False,
                                   collate_fn=collate_batch)
        
        # Load current pretrained model
        ttm_model = TILTomorrow_model.load_from_checkpoint(curr_file)
        ttm_model.eval()
        
        # Calculate uncalibrated training set outputs
        with torch.no_grad():
            for i, (curr_train_label_list, curr_train_idx_list, curr_train_bin_offsets, curr_train_gupi_offsets, curr_train_gupis) in enumerate(curr_train_DL):
                (train_yhat, out_train_gupi_offsets) = ttm_model(curr_train_idx_list, curr_train_bin_offsets, curr_train_gupi_offsets)
                curr_train_labels = torch.cat([curr_train_label_list],dim=0).cpu().numpy()
                if tuning_grid.OUTPUT_ACTIVATION[(tuning_grid.TUNE_IDX==curr_tune_idx)&(tuning_grid.FOLD==curr_fold)].values[0] == 'softmax': 
                    curr_train_preds = pd.DataFrame(F.softmax(torch.cat([train_yhat.detach()],dim=0)).cpu().numpy(),columns=['Pr(TILBasic=1)','Pr(TILBasic=2/3)','Pr(TILBasic=4)','Pr(TILBasic=5)','Pr(TILBasic=6)','Pr(TILBasic=7)','Pr(TILBasic=8)'])
                    curr_train_preds['TrueLabel'] = curr_train_labels
                else:
                    raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")
                curr_train_preds.insert(loc=0, column='GUPI', value=curr_train_gupis)        
                curr_train_preds['TUNE_IDX'] = curr_tune_idx
        curr_train_preds['WindowIdx'] = curr_train_preds.groupby('GUPI').cumcount(ascending=True)+1
        
        # Calculate threshold-level probabilities of each output
        prob_cols = [col for col in curr_train_preds if col.startswith('Pr(TILBasic=')]
        thresh_labels = ['TILBasic>0','TILBasic>1','TILBasic>2','TILBasic>3']
        for thresh in range(1,len(prob_cols)):
            cols_gt = prob_cols[thresh:]
            prob_gt = curr_train_preds[cols_gt].sum(1).values
            curr_train_preds['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
            
        # Remove GOSE probability columns
        curr_train_preds = curr_train_preds.drop(columns=prob_cols).reset_index(drop=True)
        
        # Melt dataframe into long form
        curr_train_preds = curr_train_preds.melt(id_vars=['GUPI','TrueLabel','TUNE_IDX','WindowIdx'],var_name='Threshold',value_name='Probability')
        
        # Calculate average threshold probability per threshold and window index
        curr_summ_train_preds = curr_train_preds.groupby(['TUNE_IDX','Threshold','WindowIdx'],as_index=False)['Probability'].mean()
        curr_summ_train_preds['FOLD'] = curr_fold
        
        # Append dataframe to running list
        summ_train_preds.append(curr_summ_train_preds)
        
# Concatenate summarised output list into dataframe
summ_train_preds = pd.concat(summ_train_preds,ignore_index=True)

# Sort summarised output dataframe and reorganize columns
summ_train_preds = summ_train_preds.sort_values(by=['TUNE_IDX','FOLD','Threshold','WindowIdx']).reset_index(drop=True)
summ_train_preds = summ_train_preds[['TUNE_IDX','FOLD','Threshold','WindowIdx','Probability']]

# Save summarised training set outputs into TimeSHAP directory
summ_train_preds.to_pickle(os.path.join(shap_dir,'summarised_training_set_outputs.pkl'))

## Calculate "average event" outputs for TimeSHAP
# Define variable to store average-event outputs
avg_event_preds = []

# Extract unique partitions of cross-validation
uniq_partitions = ckpt_info[['REPEAT','FOLD']].drop_duplicates(ignore_index=True)

# Iterate through folds
for curr_cv_index in tqdm(range(uniq_partitions.shape[0]),'Iterating through unique cross validation partitions to calculate average event outputs'):
    
    # Extract current repeat and fold from index
    curr_repeat = uniq_partitions.REPEAT[curr_cv_index]
    curr_fold = uniq_partitions.FOLD[curr_cv_index]

    # Define current fold token subdirectory
    token_fold_dir = os.path.join(tokens_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold))
    
    # Load current token-indexed testing set
    testing_set = pd.read_pickle(os.path.join(token_fold_dir,'from_adm_strategy_abs_testing_indices.pkl'))
    
    # Filter testing set outputs based on `WindowIdx`
    testing_set = testing_set[testing_set.WindowIdx <= 84].reset_index(drop=True)
    
    # Load current token-indexed training set
    training_set = pd.read_pickle(os.path.join(token_fold_dir,'from_adm_strategy_abs_training_indices.pkl'))
    
    # Filter training set outputs based on `WindowIdx`
    training_set = training_set[training_set.WindowIdx <= 84].reset_index(drop=True)
    
    # Retrofit dataframes
    training_set = training_set.rename(columns={'VocabTimeFromAdmIndex':'VocabDaysSinceAdmIndex'})        
    testing_set = testing_set.rename(columns={'VocabTimeFromAdmIndex':'VocabDaysSinceAdmIndex'})

    # Load current token dictionary
    curr_vocab = cp.load(open(os.path.join(token_fold_dir,'from_adm_strategy_abs_token_dictionary.pkl'),"rb"))
    unknown_index = curr_vocab['<unk>']
    
    # Iterate through tuning indices
    for curr_tune_idx in tqdm(ckpt_info[(ckpt_info.REPEAT==curr_repeat)&(ckpt_info.FOLD==curr_fold)].TUNE_IDX.unique(),'Iterating through tuning indices in repeat '+str(curr_repeat)+', fold '+str(curr_fold)+' to calculate average event outputs'):
        
        # Extract current file and required hyperparameter information
        curr_file = ckpt_info.file[(ckpt_info.REPEAT==curr_repeat)&(ckpt_info.FOLD==curr_fold)&(ckpt_info.TUNE_IDX==curr_tune_idx)].values[0]
        curr_time_tokens = tuning_grid.TIME_TOKENS[(tuning_grid.TUNE_IDX==curr_tune_idx)].values[0]
        curr_rnn_type = tuning_grid.RNN_TYPE[(tuning_grid.TUNE_IDX==curr_tune_idx)].values[0]

        # Format time tokens of index sets based on current tuning configuration
        format_testing_set,time_tokens_mask = format_time_tokens(testing_set.copy(),curr_time_tokens,False)
        format_testing_set['SeqLength'] = format_testing_set.VocabIndex.apply(len)
        format_testing_set['Unknowns'] = format_testing_set.VocabIndex.apply(lambda x: x.count(unknown_index))        
        format_training_set,time_tokens_mask = format_time_tokens(training_set.copy(),curr_time_tokens,False)

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

        # Load current pretrained model
        ttm_model = TILTomorrow_model.load_from_checkpoint(curr_file)
        ttm_model.eval()
        
        ## First, calculate average output of expected value effect calcualation
        # Initialize custom TimeSHAP model for expected value effect calculation
        ts_GOSE_model = timeshap_TILTomorrow_model(ttm_model,curr_rnn_type,-1,unknown_index,average_event.shape[1]-len(curr_vocab))
        wrapped_gose_model = TorchModelWrapper(ts_GOSE_model)
        f_hs = lambda x, y=None: wrapped_gose_model.predict_last_hs(x, y)

        # Calculate average output on expected GOSE over time based on average event
        avg_score_over_len = get_avg_score_with_avg_event(f_hs, average_event, top=84)
        avg_score_over_len = pd.DataFrame(avg_score_over_len.items(),columns=['WindowIdx','Probability'])
        
        # Add metadata
        avg_score_over_len['Threshold'] = 'ExpectedValue'
        avg_score_over_len['REPEAT'] = curr_repeat
        avg_score_over_len['FOLD'] = curr_fold
        avg_score_over_len['TUNE_IDX'] = curr_tune_idx

        # Append dataframe to running list
        avg_event_preds.append(avg_score_over_len)

        # Calculate threshold-level probabilities of each output
        thresh_labels = ['TILBasic>0','TILBasic>1','TILBasic>2','TILBasic>3']
        for thresh in range(len(thresh_labels)):     
            
            # Initialize custom TimeSHAP model for threshold value effect calculation
            ts_GOSE_model = timeshap_TILTomorrow_model(ttm_model,curr_rnn_type,thresh,unknown_index,average_event.shape[1]-len(curr_vocab))
            wrapped_gose_model = TorchModelWrapper(ts_GOSE_model)
            f_hs = lambda x, y=None: wrapped_gose_model.predict_last_hs(x, y)

            # Calculate average output over time based on average event
            avg_score_over_len = get_avg_score_with_avg_event(f_hs, average_event, top=84)
            avg_score_over_len = pd.DataFrame(avg_score_over_len.items(),columns=['WindowIdx','Probability'])
            
            # Add metadata
            avg_score_over_len['Threshold'] = thresh_labels[thresh]
            avg_score_over_len['REPEAT'] = curr_repeat
            avg_score_over_len['FOLD'] = curr_fold
            avg_score_over_len['TUNE_IDX'] = curr_tune_idx
            
            # Append dataframe to running list
            avg_event_preds.append(avg_score_over_len)

# Concatenate average-event output list into dataframe
avg_event_preds = pd.concat(avg_event_preds,ignore_index=True)

# Sort average-event output dataframe and reorganize columns
avg_event_preds = avg_event_preds.sort_values(by=['TUNE_IDX','REPEAT','FOLD','Threshold','WindowIdx']).reset_index(drop=True)
avg_event_preds = avg_event_preds[['TUNE_IDX','REPEAT','FOLD','Threshold','WindowIdx','Probability']]

# Save average-event outputs into TimeSHAP directory
avg_event_preds.to_pickle(os.path.join(shap_dir,'average_event_outputs.pkl'))

# Load average-event outputs from TimeSHAP directory
avg_event_preds = pd.read_pickle(os.path.join(shap_dir,'average_event_outputs.pkl'))

# Summarise average-event outputs
summ_avg_event_preds = avg_event_preds.groupby(['TUNE_IDX','Threshold','WindowIdx'],as_index=False)['Probability'].aggregate({'Q1':lambda x: np.quantile(x,.25),'median':np.median,'Q3':lambda x: np.quantile(x,.75),'mean':np.mean,'std':np.std,'resamples':'count'}).reset_index(drop=True)

# Save summarised average-event predictors
summ_avg_event_preds.to_csv(os.path.join(shap_dir,'summarised_average_event_outputs.csv'),index=False)

### V. Determine distribution of signficant transitions over time and entropy
## Determine distribution of significant prognostic transitions over time
# Load significant points of prognostic transition
diff_values = pd.read_pickle(os.path.join(shap_dir,'significant_transition_points.pkl'))

# Remove significant transitions from the pre-calibrated zone
diff_values = diff_values[diff_values.WindowIdx > 4].reset_index(drop=True)

# Calculate count of number of transitions above and below threshold per window index
diff_values_over_time = diff_values.groupby(['WindowIdx','Above'],as_index=False).GUPI.count()

# Save count of significant transitions over time
diff_values_over_time.to_csv(os.path.join(shap_dir,'significant_transition_count_over_time.csv'),index=False)

## Calculate Shannon's Entropy over time
# Load compiled testing set outputs
test_outputs_df = pd.read_csv(os.path.join(model_dir,'compiled_test_outputs.csv'))

# Filter testing set outputs to top-performing model
test_outputs_df = test_outputs_df[test_outputs_df.TUNE_IDX==135].reset_index(drop=True)

# Calculate Shannon's Entropy based on predicted GOSE probability
prob_cols = [col for col in test_outputs_df if col.startswith('Pr(TILBasic=')]
test_outputs_df['Entropy'] = stats.entropy(test_outputs_df[prob_cols],axis=1,base=2)

# Summarise entropy values by `WindowIdx`
summarised_entropy = test_outputs_df.groupby('WindowIdx',as_index=False)['Entropy'].aggregate({'lo':lambda x: np.quantile(x,.025),'median':np.median,'hi':lambda x: np.quantile(x,.975),'mean':np.mean,'std':np.std,'resamples':'count'}).reset_index(drop=True)

# Save summarised entropy values
summarised_entropy.to_csv(os.path.join(model_dir,'summarised_entropy_values.csv'),index=False)

### VI. Summarise average output at each threshold over time
## Load and prepare compiled testing set outputs
# Load compiled testing set outputs
test_outputs_df = pd.read_csv(os.path.join(model_dir,'compiled_test_outputs.csv'))

# Filter testing set outputs to top-performing model
test_outputs_df = test_outputs_df[test_outputs_df.TUNE_IDX==135].reset_index(drop=True)

# Remove logit columns from dataframe
logit_cols = [col for col in test_outputs_df if col.startswith('z_TILBasic=')]
test_outputs_df = test_outputs_df.drop(columns=logit_cols).reset_index(drop=True)

# Calculate threshold-level probabilities
prob_cols = [col for col in test_outputs_df if col.startswith('Pr(TILBasic=')]
thresh_labels = ['TILBasic>0','TILBasic>1','TILBasic>2','TILBasic>3']
for thresh in range(1,len(prob_cols)):
    cols_gt = prob_cols[thresh:]
    prob_gt = test_outputs_df[cols_gt].sum(1).values
    gt = (test_outputs_df['TrueLabel'] >= thresh).astype(int).values
    test_outputs_df['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
    test_outputs_df[thresh_labels[thresh-1]] = gt

# Calculate from-discharge window indices
window_totals = test_outputs_df.groupby(['GUPI','TUNE_IDX','REPEAT','FOLD'],as_index=False).WindowIdx.aggregate({'WindowTotal':'max'})
test_outputs_df = test_outputs_df.merge(window_totals,how='left')
from_discharge_test_outputs_df = test_outputs_df.copy()
from_discharge_test_outputs_df['WindowIdx'] = from_discharge_test_outputs_df['WindowIdx'] - from_discharge_test_outputs_df['WindowTotal'] - 1

## Summarise probability values by window index and threshold
# Define probability threshold columns
prob_thresh_labels = ['Pr('+t+')' for t in thresh_labels]

# Extract relevant columns
test_outputs_df = test_outputs_df[['TUNE_IDX','REPEAT','FOLD','GUPI','WindowIdx']+prob_thresh_labels]
from_discharge_test_outputs_df = from_discharge_test_outputs_df[['TUNE_IDX','REPEAT','FOLD','GUPI','WindowIdx']+prob_thresh_labels]

# Melt dataframes to long form
test_outputs_df = test_outputs_df.melt(id_vars=['TUNE_IDX','REPEAT','FOLD','GUPI','WindowIdx'],value_vars=prob_thresh_labels,var_name='THRESHOLD',value_name='PROBABILITY',ignore_index=True)
from_discharge_test_outputs_df = from_discharge_test_outputs_df.melt(id_vars=['TUNE_IDX','REPEAT','FOLD','GUPI','WindowIdx'],value_vars=prob_thresh_labels,var_name='THRESHOLD',value_name='PROBABILITY',ignore_index=True)

# Summarise probability values
summ_test_preds_df = test_outputs_df.groupby(['TUNE_IDX','WindowIdx','THRESHOLD'],as_index=False).PROBABILITY.aggregate({'Q1':lambda x: np.quantile(x,.25),'median':np.median,'Q3':lambda x: np.quantile(x,.75),'mean':np.mean,'std':np.std,'resamples':'count'}).reset_index(drop=True)
summ_from_discharge_test_preds_df = from_discharge_test_outputs_df.groupby(['TUNE_IDX','WindowIdx','THRESHOLD'],as_index=False).PROBABILITY.aggregate({'Q1':lambda x: np.quantile(x,.25),'median':np.median,'Q3':lambda x: np.quantile(x,.75),'mean':np.mean,'std':np.std,'resamples':'count'}).reset_index(drop=True)

# Save summarised testing set output values
summ_test_preds_df.to_csv(os.path.join(model_dir,'summarised_test_outputs.csv'),index=False)
summ_from_discharge_test_preds_df.to_csv(os.path.join(model_dir,'summarised_from_discharge_test_outputs.csv'),index=False)