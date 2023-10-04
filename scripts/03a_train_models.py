#### Master Script 3a: Train dynamic all-variable-based TILTomorrow models ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Create grid of training combinations
# III. Train dynamic TILTomorrow model based on provided hyperparameter row index

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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# PyTorch, PyTorch.Text, and Lightning-PyTorch methods
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# TQDM for progress tracking
from tqdm import tqdm

# Custom methods
from classes.datasets import DYN_ALL_VARIABLE_SET
from functions.model_building import collate_batch
from models.dynamic_TTM import TILTomorrow_model

## Define parameters for model training
# Set version code
VERSION = 'v2-0'

## Define and create relevant directories
# Define directory in which tokens are stored for each partition
tokens_dir = '/home/sb2406/rds/hpc-work/tokens'

# Initialise model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)
os.makedirs(model_dir,exist_ok=True)

## Load fundamental information for model training
# Load cross-validation splits of study population
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Isolate partitions
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

# Load prepared token dictionary
full_token_keys = pd.read_excel(os.path.join(tokens_dir,'TILTomorrow_full_token_keys_'+VERSION+'.xlsx'))
full_token_keys.Token = full_token_keys.Token.fillna('')
full_token_keys.BaseToken = full_token_keys.BaseToken.fillna('')

# II. Create grid of training combinations
# If tuning grid doesn't exist, create it
if not os.path.exists(os.path.join(model_dir,'tuning_grid.csv')):

    # Create parameters for training token models
    tuning_parameters = {'WINDOW_LIMIT':[6,13],
                         'RNN_TYPE':['LSTM','GRU'],
                         'LATENT_DIM':[256,512,1024],
                         'HIDDEN_DIM':[128,256,512],
                         'TOKEN_CUTS':[20],
                         'MIN_BASE_TOKEN_REPRESENATION':['None','0.05'],
                         'MAX_TOKENS_PER_BASE_TOKEN':['None','100'],
                         'PHYS_IMPRESSION_TOKENS':[True],
                         'EMBED_DROPOUT':[.2],
                         'RNN_LAYERS':[1],
                         'NUM_EPOCHS':[200],
                         'ES_PATIENCE':[30],
                         'IMBALANCE_CORRECTION':['weights'],
                         'OUTCOME_LABEL':['TomorrowTILBasic'],
                         'LEARNING_RATE':[0.001],
                         'BATCH_SIZE':[1]}
    
    # Convert parameter dictionary to dataframe
    tuning_grid = pd.DataFrame([row for row in itertools.product(*tuning_parameters.values())],columns=tuning_parameters.keys())
    
    # Assign tuning indices based on version number
    if (VERSION == 'v1-0'):
        tuning_grid['TUNE_IDX'] = list(range(1,tuning_grid.shape[0]+1))

    else:
        # Find all existing tuning grids
        existing_tuning_grids = []
        for path in Path(os.path.join(model_dir,'..')).rglob('tuning_grid.csv'):
            existing_tuning_grids.append(str(path.resolve()))

        # Characterise existing tuning grids
        existing_tuning_grids = pd.DataFrame({'FILE':existing_tuning_grids,
                                              'VERSION':[re.search('_model_outputs/(.*)/tuning_grid.csv', curr_file).group(1) for curr_file in existing_tuning_grids]
                                              }).sort_values(by=['VERSION','FILE']).reset_index(drop=True)
        
        # Encode version as number for comparison
        existing_tuning_grids['NUM_VERSION'] = existing_tuning_grids['VERSION'].str.replace('-','.').str.replace('v','').astype(float)

        # Load latest tuning grid
        latest_tuning_grid = pd.read_csv(existing_tuning_grids.FILE.loc[existing_tuning_grids.NUM_VERSION.idxmax()])

        # Extract unique tuning configurations from latest version
        latest_tuning_grid = latest_tuning_grid.drop(columns=['REPEAT','FOLD']).drop_duplicates(ignore_index=True)

        # Last max tuning index
        last_max_tune_idx = latest_tuning_grid.TUNE_IDX.max()

        # First identify existing configurations from previous tuning grid
        tuning_grid = tuning_grid.merge(latest_tuning_grid,how='left')

        # Add new tuning indices
        tuning_grid.TUNE_IDX[tuning_grid.TUNE_IDX.isna()] = list(range(last_max_tune_idx+1,tuning_grid.TUNE_IDX[tuning_grid.TUNE_IDX.isna()].shape[0]+last_max_tune_idx+1))

        # Cast tuning index as integer
        tuning_grid.TUNE_IDX = tuning_grid.TUNE_IDX.astype(int)

        # Sort tuning grid by tuning index
        tuning_grid = tuning_grid.sort_values(by='TUNE_IDX',ignore_index=True)
    
    # Reorder tuning grid columns
    tuning_grid = tuning_grid[['TUNE_IDX','WINDOW_LIMIT','LATENT_DIM','HIDDEN_DIM','TOKEN_CUTS','MIN_BASE_TOKEN_REPRESENATION','MAX_TOKENS_PER_BASE_TOKEN','PHYS_IMPRESSION_TOKENS','RNN_TYPE','EMBED_DROPOUT','RNN_LAYERS','NUM_EPOCHS','ES_PATIENCE','IMBALANCE_CORRECTION','OUTCOME_LABEL','LEARNING_RATE','BATCH_SIZE']].reset_index(drop=True)
    
    # Expand tuning grid per cross-validation folds
    partitions['key'] = 1
    tuning_grid['key'] = 1
    tuning_grid = tuning_grid.merge(partitions,how='outer',on='key').drop(columns='key').reset_index(drop=True)

    # Save tuning grid to model directory
    tuning_grid.to_csv(os.path.join(model_dir,'tuning_grid.csv'),index=False)

else:
    
    # Load optimised tuning grid
    tuning_grid = pd.read_csv(os.path.join(model_dir,'tuning_grid.csv'))

## Manually determine repeats and folds to use for current training session
tuning_grid = tuning_grid[tuning_grid.REPEAT.isin([1])].reset_index(drop=True)

### III. Train dynamic TILTomorrow model based on provided hyperparameter row index
# Argument-induced training functions
def main(array_task_id):
    
    # Extract current tuning grid parameters related to cross-validation and token preparation
    curr_repeat = tuning_grid.REPEAT[array_task_id]
    curr_fold = tuning_grid.FOLD[array_task_id]
    curr_batch_size = tuning_grid.BATCH_SIZE[array_task_id]
    curr_window_limit = tuning_grid.WINDOW_LIMIT[array_task_id]
    curr_tune_idx = tuning_grid.TUNE_IDX[array_task_id]
    curr_base_token_representation = tuning_grid.MIN_BASE_TOKEN_REPRESENATION[array_task_id]
    curr_max_tokens_per_base_token = tuning_grid.MAX_TOKENS_PER_BASE_TOKEN[array_task_id]
    curr_physician_impressions = tuning_grid.PHYS_IMPRESSION_TOKENS[array_task_id]
    curr_outcome_label = tuning_grid.OUTCOME_LABEL[array_task_id]
    
    # Create a directory for the current fold
    fold_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(int(np.log10(tuning_grid.FOLD.max()))+1))
    os.makedirs(fold_dir,exist_ok=True)
    
    # Create a directory for the current tuning index
    tune_dir = os.path.join(fold_dir,'tune'+str(curr_tune_idx).zfill(4))
    os.makedirs(tune_dir,exist_ok = True)
    
    # Initialize a variable to store the token subdirectory of the current fold
    token_fold_dir = os.path.join(tokens_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(1))

    # Load current token-indexed training and testing sets
    training_set = pd.read_pickle(os.path.join(token_fold_dir,'TILTomorrow_training_indices.pkl'))
    validation_set = pd.read_pickle(os.path.join(token_fold_dir,'TILTomorrow_validation_indices.pkl'))
    testing_set = pd.read_pickle(os.path.join(token_fold_dir,'TILTomorrow_testing_indices.pkl'))
    
    # Define the limit of windows for model training (1 WINDOW = 24 HOURS)
    if curr_window_limit != 'None':
        training_set = training_set[training_set.WindowIdx <= int(curr_window_limit)].sort_values(by=['GUPI','WindowIdx'],ignore_index=True)

    # Load current token dictionary
    curr_vocab = cp.load(open(os.path.join(token_fold_dir,'TILTomorrow_token_dictionary.pkl'),"rb"))

    # Create dataframe version of vocabulary
    curr_vocab_df = pd.DataFrame({'VocabIndex':list(range(len(curr_vocab))),'Token':curr_vocab.get_itos()})

    # Merge token dictionary information onto current vocabulary
    curr_vocab_df = curr_vocab_df.merge(full_token_keys,how='left')

    # Create a list to store masked indices for current training run
    mask_indices = []

    # If there is a maximum on the number of tokens per base token, remove base tokens which violate this limit
    if curr_max_tokens_per_base_token != 'None':
        tokens_per_base_token = full_token_keys[(~full_token_keys.Missing)&(~full_token_keys.BaseToken.isin(['','<unk>','DayOfICUStay']))].groupby(['BaseToken'],as_index=False).Token.nunique()
        base_tokens_to_mask = tokens_per_base_token.BaseToken[tokens_per_base_token.Token > int(curr_max_tokens_per_base_token)].unique()
        mask_indices += curr_vocab_df[curr_vocab_df.BaseToken.isin(base_tokens_to_mask)].VocabIndex.unique().tolist()

    # If there is a minimum on the number of patients needed per base token, remove base tokens which violate this limit
    if curr_base_token_representation != 'None':
        token_counts_per_patient = pd.read_pickle(os.path.join(token_fold_dir,'TILTomorrow_token_incidences_per_patient.pkl')).merge(full_token_keys[['Token','BaseToken']],how='left')
        token_counts_per_patient = token_counts_per_patient[token_counts_per_patient.GUPI.isin(training_set.GUPI.unique())].reset_index(drop=True)
        patient_counts_per_base_token = token_counts_per_patient.groupby('BaseToken',as_index=False).GUPI.nunique()
        base_tokens_to_mask = patient_counts_per_base_token.BaseToken[patient_counts_per_base_token.GUPI<(float(curr_base_token_representation)*training_set.GUPI.nunique())].unique()
        mask_indices += curr_vocab_df[curr_vocab_df.BaseToken.isin(base_tokens_to_mask)].VocabIndex.unique().tolist()

    # Format tokens based on physician-impression decision
    if curr_physician_impressions:
        training_set.VocabIndex = training_set.VocabIndex + training_set.VocabPhysImpressionIndex
        validation_set.VocabIndex = validation_set.VocabIndex + validation_set.VocabPhysImpressionIndex
        testing_set.VocabIndex = testing_set.VocabIndex + testing_set.VocabPhysImpressionIndex
    else:
        mask_indices += training_set.VocabPhysImpressionIndex.explode().unique().tolist()

    # Ensure indices are unique
    training_set.VocabIndex = training_set.VocabIndex.apply(lambda x: np.unique(x).tolist())
    validation_set.VocabIndex = validation_set.VocabIndex.apply(lambda x: np.unique(x).tolist())
    testing_set.VocabIndex = testing_set.VocabIndex.apply(lambda x: np.unique(x).tolist())
    mask_indices = np.unique(mask_indices).tolist()

    # Create PyTorch Dataset objects
    train_Dataset = DYN_ALL_VARIABLE_SET(training_set,curr_outcome_label)
    val_Dataset = DYN_ALL_VARIABLE_SET(validation_set,curr_outcome_label)
    test_Dataset = DYN_ALL_VARIABLE_SET(testing_set,curr_outcome_label)

    # Create PyTorch DataLoader objects
    curr_train_DL = DataLoader(train_Dataset,
                               batch_size=int(curr_batch_size),
                               shuffle=True,
                               collate_fn=collate_batch)
    
    curr_val_DL = DataLoader(val_Dataset,
                             batch_size=len(val_Dataset), 
                             shuffle=False,
                             collate_fn=collate_batch)
    
    curr_test_DL = DataLoader(test_Dataset,
                              batch_size=len(test_Dataset),
                              shuffle=False,
                              collate_fn=collate_batch)
    
    # Initialize current model class based on hyperparameter selections
    model = TILTomorrow_model(len(curr_vocab),
                              tuning_grid.LATENT_DIM[array_task_id],
                              tuning_grid.EMBED_DROPOUT[array_task_id],
                              tuning_grid.RNN_TYPE[array_task_id],
                              tuning_grid.HIDDEN_DIM[array_task_id],
                              tuning_grid.RNN_LAYERS[array_task_id],
                              curr_outcome_label,
                              tuning_grid.LEARNING_RATE[array_task_id],
                              True,
                              train_Dataset.y,
                              mask_indices+[0])
    
    early_stop_callback = EarlyStopping(
        monitor='val_metric',
        patience=tuning_grid.ES_PATIENCE[array_task_id],
        mode='max'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_metric',
        dirpath=tune_dir,
        filename='{epoch:02d}-{val_metric:.2f}',
        save_top_k=1,
        mode='max'
    )
      
    csv_logger = pl.loggers.CSVLogger(save_dir=fold_dir,name='tune'+str(curr_tune_idx).zfill(4))

    trainer = pl.Trainer(gpus = 1,
                         accelerator='gpu',
                         logger = csv_logger,
                         max_epochs = tuning_grid.NUM_EPOCHS[array_task_id],
                         enable_progress_bar = True,
                         enable_model_summary = True,
                         callbacks=[early_stop_callback,checkpoint_callback])
    
    trainer.fit(model=model,train_dataloaders=curr_train_DL,val_dataloaders=curr_val_DL)
    
    best_model = TILTomorrow_model.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()
    
    ## Calculate and save uncalibrated validation set
    with torch.no_grad():
        for i, (curr_val_label_list, curr_val_idx_list, curr_val_bin_offsets, curr_val_gupi_offsets, curr_val_gupis) in enumerate(curr_val_DL):
            (val_yhat, out_val_gupi_offsets) = best_model(curr_val_idx_list, curr_val_bin_offsets, curr_val_gupi_offsets)
            curr_val_labels = torch.cat([curr_val_label_list],dim=0).cpu().numpy()
            if curr_outcome_label == 'TomorrowTILBasic': 
                curr_val_logits = torch.cat([val_yhat.detach()],dim=0).cpu().numpy()
                curr_val_probs = pd.DataFrame(F.softmax(torch.tensor(curr_val_logits)).cpu().numpy(),columns=['Pr(TILBasic=0)','Pr(TILBasic=1)','Pr(TILBasic=2)','Pr(TILBasic=3)','Pr(TILBasic=4)'])
                curr_val_preds = pd.DataFrame(curr_val_logits,columns=['z_TILBasic=0','z_TILBasic=1','z_TILBasic=2','z_TILBasic=3','z_TILBasic=4'])
                curr_val_preds = pd.concat([curr_val_preds,curr_val_probs], axis=1)
                curr_val_preds['TrueLabel'] = curr_val_labels
            elif curr_outcome_label == 'TomorrowHighIntensityTherapy':
                curr_val_logits = torch.cat([val_yhat.detach()],dim=0).cpu().numpy()
                curr_val_probs = pd.DataFrame(F.sigmoid(torch.tensor(curr_val_logits)).cpu().numpy(),columns=['Pr(HighTIL=1)'])
                curr_val_preds = pd.DataFrame(curr_val_logits,columns=['z_HighTIL=1'])
                curr_val_preds = pd.concat([curr_val_preds,curr_val_probs], axis=1)
                curr_val_preds['TrueLabel'] = curr_val_labels
            else:
                raise ValueError("Invalid outcome label. Must be 'TomorrowTILBasic' or 'TomorrowHighIntensityTherapy'")
            curr_val_preds.insert(loc=0, column='GUPI', value=curr_val_gupis)        
            curr_val_preds['TUNE_IDX'] = curr_tune_idx
            curr_val_preds['WindowIdx'] = curr_val_preds.groupby('GUPI').cumcount(ascending=True)+1
            curr_val_preds.to_csv(os.path.join(tune_dir,'uncalibrated_val_predictions.csv'),index=False)

    ## Calculate and save uncalibrated testing set
    with torch.no_grad():
        for i, (curr_test_label_list, curr_test_idx_list, curr_test_bin_offsets, curr_test_gupi_offsets, curr_test_gupis) in enumerate(curr_test_DL):
            (test_yhat, out_test_gupi_offsets) = best_model(curr_test_idx_list, curr_test_bin_offsets, curr_test_gupi_offsets)
            curr_test_labels = torch.cat([curr_test_label_list],dim=0).cpu().numpy()
            if curr_outcome_label == 'TomorrowTILBasic': 
                curr_test_logits = torch.cat([test_yhat.detach()],dim=0).cpu().numpy()
                curr_test_probs = pd.DataFrame(F.softmax(torch.tensor(curr_test_logits)).cpu().numpy(),columns=['Pr(TILBasic=0)','Pr(TILBasic=1)','Pr(TILBasic=2)','Pr(TILBasic=3)','Pr(TILBasic=4)'])
                curr_test_preds = pd.DataFrame(curr_test_logits,columns=['z_TILBasic=0','z_TILBasic=1','z_TILBasic=2','z_TILBasic=3','z_TILBasic=4'])
                curr_test_preds = pd.concat([curr_test_preds,curr_test_probs], axis=1)
                curr_test_preds['TrueLabel'] = curr_test_labels
            elif curr_outcome_label == 'TomorrowHighIntensityTherapy':
                curr_test_logits = torch.cat([test_yhat.detach()],dim=0).cpu().numpy()
                curr_test_probs = pd.DataFrame(F.sigmoid(torch.tensor(curr_test_logits)).cpu().numpy(),columns=['Pr(HighTIL=1)'])
                curr_test_preds = pd.DataFrame(curr_test_logits,columns=['z_HighTIL=1'])
                curr_test_preds = pd.concat([curr_test_preds,curr_test_probs], axis=1)
                curr_test_preds['TrueLabel'] = curr_test_labels
            else:
                raise ValueError("Invalid outcome label. Must be 'TomorrowTILBasic' or 'TomorrowHighIntensityTherapy'")
            curr_test_preds.insert(loc=0, column='GUPI', value=curr_test_gupis)        
            curr_test_preds['TUNE_IDX'] = curr_tune_idx
            curr_test_preds['WindowIdx'] = curr_test_preds.groupby('GUPI').cumcount(ascending=True)+1
            curr_test_preds.to_csv(os.path.join(tune_dir,'uncalibrated_test_predictions.csv'),index=False)

    # Extract names of important columns
    logit_cols = [col for col in curr_val_preds if col.startswith('z_')]
    prob_cols = [col for col in curr_val_preds if col.startswith('Pr(')]
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)