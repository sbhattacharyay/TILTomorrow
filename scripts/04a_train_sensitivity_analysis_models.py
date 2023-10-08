#### Master Script 4a: Train dynamic TILTomorrow models with focused input sets for sensitivity analysis ####
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
from classes.calibration import TemperatureScaling, VectorScaling
from functions.model_building import collate_batch
from models.dynamic_TTM import TILTomorrow_model

## Define parameters for model training
# Set version code
VERSION = 'v2-0'

# Choose tuning configurations for sensitivity analysis
OPT_TUNE_IDX = [332]

# Define combinations of variables to drop for sensitivity analysis
DROPOUT_VARS = ['dynamic','clinician_impressions','treatments','clinician_impressions_and_treatments']

## Define and create relevant directories
# Define directory in which tokens are stored for each partition
tokens_dir = '/home/sb2406/rds/hpc-work/tokens'

# Initialise model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)

# Define model performance directory based on version code
model_perf_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_performance',VERSION)

# Define directory in which calibration performance results are stored
calibration_dir = os.path.join(model_perf_dir,'calibration_performance')

## Load fundamental information for model training
# Load cross-validation splits of study population
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Isolate partitions
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

# Load prepared token dictionary
full_token_keys = pd.read_excel(os.path.join(tokens_dir,'TILTomorrow_full_token_keys_'+VERSION+'.xlsx'))
full_token_keys.Token = full_token_keys.Token.fillna('')
full_token_keys.BaseToken = full_token_keys.BaseToken.fillna('')

# Load best calibration slope combination information
best_cal_slopes_combos = pd.read_csv(os.path.join(calibration_dir,'best_calibration_configurations.csv'))

# Load post-dropout tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))

### II. Create grid of training combinations for sensitivity analysis
# If sensitivity analysis grid doesn't exist, create it
if not os.path.exists(os.path.join(model_dir,'sens_analysis_grid.csv')):
    
    # Extract hyperparameters from chosen tuning configurations
    filt_tuning_grid = tuning_grid[tuning_grid.TUNE_IDX.isin(OPT_TUNE_IDX)].drop(columns=['REPEAT','FOLD']).drop_duplicates(ignore_index=True)

    # Define and enumerate sensitivity analysis parameters
    sens_analysis_grid = pd.DataFrame({'SENS_IDX':[i+1 for i in range(len(DROPOUT_VARS))],'DROPOUT_VARS':DROPOUT_VARS})
    
    # Merge sensitivity analysis paramters and model hyperparameter dataframes
    sens_analysis_grid['key'] = 1
    filt_tuning_grid['key'] = 1
    sens_analysis_grid = sens_analysis_grid.merge(filt_tuning_grid,how='left')
    
    # Expand tuning grid per cross-validation folds
    partitions['key'] = 1
    sens_analysis_grid['key'] = 1
    sens_analysis_grid = sens_analysis_grid.merge(partitions,how='outer',on='key').drop(columns='key').sort_values(by=['REPEAT','FOLD','TUNE_IDX','SENS_IDX'],ignore_index=True)

    # Save tuning grid to model directory
    sens_analysis_grid.to_csv(os.path.join(model_dir,'sens_analysis_grid.csv'),index=False)

else:
    
    # Load optimised tuning grid
    sens_analysis_grid = pd.read_csv(os.path.join(model_dir,'sens_analysis_grid.csv'))
#     sens_analysis_grid = pd.read_csv(os.path.join(model_dir,'remaining_sens_analysis_grid.csv'))

## Manually determine repeats and folds to use for current training session
sens_analysis_grid = sens_analysis_grid[sens_analysis_grid.REPEAT.isin([1])].reset_index(drop=True)

### III. Train dynamic TILTomorrow model based on provided hyperparameter row index
# Argument-induced training functions
def main(array_task_id):
    
    # Extract current tuning grid parameters related to cross-validation and token preparation
    curr_sens_idx = sens_analysis_grid.SENS_IDX[array_task_id]
    curr_dropout_vars = sens_analysis_grid.DROPOUT_VARS[array_task_id]
    curr_repeat = sens_analysis_grid.REPEAT[array_task_id]
    curr_fold = sens_analysis_grid.FOLD[array_task_id]
    curr_batch_size = sens_analysis_grid.BATCH_SIZE[array_task_id]
    curr_window_limit = sens_analysis_grid.WINDOW_LIMIT[array_task_id]
    curr_tune_idx = sens_analysis_grid.TUNE_IDX[array_task_id]
    curr_base_token_representation = sens_analysis_grid.MIN_BASE_TOKEN_REPRESENATION[array_task_id]
    curr_max_tokens_per_base_token = sens_analysis_grid.MAX_TOKENS_PER_BASE_TOKEN[array_task_id]
    curr_physician_impressions = sens_analysis_grid.PHYS_IMPRESSION_TOKENS[array_task_id]
    curr_outcome_label = sens_analysis_grid.OUTCOME_LABEL[array_task_id]
    
    # Create a directory for the current fold
    fold_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(int(np.log10(sens_analysis_grid.FOLD.max()))+1))
    os.makedirs(fold_dir,exist_ok=True)
    
    # Create a directory for the current tuning index
    tune_dir = os.path.join(fold_dir,'tune'+str(curr_tune_idx).zfill(4))
    os.makedirs(tune_dir,exist_ok = True)
    
    # Create a directory for the current sensitivity analysis index
    sens_dir = os.path.join(tune_dir,'sens'+str(curr_sens_idx).zfill(4))
    os.makedirs(sens_dir,exist_ok = True)
    
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
    
    # Remove tokens corresponding to current dropout set
    if curr_dropout_vars == 'dynamic':
        banned_token_indices = np.sort(curr_vocab_df[~curr_vocab_df.Baseline].VocabIndex.unique())
    elif curr_dropout_vars == 'clinician_impressions':
        banned_token_indices = np.sort(curr_vocab_df[curr_vocab_df.ClinicianInput].VocabIndex.unique())
    elif curr_dropout_vars == 'treatments':
        banned_token_indices = np.sort(curr_vocab_df[curr_vocab_df.ICUIntervention].VocabIndex.unique())
    elif curr_dropout_vars == 'clinician_impressions_and_treatments':
        banned_token_indices = np.sort(curr_vocab_df[curr_vocab_df.ClinicianInput | curr_vocab_df.ICUIntervention].VocabIndex.unique())    
    training_set.VocabIndex = training_set.VocabIndex.apply(lambda x: list(set(x)-set(banned_token_indices)))
    validation_set.VocabIndex = validation_set.VocabIndex.apply(lambda x: list(set(x)-set(banned_token_indices)))
    testing_set.VocabIndex = testing_set.VocabIndex.apply(lambda x: list(set(x)-set(banned_token_indices)))
    
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
                              sens_analysis_grid.LATENT_DIM[array_task_id],
                              sens_analysis_grid.EMBED_DROPOUT[array_task_id],
                              sens_analysis_grid.RNN_TYPE[array_task_id],
                              sens_analysis_grid.HIDDEN_DIM[array_task_id],
                              sens_analysis_grid.RNN_LAYERS[array_task_id],
                              curr_outcome_label,
                              sens_analysis_grid.LEARNING_RATE[array_task_id],
                              True,
                              train_Dataset.y,
                              mask_indices+[0])
    
    early_stop_callback = EarlyStopping(
        monitor='val_metric',
        patience=sens_analysis_grid.ES_PATIENCE[array_task_id],
        mode='max'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_metric',
        dirpath=sens_dir,
        filename='{epoch:02d}-{val_metric:.2f}',
        save_top_k=1,
        mode='max'
    )
      
    csv_logger = pl.loggers.CSVLogger(save_dir=tune_dir,name='sens'+str(curr_sens_idx).zfill(4))

    trainer = pl.Trainer(gpus = 1,
                         accelerator='gpu',
                         logger = csv_logger,
                         max_epochs = sens_analysis_grid.NUM_EPOCHS[array_task_id],
                         enable_progress_bar = True,
                         enable_model_summary = True,
                         callbacks=[early_stop_callback,checkpoint_callback])
    
    trainer.fit(model=model,train_dataloaders=curr_train_DL,val_dataloaders=curr_val_DL)
    
    best_model = TILTomorrow_model.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()
    
    ## Calculate uncalibrated validation and testing set outputs
    # Calculate uncalibrated validation set outputs
    with torch.no_grad():
        for i, (curr_val_label_list, curr_val_idx_list, curr_val_bin_offsets, curr_val_gupi_offsets, curr_val_gupis) in enumerate(curr_val_DL):
            (val_yhat, out_val_gupi_offsets) = best_model(curr_val_idx_list, curr_val_bin_offsets, curr_val_gupi_offsets)
            curr_val_labels = torch.cat([curr_val_label_list],dim=0).cpu().numpy()
            if curr_outcome_label == 'TomorrowTILBasic': 
                curr_val_logits = torch.cat([val_yhat.detach()],dim=0).cpu().numpy()
                curr_val_probs = pd.DataFrame(F.softmax(torch.tensor(curr_val_logits)).cpu().numpy(),columns=['Pr(TILBasic=0)','Pr(TILBasic=1)','Pr(TILBasic=2)','Pr(TILBasic=3)','Pr(TILBasic=4)'])
                curr_val_outputs = pd.DataFrame(curr_val_logits,columns=['z_TILBasic=0','z_TILBasic=1','z_TILBasic=2','z_TILBasic=3','z_TILBasic=4'])
                curr_val_outputs = pd.concat([curr_val_outputs,curr_val_probs], axis=1)
                curr_val_outputs['TrueLabel'] = curr_val_labels
            elif curr_outcome_label == 'TomorrowHighIntensityTherapy':
                curr_val_logits = torch.cat([val_yhat.detach()],dim=0).cpu().numpy()
                curr_val_probs = pd.DataFrame(F.sigmoid(torch.tensor(curr_val_logits)).cpu().numpy(),columns=['Pr(HighTIL=1)'])
                curr_val_outputs = pd.DataFrame(curr_val_logits,columns=['z_HighTIL=1'])
                curr_val_outputs = pd.concat([curr_val_outputs,curr_val_probs], axis=1)
                curr_val_outputs['TrueLabel'] = curr_val_labels
            else:
                raise ValueError("Invalid outcome label. Must be 'TomorrowTILBasic' or 'TomorrowHighIntensityTherapy'")
            curr_val_outputs.insert(loc=0, column='GUPI', value=curr_val_gupis)        
            curr_val_outputs['TUNE_IDX'] = curr_tune_idx
            curr_val_outputs['WindowIdx'] = curr_val_outputs.groupby('GUPI').cumcount(ascending=True)+1

    # Calculate uncalibrated testing set outputs
    with torch.no_grad():
        for i, (curr_test_label_list, curr_test_idx_list, curr_test_bin_offsets, curr_test_gupi_offsets, curr_test_gupis) in enumerate(curr_test_DL):
            (test_yhat, out_test_gupi_offsets) = best_model(curr_test_idx_list, curr_test_bin_offsets, curr_test_gupi_offsets)
            curr_test_labels = torch.cat([curr_test_label_list],dim=0).cpu().numpy()
            if curr_outcome_label == 'TomorrowTILBasic': 
                curr_test_logits = torch.cat([test_yhat.detach()],dim=0).cpu().numpy()
                curr_test_probs = pd.DataFrame(F.softmax(torch.tensor(curr_test_logits)).cpu().numpy(),columns=['Pr(TILBasic=0)','Pr(TILBasic=1)','Pr(TILBasic=2)','Pr(TILBasic=3)','Pr(TILBasic=4)'])
                curr_test_outputs = pd.DataFrame(curr_test_logits,columns=['z_TILBasic=0','z_TILBasic=1','z_TILBasic=2','z_TILBasic=3','z_TILBasic=4'])
                curr_test_outputs = pd.concat([curr_test_outputs,curr_test_probs], axis=1)
                curr_test_outputs['TrueLabel'] = curr_test_labels
            elif curr_outcome_label == 'TomorrowHighIntensityTherapy':
                curr_test_logits = torch.cat([test_yhat.detach()],dim=0).cpu().numpy()
                curr_test_probs = pd.DataFrame(F.sigmoid(torch.tensor(curr_test_logits)).cpu().numpy(),columns=['Pr(HighTIL=1)'])
                curr_test_outputs = pd.DataFrame(curr_test_logits,columns=['z_HighTIL=1'])
                curr_test_outputs = pd.concat([curr_test_outputs,curr_test_probs], axis=1)
                curr_test_outputs['TrueLabel'] = curr_test_labels
            else:
                raise ValueError("Invalid outcome label. Must be 'TomorrowTILBasic' or 'TomorrowHighIntensityTherapy'")
            curr_test_outputs.insert(loc=0, column='GUPI', value=curr_test_gupis)        
            curr_test_outputs['TUNE_IDX'] = curr_tune_idx
            curr_test_outputs['WindowIdx'] = curr_test_outputs.groupby('GUPI').cumcount(ascending=True)+1

    ## Calibrate model outputs based on optimal methodology
    # Determine optimal calibration method per window index
    curr_calib_combos = best_cal_slopes_combos[(best_cal_slopes_combos.SET=='test')&(best_cal_slopes_combos.TUNE_IDX==curr_tune_idx)].reset_index(drop=True)
        
    # Create lists to store calibrated outputs
    calibrated_test_outputs = []
    
    # Add outputs in optimally uncalibrated window indices to lists
    calibrated_test_outputs.append(curr_test_outputs[(~curr_test_outputs.WindowIdx.isin(curr_calib_combos.WINDOW_IDX))|(curr_test_outputs.WindowIdx.isin(curr_calib_combos[curr_calib_combos.CALIBRATION=='None'].WINDOW_IDX))].reset_index(drop=True))
    
    # Extract names of important columns
    logit_cols = [col for col in curr_val_outputs if col.startswith('z_TILBasic=')]
    prob_cols = [col for col in curr_val_outputs if col.startswith('Pr(TILBasic=')]
    
    # Iterate through window indices in which calibration is optimal
    for curr_wi in curr_calib_combos[curr_calib_combos.CALIBRATION!='None'].WINDOW_IDX:
        
        # Extract current calibration parameters
        curr_optimization = curr_calib_combos[curr_calib_combos.WINDOW_IDX==curr_wi].OPTIMIZATION.values[0]
        curr_calibration = curr_calib_combos[curr_calib_combos.WINDOW_IDX==curr_wi].CALIBRATION.values[0]
        
        # Extract outputs at current window index
        curr_wi_val_outputs = curr_val_outputs[curr_val_outputs.WindowIdx==curr_wi].reset_index(drop=True)
        curr_wi_test_outputs = curr_test_outputs[curr_test_outputs.WindowIdx==curr_wi].reset_index(drop=True)
        
        # Create calibration object based on desired scaling type
        if curr_calibration == 'T':
            scale_object = TemperatureScaling(curr_wi_val_outputs[curr_wi_val_outputs.TrueLabel.notna()].reset_index(drop=True))
            scale_object.set_temperature(curr_optimization)
            with torch.no_grad():
                opt_temperature = scale_object.temperature.detach().item()
            if opt_temperature != opt_temperature:
                opt_temperature = 1    
            calib_test_logits = torch.tensor((curr_wi_test_outputs[logit_cols] / opt_temperature).values,dtype=torch.float32)
            calib_test_probs = F.softmax(calib_test_logits)

        elif curr_calibration == 'vector':
            scale_object = VectorScaling(curr_wi_val_outputs[curr_wi_val_outputs.TrueLabel.notna()].reset_index(drop=True))
            scale_object.set_vector(curr_optimization)
            with torch.no_grad():
                opt_vector = scale_object.vector.detach().data
                opt_biases = scale_object.biases.detach().data
            calib_test_logits = torch.matmul(torch.tensor(curr_wi_test_outputs[logit_cols].values,dtype=torch.float32),torch.diag_embed(opt_vector.squeeze(1))) + opt_biases.squeeze(1)
            calib_test_probs = F.softmax(calib_test_logits)

        else:
            raise ValueError("Invalid scaling type. Must be 'T' or 'vector'")
        
        # Properly format calibrated testing set outputs
        calib_test_outputs = pd.DataFrame(torch.cat([calib_test_logits,calib_test_probs],1).numpy(),columns=logit_cols+prob_cols)
        calib_test_outputs.insert(loc=0, column='GUPI', value=curr_wi_test_outputs['GUPI'])
        calib_test_outputs['TrueLabel'] = curr_wi_test_outputs['TrueLabel']
        calib_test_outputs['TUNE_IDX'] = curr_tune_idx
        calib_test_outputs['WindowIdx'] = curr_wi
        
        # Append formatted calibrated testing set output dataframe to running list
        calibrated_test_outputs.append(calib_test_outputs)
    
    # Concatenate calibrated testing set outputs and sort properly
    calibrated_test_outputs = pd.concat(calibrated_test_outputs,ignore_index=True).sort_values(by=['TUNE_IDX','GUPI','WindowIdx'],ignore_index=True)
    
    # Save calibrated testing set outputs
    calibrated_test_outputs.to_csv(os.path.join(sens_dir,'sens_analysis_calibrated_test_predictions.csv'),index=False)
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)
