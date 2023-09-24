#### Master Script 03e: Assess calibration methods for remaining TILTomorrow configurations ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Create grid of calibration combinations
# III. Calibrate TILTomorrow model based on provided hyperparameter row index

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
from scipy.special import logit
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

# StatsModel methods
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

# TQDM for progress tracking
from tqdm import tqdm

# Custom methods
from classes.calibration import TemperatureScaling, VectorScaling
from functions.analysis import prepare_df, calc_ORC, calc_thresh_calibration

## Define parameters for model training
# Set version code
VERSION = 'v1-0'

# Define number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Window indices at which to calculate performance metrics
PERF_WINDOW_INDICES = [1,2,3,4,5,6,9,13,20]

## Define and create relevant directories
# Initialise model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)

# Define model performance directory based on version code
model_perf_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_performance',VERSION)

# Create directory to store calibration performance results
calibration_dir = os.path.join(model_perf_dir,'calibration_performance')
os.makedirs(calibration_dir,exist_ok=True)

## Load fundamental information for model training
# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

# Load the post-dropout tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'post_dropout_tuning_grid.csv'))

# Focus calibration efforts on TILBasic models
tuning_grid = tuning_grid[tuning_grid.OUTCOME_LABEL=='TomorrowTILBasic'].reset_index(drop=True)

# Extract unique list of remaining tuning configuration indices
remaining_tune_idx = tuning_grid.TUNE_IDX.unique()

# Load compiled uncalibrated validation set outputs
uncalib_val_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_val_uncalibrated_outputs.pkl'))

# Load compiled uncalibrated testing set outputs
uncalib_test_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_test_uncalibrated_outputs.pkl'))

### II. Create grid of calibration combinations
# If bootstrapping resamples for calibration don't exist, create them
if not os.path.exists(os.path.join(calibration_dir,'calibration_grid.csv')):

    # Create parameters for training differential token models
    calibration_parameters = {'TUNE_IDX':remaining_tune_idx,
                              'SCALING':['T','vector'],
                              'OPTIMIZATION':['nominal'],
                              'WINDOW_IDX':PERF_WINDOW_INDICES,
                              'REPEAT':[1],
                              'FOLD':list(range(1,6))}
    
    # Convert parameter dictionary to dataframe
    calibration_grid = pd.DataFrame([row for row in itertools.product(*calibration_parameters.values())],columns=calibration_parameters.keys()).sort_values(by=['TUNE_IDX','SCALING','OPTIMIZATION','REPEAT','FOLD','WINDOW_IDX'],ignore_index=True)

    # Save calibration grid to model directory
    calibration_grid.to_csv(os.path.join(calibration_dir,'calibration_grid.csv'),index=False)

else:
    # Load calibration grid
    calibration_grid = pd.read_csv(os.path.join(calibration_dir,'calibration_grid.csv'))

### III. Calibrate TILTomorrow model based on provided hyperparameter row index
# Argument-induced training functions
def main(array_task_id):

    # Extract current row informmation
    curr_tune_idx = calibration_grid.TUNE_IDX[array_task_id]
    curr_scaling = calibration_grid.SCALING[array_task_id]
    curr_optimization = calibration_grid.OPTIMIZATION[array_task_id]
    curr_window_idx = calibration_grid.WINDOW_IDX[array_task_id]
    curr_repeat = calibration_grid.REPEAT[array_task_id]
    curr_fold = calibration_grid.FOLD[array_task_id]

    # Define current tuning configuration directory
    tune_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(1),'tune'+str(curr_tune_idx).zfill(4))
    
    # Filter uncalibrated validation set outputs to specifications of current calibration combination
    uncalib_val_outputs = uncalib_val_outputs[(uncalib_val_outputs.TUNE_IDX == curr_tune_idx)&(uncalib_val_outputs.REPEAT == curr_repeat)&(uncalib_val_outputs.FOLD == curr_fold)].reset_index(drop=True)

    # Filter uncalibrated testing set outputs to specifications of current calibration combination
    uncalib_test_outputs = uncalib_test_outputs[(uncalib_test_outputs.TUNE_IDX == curr_tune_idx)&(uncalib_test_outputs.REPEAT == curr_repeat)&(uncalib_test_outputs.FOLD == curr_fold)].reset_index(drop=True)

    # Extract names of important columns
    logit_cols = [col for col in uncalib_val_outputs if col.startswith('z_TILBasic=')]
    prob_cols = [col for col in uncalib_val_outputs if col.startswith('Pr(TILBasic=')]

    # Calculate intermediate values for TILBasic validation set outputs
    prob_matrix = uncalib_val_outputs[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    index_vector = np.array(list(range(prob_matrix.shape[1])), ndmin=2).T
    uncalib_val_outputs['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    uncalib_val_outputs['PredLabel'] = prob_matrix.idxmax(axis=1)

    # Calculate intermediate values for TILBasic testing set outputs
    prob_matrix = uncalib_test_outputs[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    index_vector = np.array(list(range(prob_matrix.shape[1])), ndmin=2).T
    uncalib_test_outputs['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    uncalib_test_outputs['PredLabel'] = prob_matrix.idxmax(axis=1)

    # Prepare validation and testing set output dataframes for performance calculation
    filt_val_outputs = prepare_df(uncalib_val_outputs,PERF_WINDOW_INDICES)
    filt_test_outputs = prepare_df(uncalib_test_outputs,PERF_WINDOW_INDICES)

    # Filter validation and testing set outputs to current window index
    filt_val_outputs = filt_val_outputs[filt_val_outputs.WindowIdx == curr_window_idx].reset_index(drop=True)
    filt_test_outputs = filt_test_outputs[filt_test_outputs.WindowIdx == curr_window_idx].reset_index(drop=True)

    # Calculate pre-calibration validation and testing calibration metrics at each threshold
    pre_cal_val_thresh_calib = calc_thresh_calibration(filt_val_outputs,[curr_window_idx],True,'Calculating pre-calibration validation set calibration metrics')
    pre_cal_test_thresh_calib = calc_thresh_calibration(filt_test_outputs,[curr_window_idx],True,'Calculating pre-calibration testing set calibration metrics')

    # Calculate pre-calibration discrimination performance
    pre_cal_val_ORC = calc_ORC(filt_val_outputs,[curr_window_idx],True,'Calculating pre-calibration validation set ORC')
    pre_cal_test_ORC = calc_ORC(filt_test_outputs,[curr_window_idx],True,'Calculating pre-calibration testing set ORC')

    # Create calibration object based on desired scaling type
    if curr_scaling == 'T':
        scale_object = TemperatureScaling(filt_val_outputs[filt_val_outputs.TrueLabel.notna()].reset_index(drop=True))
        scale_object.set_temperature(curr_optimization)
        with torch.no_grad():
            opt_temperature = scale_object.temperature.detach().item()
        if opt_temperature != opt_temperature:
            opt_temperature = 1    
        calib_val_logits = torch.tensor((filt_val_outputs[logit_cols] / opt_temperature).values,dtype=torch.float32)
        calib_val_probs = F.softmax(calib_val_logits)
        calib_test_logits = torch.tensor((filt_test_outputs[logit_cols] / opt_temperature).values,dtype=torch.float32)
        calib_test_probs = F.softmax(calib_test_logits)
        
    elif curr_scaling == 'vector':
        scale_object = VectorScaling(filt_val_outputs[filt_val_outputs.TrueLabel.notna()].reset_index(drop=True))
        scale_object.set_vector(curr_optimization)
        with torch.no_grad():
            opt_vector = scale_object.vector.detach().data
            opt_biases = scale_object.biases.detach().data
        calib_val_logits = torch.matmul(torch.tensor(filt_val_outputs[logit_cols].values,dtype=torch.float32),torch.diag_embed(opt_vector.squeeze(1))) + opt_biases.squeeze(1)
        calib_val_probs = F.softmax(calib_val_logits)
        calib_test_logits = torch.matmul(torch.tensor(filt_test_outputs[logit_cols].values,dtype=torch.float32),torch.diag_embed(opt_vector.squeeze(1))) + opt_biases.squeeze(1)
        calib_test_probs = F.softmax(calib_test_logits)
        
    else:
        raise ValueError("Invalid scaling type. Must be 'T' or 'vector'")
    
    # Properly format calibrated validation set outputs
    calib_val_outputs = pd.DataFrame(torch.cat([calib_val_logits,calib_val_probs],1).numpy(),columns=logit_cols+prob_cols)
    calib_val_outputs.insert(loc=0, column='GUPI', value=filt_val_outputs['GUPI'])
    calib_val_outputs['TrueLabel'] = filt_val_outputs['TrueLabel']
    calib_val_outputs['TUNE_IDX'] = curr_tune_idx
    calib_val_outputs['WindowIdx'] = curr_window_idx
    calib_val_outputs['REPEAT'] = curr_repeat
    calib_val_outputs['FOLD'] = curr_fold

    # Save formatted calibrated validation set outputs
    calib_val_outputs.to_pickle(os.path.join(tune_dir,'set_val_opt_'+curr_optimization+'_window_idx_'+str(curr_window_idx).zfill(2)+'_scaling_'+curr_scaling+'.pkl'))

    # Properly format calibrated validation set outputs
    calib_test_outputs = pd.DataFrame(torch.cat([calib_test_logits,calib_test_probs],1).numpy(),columns=logit_cols+prob_cols)
    calib_test_outputs.insert(loc=0, column='GUPI', value=filt_test_outputs['GUPI'])
    calib_test_outputs['TrueLabel'] = filt_test_outputs['TrueLabel']
    calib_test_outputs['TUNE_IDX'] = curr_tune_idx
    calib_test_outputs['WindowIdx'] = curr_window_idx
    calib_test_outputs['REPEAT'] = curr_repeat
    calib_test_outputs['FOLD'] = curr_fold

    # Save formatted calibrated testing set outputs
    calib_test_outputs.to_pickle(os.path.join(tune_dir,'set_test_opt_'+curr_optimization+'_window_idx_'+str(curr_window_idx).zfill(2)+'_scaling_'+curr_scaling+'.pkl'))
    
    # Calculate post-calibration validation and testing calibration metrics at each threshold
    post_cal_val_thresh_calib = calc_thresh_calibration(calib_val_outputs,[curr_window_idx],True,'Calculating post-calibration validation set calibration metrics')
    post_cal_test_thresh_calib = calc_thresh_calibration(calib_test_outputs,[curr_window_idx],True,'Calculating post-calibration testing set calibration metrics')

    # Calculate intermediate values for calibrated TILBasic validation set outputs
    prob_matrix = calib_val_outputs[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    index_vector = np.array(list(range(prob_matrix.shape[1])), ndmin=2).T
    calib_val_outputs['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    calib_val_outputs['PredLabel'] = prob_matrix.idxmax(axis=1)

    # Calculate intermediate values for calibrated TILBasic testing set outputs
    prob_matrix = calib_test_outputs[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    index_vector = np.array(list(range(prob_matrix.shape[1])), ndmin=2).T
    calib_test_outputs['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    calib_test_outputs['PredLabel'] = prob_matrix.idxmax(axis=1)

    # Calculate post-calibration discrimination performance
    post_cal_val_ORC = calc_ORC(calib_val_outputs,[curr_window_idx],True,'Calculating post-calibration validation set ORC')
    post_cal_test_ORC = calc_ORC(calib_test_outputs,[curr_window_idx],True,'Calculating post-calibration testing set ORC')

    # Format pre-calibration threshold metrics 
    pre_cal_val_thresh_calib['SET'] = 'val'
    pre_cal_test_thresh_calib['SET'] = 'test'
    pre_cal_thresh_calib = pd.concat([pre_cal_val_thresh_calib,pre_cal_test_thresh_calib],ignore_index=True)
    pre_cal_thresh_calib['CALIBRATION'] = 'None'
    
    # Format post-calibration threshold metrics 
    post_cal_val_thresh_calib['SET'] = 'val'
    post_cal_test_thresh_calib['SET'] = 'test'
    post_cal_thresh_calib = pd.concat([post_cal_val_thresh_calib,post_cal_test_thresh_calib],ignore_index=True)
    post_cal_thresh_calib['CALIBRATION'] = curr_scaling

    # Concatenate pre- and post-calibration threshold metrics and calculate macro-averages
    thresh_calib = pd.concat([pre_cal_thresh_calib,post_cal_thresh_calib],ignore_index=True)
    ave_thresh_calib = thresh_calib.groupby(['TUNE_IDX','WINDOW_IDX','METRIC','SET','CALIBRATION'],as_index=False)['VALUE'].mean()
    ave_thresh_calib.insert(loc=0, column='THRESHOLD', value='Average')
    thresh_calib = pd.concat([thresh_calib,ave_thresh_calib],ignore_index=True)

    # Format pre-calibration ORC value dataframes
    pre_cal_val_ORC['SET'] = 'val'
    pre_cal_test_ORC['SET'] = 'test'
    pre_cal_ORC = pd.concat([pre_cal_val_ORC,pre_cal_test_ORC],ignore_index=True)
    pre_cal_ORC['CALIBRATION'] = 'None'

    # Format post-calibration ORC value dataframes
    post_cal_val_ORC['SET'] = 'val'
    post_cal_test_ORC['SET'] = 'test'
    post_cal_ORC = pd.concat([post_cal_val_ORC,post_cal_test_ORC],ignore_index=True)
    post_cal_ORC['CALIBRATION'] = curr_scaling

    # Concatenate pre- and post-calibration ORC values
    cal_ORC = pd.concat([pre_cal_ORC, post_cal_ORC],ignore_index=True)
    cal_ORC['THRESHOLD'] = 'None'

    # Concatenate metrics and sort
    metrics = pd.concat([thresh_calib,cal_ORC],ignore_index=True).sort_values(by=['METRIC','THRESHOLD','SET','CALIBRATION'],ignore_index=True)
    metrics.insert(loc=1, column='OPTIMIZATION', value=curr_optimization)
    metrics.insert(loc=3, column='REPEAT', value=curr_repeat)
    metrics.insert(loc=4, column='FOLD', value=curr_fold)

    # Create directory to store current pre- and post-calibration metrics
    file_dir = os.path.join(calibration_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(1))
    os.makedirs(file_dir,exist_ok=True)
    
    # Save compiled pre- and post-calibration metrics
    metrics.to_pickle(os.path.join(file_dir,'tune_idx_'+str(curr_tune_idx).zfill(4)+'_opt_'+curr_optimization+'_window_idx_'+str(curr_window_idx).zfill(2)+'_scaling_'+curr_scaling+'.pkl'))
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)