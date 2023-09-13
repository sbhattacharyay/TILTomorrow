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

# Define function for collating indices from a training batch
def collate_batch(batch):
    (label_list, idx_list, bin_offsets, gupi_offsets, gupis) = ([], [], [0], [0], [])
    for (seq_lists, curr_GUPI, curr_label) in batch:
        gupi_offsets.append(len(seq_lists))
        for curr_bin_idx in range(len(seq_lists)):            
            label_list.append(curr_label[curr_bin_idx])
            gupis.append(curr_GUPI)
            processed_bin_seq = torch.tensor(seq_lists[curr_bin_idx],dtype=torch.int64)
            idx_list.append(processed_bin_seq)
            bin_offsets.append(processed_bin_seq.size(0))
    label_list = torch.tensor(label_list, dtype=torch.float)
    gupi_offsets = torch.tensor(gupi_offsets[:-1]).cumsum(dim=0)
    bin_offsets = torch.tensor(bin_offsets[:-1]).cumsum(dim=0)
    idx_list = torch.cat(idx_list)
    return (label_list, idx_list, bin_offsets, gupi_offsets, gupis)

# Define function for collecting uncalibrated outputs from a trained model
def calc_uncalib_outputs(ckpt_file, dl):
    
    # Load trained model from checkpoint file
    trained_model = TILTomorrow_model.load_from_checkpoint(ckpt_file)
    trained_model.eval()
    
    # Extract current set outputs based on outcome label type and provided dataloader object
    with torch.no_grad():
        for i, (curr_label_list, curr_idx_list, curr_bin_offsets, curr_gupi_offsets, curr_gupis) in enumerate(dl):
            (yhat, out_gupi_offsets) = trained_model(curr_idx_list, curr_bin_offsets, curr_gupi_offsets)
            curr_labels = torch.cat([curr_label_list],dim=0).cpu().numpy()
            if trained_model.outcome_name == 'TomorrowTILBasic': 
                curr_logits = torch.cat([yhat.detach()],dim=0).cpu().numpy()
                curr_probs = pd.DataFrame(F.softmax(torch.tensor(curr_logits)).cpu().numpy(),columns=['Pr(TILBasic=0)','Pr(TILBasic=1)','Pr(TILBasic=2)','Pr(TILBasic=3)','Pr(TILBasic=4)'])
                curr_outputs = pd.DataFrame(curr_logits,columns=['z_TILBasic=0','z_TILBasic=1','z_TILBasic=2','z_TILBasic=3','z_TILBasic=4'])
                curr_outputs = pd.concat([curr_outputs,curr_probs], axis=1)
                curr_outputs['TrueLabel'] = curr_labels
            elif trained_model.outcome_name == 'TomorrowHighIntensityTherapy':
                curr_logits = torch.cat([yhat.detach()],dim=0).cpu().numpy()
                curr_probs = pd.DataFrame(F.sigmoid(torch.tensor(curr_logits)).cpu().numpy(),columns=['Pr(HighTIL=1)'])
                curr_outputs = pd.DataFrame(curr_logits,columns=['z_HighTIL=1'])
                curr_outputs = pd.concat([curr_outputs,curr_probs], axis=1)
                curr_outputs['TrueLabel'] = curr_labels
            else:
                raise ValueError("Invalid outcome label. Must be 'TomorrowTILBasic' or 'TomorrowHighIntensityTherapy'")
            curr_outputs.insert(loc=0, column='GUPI', value=curr_gupis)        
            curr_outputs['WindowIdx'] = curr_outputs.groupby('GUPI').cumcount(ascending=True)+1
    
    # Return combined output dataframe
    return(curr_outputs)

# Define function for loading model outputs
def load_model_outputs(info_df, progress_bar=True, progress_bar_desc=''):
    
    compiled_predictions = []
        
    if progress_bar:
        iterator = tqdm(range(info_df.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(info_df.shape[0])
    
    # Load each output file, add 'WindowIdx' and repeat/fold information
    for curr_row in iterator:
        try:
            curr_preds = pd.read_csv(info_df.FILE[curr_row])
            curr_preds['REPEAT'] = info_df.REPEAT[curr_row]
            curr_preds['FOLD'] = info_df.FOLD[curr_row]
            curr_preds['SET'] = info_df.SET[curr_row]
            compiled_predictions.append(curr_preds)
        except:
            print("An exception occurred for file: "+info_df.FILE[curr_row])         
    return pd.concat(compiled_predictions,ignore_index=True)
