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
import seaborn as sns
import multiprocessing
from scipy import stats
from pathlib import Path
from ast import literal_eval
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter, OrderedDict
from pandas.api.types import is_integer_dtype, is_float_dtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

from tqdm import tqdm

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Function to categorize tokens under certain conditions
def categorizer(x,threshold=20):
    if is_integer_dtype(x) & (len(x.unique()) <= threshold):
        new_x = x.astype(str).str.zfill(3)
        new_x[new_x == 'nan'] = np.nan
        return new_x
    elif is_float_dtype(x) & (len(x.unique()) <= threshold):
        new_x = x.astype(str).str.replace('.','dec',regex=False)
        new_x[new_x.str.endswith('dec0')] = new_x[new_x.str.endswith('dec0')].str.replace('dec0','',regex=False)
        new_x = new_x.str.zfill(3)
        new_x[new_x == 'nan'] = np.nan
        return new_x
    else:
        return x

# Function to clean categorical token rows
def clean_token_rows(tokens_df_slice,progress_bar=True,progress_bar_desc=''):    
    
    if progress_bar:
        iterator = tqdm(range(tokens_df_slice.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(tokens_df_slice.shape[0])
    
    for curr_row in iterator:
        curr_token_set = tokens_df_slice.TOKENS.iloc[curr_row]
        cleaned_token_set = ' '.join(np.sort(np.unique(curr_token_set.split())))
        tokens_df_slice.TOKENS.iloc[curr_row] = cleaned_token_set

        curr_phys_impression_token_set = tokens_df_slice.PHYSIMPRESSIONTOKENS.iloc[curr_row]
        cleaned_phys_impression_token_set = ' '.join(np.sort(np.unique(curr_phys_impression_token_set.split())))
        tokens_df_slice.PHYSIMPRESSIONTOKENS.iloc[curr_row] = cleaned_phys_impression_token_set
        
    return tokens_df_slice

# Function to characterise tokens of each study window
def get_token_info(index_df,vocab_df,missing = True, progress_bar=True, progress_bar_desc=''):
    
    compiled_token_characteristics = []
    
    if progress_bar:
        iterator = tqdm(range(index_df.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(index_df.shape[0])
    
    for curr_row in iterator:
        
        curr_IndexList = index_df.VocabIndex[curr_row]
        
        if np.isnan(curr_IndexList).all():
            compiled_token_characteristics.append(pd.DataFrame({'GUPI':index_df.GUPI[curr_row],
                                                                'TimeStampStart':index_df.TimeStampStart[curr_row],
                                                                'TimeStampEnd':index_df.TimeStampEnd[curr_row],
                                                                'WindowIdx':index_df.WindowIdx[curr_row],
                                                                'WindowTotal':index_df.WindowTotal[curr_row],
                                                                'Set':index_df.Set[curr_row],
                                                                'TotalTokens':0,
                                                                'UnknownNonMissing':0,
                                                                'Numeric':0,
                                                                'Baseline':0,
                                                                'Discharge':0,
                                                                'Ordered':0,
                                                                'Binary':0,
                                                                'ICUIntervention':0,
                                                                'ClinicianInput':0},index=[0]))
        elif (len(curr_IndexList) == 1):
            filt_vocab = vocab_df[vocab_df.VocabIndex.isin(curr_IndexList)].reset_index(drop=True)
            
            if not missing:
                filt_vocab = filt_vocab[~filt_vocab.Missing].reset_index(drop=True)
            
            if (filt_vocab.shape[0] == 0):
                compiled_token_characteristics.append(pd.DataFrame({'GUPI':index_df.GUPI[curr_row],
                                                                    'TimeStampStart':index_df.TimeStampStart[curr_row],
                                                                    'TimeStampEnd':index_df.TimeStampEnd[curr_row],
                                                                    'WindowIdx':index_df.WindowIdx[curr_row],
                                                                    'WindowTotal':index_df.WindowTotal[curr_row],
                                                                    'Set':index_df.Set[curr_row],
                                                                    'TotalTokens':0,
                                                                    'UnknownNonMissing':0,
                                                                    'Numeric':0,
                                                                    'Baseline':0,
                                                                    'Discharge':0,
                                                                    'Ordered':0,
                                                                    'Binary':0,
                                                                    'ICUIntervention':0,
                                                                    'ClinicianInput':0},index=[0]))
            else:
                compiled_token_characteristics.append(pd.DataFrame({'GUPI':index_df.GUPI[curr_row],
                                                                    'TimeStampStart':index_df.TimeStampStart[curr_row],
                                                                    'TimeStampEnd':index_df.TimeStampEnd[curr_row],
                                                                    'WindowIdx':index_df.WindowIdx[curr_row],
                                                                    'WindowTotal':index_df.WindowTotal[curr_row],
                                                                    'Set':index_df.Set[curr_row],
                                                                    'TotalTokens':filt_vocab.shape[0],
                                                                    'UnknownNonMissing':int(filt_vocab.UnknownNonMissing[0]),
                                                                    'Numeric':int(filt_vocab.Numeric[0]),
                                                                    'Baseline':int(filt_vocab.Baseline[0]),
                                                                    'Discharge':int(filt_vocab.Discharge[0]),
                                                                    'Ordered':int(filt_vocab.Ordered[0]),
                                                                    'Binary':int(filt_vocab.Binary[0]),
                                                                    'ICUIntervention':int(filt_vocab.ICUIntervention[0]),
                                                                    'ClinicianInput':int(filt_vocab.ClinicianInput[0])},index=[0]))
            
        else:
            filt_vocab = vocab_df[vocab_df.VocabIndex.isin(curr_IndexList)].reset_index(drop=True)

            if not missing:
                filt_vocab = filt_vocab[~filt_vocab.Missing].reset_index(drop=True)

            compiled_token_characteristics.append(pd.DataFrame({'GUPI':index_df.GUPI[curr_row],
                                                                'TimeStampStart':index_df.TimeStampStart[curr_row],
                                                                'TimeStampEnd':index_df.TimeStampEnd[curr_row],
                                                                'WindowIdx':index_df.WindowIdx[curr_row],
                                                                'WindowTotal':index_df.WindowTotal[curr_row],
                                                                'Set':index_df.Set[curr_row],
                                                                'TotalTokens':filt_vocab.shape[0],
                                                                'UnknownNonMissing':filt_vocab.UnknownNonMissing.sum(),
                                                                'Numeric':filt_vocab.Numeric.sum(),
                                                                'Baseline':filt_vocab.Baseline.sum(),
                                                                'Discharge':filt_vocab.Discharge.sum(),
                                                                'Ordered':filt_vocab.Ordered.sum(),
                                                                'Binary':filt_vocab.Binary.sum(),
                                                                'ICUIntervention':filt_vocab.ICUIntervention.sum(),
                                                                'ClinicianInput':filt_vocab.ClinicianInput.sum()},index=[0]))
    return pd.concat(compiled_token_characteristics,ignore_index=True)

# Function to count specific token incidences of each study window
def count_token_incidences(index_df,curr_vocab,vocab_df,missing = True, progress_bar=True, progress_bar_desc=''):
    
    compiled_token_incidences = []
    
    if progress_bar:
        iterator = tqdm(index_df.GUPI.unique(),desc=progress_bar_desc)
    else:
        iterator = index_df.GUPI.unique()
        
    for curr_GUPI in iterator:
        
        filt_index_df = index_df[index_df.GUPI==curr_GUPI].reset_index(drop=True)
        full_index_list = list(itertools.chain.from_iterable(filt_index_df.VocabIndex.tolist()))
        full_vocab_list = curr_vocab.lookup_tokens(full_index_list)
        filt_vocab_df = vocab_df[vocab_df.VocabIndex.isin(np.unique(full_index_list))].reset_index(drop=True)
        
        if not missing:
            filt_vocab_df = filt_vocab_df[~filt_vocab_df.Missing].reset_index(drop=True)
            
        token_freqs = pd.DataFrame.from_dict(OrderedDict(Counter(full_vocab_list).most_common()),orient='index').reset_index().rename(columns={'index':'Token',0:'Count'})
        token_freqs = token_freqs[token_freqs.Token.isin(filt_vocab_df.Token)].reset_index(drop=True)
        token_freqs['GUPI'] = curr_GUPI
        
        compiled_token_incidences.append(token_freqs)
    
    return pd.concat(compiled_token_incidences,ignore_index=True)