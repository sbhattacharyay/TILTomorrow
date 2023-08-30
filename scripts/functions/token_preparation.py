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