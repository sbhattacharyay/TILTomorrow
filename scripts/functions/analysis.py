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
from pandas.api.types import CategoricalDtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
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

# Function to prepare model output dataframe for performance metric calculation
def prepare_df(pred_df,window_indices):
    
    # Determine non-consecutive window indices
    non_consec_wis = [window_indices[i] for i in (np.where(np.diff(window_indices) != 1)[0]+1)]
    
    # Iterate through non-consecutive windows
    for curr_idx in non_consec_wis:
        
        # Identify GUPIs with missing true label at current non-consecutive window
        curr_missing_GUPIs = pred_df[(pred_df.WindowIdx==curr_idx)&(pred_df.TrueLabel.isna())].GUPI.unique()
        
        # Identify instances in which the consecutive window index has a non-missing true label for the current missing GUPI set
        replacements = pred_df[pred_df.GUPI.isin(curr_missing_GUPIs) & (pred_df.WindowIdx.isin([curr_idx-1,curr_idx+1])) & (pred_df.TrueLabel.notna())].reset_index(drop=True)
        
        # If there are viable consecutive window indices, replace missing values with them
        if replacements.shape[0] != 0:
            
            # Use the highest window index if others are available
            replacements = replacements.loc[replacements.groupby(['GUPI','TUNE_IDX','REPEAT','FOLD','SET']).WindowIdx.idxmax()].reset_index(drop=True)
            
            # Identify which rows shall be replaced 
            remove_rows = replacements[['GUPI','TUNE_IDX','REPEAT','FOLD','SET']]
            remove_rows['WindowIdx'] = curr_idx
            
            # Add indicator designating rows for replacement
            pred_df = pred_df.merge(remove_rows,how='left',indicator=True)
            
            # Rectify window index in replacement dataframe
            replacements['WindowIdx'] = curr_idx
            
            # Replace rows with missing true label with viable, consecutive-window replacement
            pred_df = pd.concat([pred_df[pred_df._merge!='both'].drop(columns='_merge'),replacements],ignore_index=True).sort_values(by=['REPEAT','FOLD','TUNE_IDX','GUPI']).reset_index(drop=True)
            
        else:
            pass
    
    # Filter dataframe to desired window indices
    pred_df = pred_df[pred_df.WindowIdx.isin(window_indices)].reset_index(drop=True)
    
    # Return filtered dataframe
    return(pred_df)
        
# Function to calculate ORC on given set outputs
def calc_ORC(pred_df,window_indices,progress_bar = True,progress_bar_desc = ''):
    orcs = []
    if progress_bar:
        iterator = tqdm(pred_df.TUNE_IDX.unique(),desc=progress_bar_desc)
    else:
        iterator = pred_df.TUNE_IDX.unique()
    for curr_tune_idx in iterator:
        for curr_wi in window_indices:
            filt_is_preds = pred_df[(pred_df.WindowIdx == curr_wi)&(pred_df.TUNE_IDX == curr_tune_idx)].reset_index(drop=True)
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(filt_is_preds.TrueLabel.dropna().unique()), 2)):
                filt_prob_matrix = filt_is_preds[filt_is_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
                filt_prob_matrix['ConditLabel'] = (filt_prob_matrix.TrueLabel == b).astype(int)
                aucs.append(roc_auc_score(filt_prob_matrix['ConditLabel'],filt_prob_matrix['ExpectedValue']))
            orcs.append(pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                      'WINDOW_IDX':curr_wi,
                                      'METRIC':'ORC',
                                      'VALUE':np.mean(aucs)},index=[0]))
    return pd.concat(orcs,ignore_index=True)

# Function to calculate AUC on given set outputs
def calc_AUC(pred_df,window_indices,progress_bar = True,progress_bar_desc = ''):
    AUCs = []
    if progress_bar:
        iterator = tqdm(pred_df.TUNE_IDX.unique(),desc=progress_bar_desc)
    else:
        iterator = pred_df.TUNE_IDX.unique()
    for curr_tune_idx in iterator:
        for curr_wi in window_indices:
            filt_is_preds = pred_df[(pred_df.WindowIdx == curr_wi)&(pred_df.TUNE_IDX == curr_tune_idx)].reset_index(drop=True)
            curr_AUC = roc_auc_score(filt_is_preds.TrueLabel[filt_is_preds.TrueLabel.notna()],filt_is_preds.ExpectedValue[filt_is_preds.TrueLabel.notna()])
            AUCs.append(pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                      'WINDOW_IDX':curr_wi,
                                      'METRIC':'AUC',
                                      'VALUE':curr_AUC},index=[0]))
    return pd.concat(AUCs,ignore_index=True)

# Function to calculate Somers_D on given set outputs
def calc_Somers_D(pred_df,window_indices,progress_bar = True,progress_bar_desc = ''):
    somers_d = []
    if progress_bar:
        iterator = tqdm(pred_df.TUNE_IDX.unique(),desc=progress_bar_desc)
    else:
        iterator = pred_df.TUNE_IDX.unique()
    for curr_tune_idx in iterator:
        for curr_wi in window_indices:
            filt_is_preds = pred_df[(pred_df.WindowIdx == curr_wi)&(pred_df.TUNE_IDX == curr_tune_idx)].reset_index(drop=True)
            aucs = []
            prevalence = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(filt_is_preds.TrueLabel.dropna().unique()), 2)):
                filt_prob_matrix = filt_is_preds[filt_is_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
                filt_prob_matrix['ConditLabel'] = (filt_prob_matrix.TrueLabel == b).astype(int)
                prevalence.append((filt_prob_matrix.TrueLabel == a).sum()*(filt_prob_matrix.TrueLabel == b).sum())
                aucs.append(roc_auc_score(filt_prob_matrix['ConditLabel'],filt_prob_matrix['ExpectedValue']))
            somers_d.append(pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                          'WINDOW_IDX':curr_wi,
                                          'METRIC':'Somers D',
                                          'VALUE':2*(np.sum(np.multiply(aucs,prevalence))/np.sum(prevalence))-1},index=[0]))
    return pd.concat(somers_d,ignore_index=True)

# Function to calculate threshold-level calibration metrics on given set outputs
def calc_thresh_calibration(pred_df,window_indices,progress_bar = True,progress_bar_desc = ''):
    
    prob_cols = [col for col in pred_df if col.startswith('Pr(TILBasic=')]
    thresh_labels = ['TILBasic>0','TILBasic>1','TILBasic>2','TILBasic>3']
    calib_metrics = []

    if progress_bar:
        iterator = tqdm(pred_df.TUNE_IDX.unique(),desc=progress_bar_desc)
    else:
        iterator = pred_df.TUNE_IDX.unique()
    
    for thresh in range(1,len(prob_cols)):
        cols_gt = prob_cols[thresh:]
        prob_gt = pred_df[cols_gt].sum(1).values
        gt = (pred_df['TrueLabel'] >= thresh).astype(int).values
        pred_df['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
        pred_df[thresh_labels[thresh-1]] = gt
    
    for curr_tune_idx in iterator:
        for curr_wi in window_indices:
            filt_is_preds = pred_df[(pred_df.WindowIdx == curr_wi)&(pred_df.TUNE_IDX == curr_tune_idx)&(pred_df.TrueLabel.notna())].reset_index(drop=True)
            for thresh in thresh_labels:
                thresh_prob_name = 'Pr('+thresh+')'
                try:
                    logit_gt = np.nan_to_num(logit(filt_is_preds[thresh_prob_name]),neginf=-100,posinf=100)
                    calib_glm = Logit(filt_is_preds[thresh], add_constant(logit_gt))
                    calib_glm_res = calib_glm.fit(disp=False)
                    curr_calib_slope = calib_glm_res.params[1]
                except:
                    curr_calib_slope = np.nan
                thresh_calib_linspace = np.linspace(filt_is_preds[thresh_prob_name].min(),filt_is_preds[thresh_prob_name].max(),200)
                TrueProb = lowess(endog = filt_is_preds[thresh], exog = filt_is_preds[thresh_prob_name], it = 0, xvals = thresh_calib_linspace)
                filt_is_preds['TruePr('+thresh+')'] = filt_is_preds[thresh_prob_name].apply(lambda x: TrueProb[(np.abs(x - thresh_calib_linspace)).argmin()])
                ICI = (filt_is_preds['TruePr('+thresh+')'] - filt_is_preds[thresh_prob_name]).abs().mean()
                Emax = (filt_is_preds['TruePr('+thresh+')'] - filt_is_preds[thresh_prob_name]).abs().max()
                calib_metrics.append(pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                                   'WINDOW_IDX':curr_wi,
                                                   'THRESHOLD':thresh,
                                                   'METRIC':['CALIB_SLOPE','ICI','Emax'],
                                                   'VALUE':[curr_calib_slope,ICI,Emax]}))
    calib_metrics = pd.concat(calib_metrics,ignore_index = True).reset_index(drop=True)
    return calib_metrics

# Function to calculate binary calibration metrics on given set outputs
def calc_binary_calibration(pred_df,window_indices,progress_bar = True,progress_bar_desc = ''):
    
    calib_metrics = []

    if progress_bar:
        iterator = tqdm(pred_df.TUNE_IDX.unique(),desc=progress_bar_desc)
    else:
        iterator = pred_df.TUNE_IDX.unique()
    
    for curr_tune_idx in iterator:
        for curr_wi in window_indices:
            filt_is_preds = pred_df[(pred_df.WindowIdx == curr_wi)&(pred_df.TUNE_IDX == curr_tune_idx)&(pred_df.TrueLabel.notna())].reset_index(drop=True)
            
            try:
                logit_gt = np.nan_to_num(logit(filt_is_preds['ExpectedValue']),neginf=-100,posinf=100)
                calib_glm = Logit(filt_is_preds['TrueLabel'], add_constant(logit_gt))
                calib_glm_res = calib_glm.fit(disp=False)
                curr_calib_slope = calib_glm_res.params[1]
            except:
                curr_calib_slope = np.nan
            thresh_calib_linspace = np.linspace(filt_is_preds['ExpectedValue'].min(),filt_is_preds['ExpectedValue'].max(),200)
            TrueProb = lowess(endog = filt_is_preds['TrueLabel'], exog = filt_is_preds['ExpectedValue'], it = 0, xvals = thresh_calib_linspace)
            filt_is_preds['TruePr('+'HighTIL=1'+')'] = filt_is_preds['ExpectedValue'].apply(lambda x: TrueProb[(np.abs(x - thresh_calib_linspace)).argmin()])
            ICI = (filt_is_preds['TruePr('+'HighTIL=1'+')'] - filt_is_preds['ExpectedValue']).abs().mean()
            Emax = (filt_is_preds['TruePr('+'HighTIL=1'+')'] - filt_is_preds['ExpectedValue']).abs().max()
            calib_metrics.append(pd.DataFrame({'TUNE_IDX':curr_tune_idx,
                                               'WINDOW_IDX':curr_wi,
                                               'METRIC':['CALIB_SLOPE','ICI','Emax'],
                                               'VALUE':[curr_calib_slope,ICI,Emax]}))
    calib_metrics = pd.concat(calib_metrics,ignore_index = True).reset_index(drop=True)
    return calib_metrics