#### Master Script 2b: Convert full patient information from ICU stays into tokenised sets ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Tokenise numeric variables and place into study windows

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
from tqdm import tqdm
import multiprocessing
from scipy import stats
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
warnings.filterwarnings(action="ignore")
from collections import Counter, OrderedDict

# SciKit-Learn methods
from sklearn.preprocessing import KBinsDiscretizer

# PyTorch and PyTorch.Text methods
from torchtext.vocab import vocab, Vocab

# Custom methods
from functions.token_preparation import categorizer, clean_token_rows

## Define and create relevant directories
# Define directory in which CENTER-TBI data is stored
dir_CENTER_TBI = '/home/sb2406/rds/hpc-work/CENTER-TBI'

# Define subdirectory in which formatted TIL values are stored
form_TIL_dir = os.path.join(dir_CENTER_TBI,'FormattedTIL')

# Create directory for storing tokens for each partition
tokens_dir = '/home/sb2406/rds/hpc-work/tokens'
os.makedirs(tokens_dir,exist_ok=True)

## Load fundamental information for variable tokenisation
# Load cross-validation splits of study population
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Extract unique repeated cross-validation partitions
uniq_partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates(ignore_index=True)

# Load study included set
study_included_set = pd.read_csv(os.path.join(form_TIL_dir,'study_included_set.csv'))

# Format timestamps in study included set properly
study_included_set['ICUAdmTimeStamp'] = pd.to_datetime(study_included_set['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
study_included_set['ICUDischTimeStamp'] = pd.to_datetime(study_included_set['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
study_included_set['WLSTDecisionTimeStamp'] = pd.to_datetime(study_included_set['WLSTDecisionTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
study_included_set['EndTimeStamp'] = pd.to_datetime(study_included_set['EndTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# Extract dates from ICU admission and study end timestamps
study_included_set['ICUAdmDate'] = study_included_set['ICUAdmTimeStamp'].dt.date
study_included_set['EndDate'] = study_included_set['EndTimeStamp'].dt.date
study_included_set['WLSTDecisionDate'] = study_included_set['WLSTDecisionTimeStamp'].dt.date

# Extract times from ICU admission and study end timestamps
study_included_set['ICUAdmTime'] = study_included_set['ICUAdmTimeStamp'].dt.time
study_included_set['EndTime'] = study_included_set['EndTimeStamp'].dt.time
study_included_set['WLSTDecisionTime'] = study_included_set['WLSTDecisionTimeStamp'].dt.time

# Load formatted TIL values
formatted_TIL_values = pd.read_csv(os.path.join(form_TIL_dir,'formatted_TIL_values.csv'))

# Format date timestamp in formatted TIL dataframe properly
formatted_TIL_values['TILDate'] = pd.to_datetime(formatted_TIL_values['TILDate'],format = '%Y-%m-%d' )

## Define parameters for token preparation
# Define the number of bins for discretising numeric variables
BINS = 20

# Define number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Tokenise numeric variables and place into study windows
def main(array_task_id):

    ## Extract current repeated cross-validation parameters
    # Current repeat
    curr_repeat = uniq_partitions.REPEAT[array_task_id]

    # Current fold
    curr_fold = uniq_partitions.FOLD[array_task_id]

    ## Load cleaned categorical tokens in study windows
    cleaned_study_tokens_df = pd.read_pickle(os.path.join(form_TIL_dir,'categorical_tokens_in_study_windows.pkl'))
    cleaned_study_tokens_df['TOKENS'] = cleaned_study_tokens_df.TOKENS.str.strip()
    cleaned_study_tokens_df['PHYSIMPRESSIONTOKENS'] = cleaned_study_tokens_df.PHYSIMPRESSIONTOKENS.str.strip()

    ## Load formatted numeric variables
    # Numeric baseline variables
    numeric_baseline_variables = pd.read_pickle(os.path.join(form_TIL_dir,'numeric_baseline_variables.pkl')).reset_index(drop=True)
    numeric_baseline_variables['VARIABLE'] = numeric_baseline_variables.VARIABLE.str.strip().str.replace('_','')

    # Numeric baseline physician impressions
    numeric_baseline_physician_impressions = pd.read_pickle(os.path.join(form_TIL_dir,'numeric_baseline_physician_impressions.pkl')).reset_index(drop=True)
    numeric_baseline_physician_impressions['VARIABLE'] = numeric_baseline_physician_impressions.VARIABLE.str.strip().str.replace('_','')

    # Numeric discharge variables
    numeric_discharge_variables = pd.read_pickle(os.path.join(form_TIL_dir,'numeric_discharge_variables.pkl')).reset_index(drop=True)
    numeric_discharge_variables['VARIABLE'] = numeric_discharge_variables.VARIABLE.str.strip().str.replace('_','')

    # Numeric date-intervalled variables
    numeric_date_interval_variables = pd.read_pickle(os.path.join(form_TIL_dir,'numeric_date_interval_variables.pkl')).reset_index(drop=True)
    numeric_date_interval_variables['VARIABLE'] = numeric_date_interval_variables.VARIABLE.str.strip().str.replace('_','')

    # Numeric time-intervalled physician impressions
    numeric_time_interval_physician_impressions = pd.read_pickle(os.path.join(form_TIL_dir,'numeric_time_interval_physician_impressions.pkl')).reset_index(drop=True)
    numeric_time_interval_physician_impressions['VARIABLE'] = numeric_time_interval_physician_impressions.VARIABLE.str.strip().str.replace('_','')

    # Numeric dated single-event variables
    numeric_date_event_variables = pd.read_pickle(os.path.join(form_TIL_dir,'numeric_date_event_variables.pkl')).reset_index(drop=True)
    numeric_date_event_variables['VARIABLE'] = numeric_date_event_variables.VARIABLE.str.strip().str.replace('_','')

    # Numeric dated single-event physician impressions
    numeric_date_event_physician_impressions = pd.read_pickle(os.path.join(form_TIL_dir,'numeric_date_event_physician_impressions.pkl')).reset_index(drop=True)
    numeric_date_event_physician_impressions['VARIABLE'] = numeric_date_event_physician_impressions.VARIABLE.str.strip().str.replace('_','')

    # Numeric timestamped single-event variables
    numeric_timestamp_event_variables = pd.read_pickle(os.path.join(form_TIL_dir,'numeric_timestamp_event_variables.pkl')).reset_index(drop=True)
    numeric_timestamp_event_variables['VARIABLE'] = numeric_timestamp_event_variables.VARIABLE.str.strip().str.replace('_','')

    # Create a subdirectory for the current repeated cross-validation partition
    fold_dir = os.path.join(tokens_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(1))
    os.makedirs(fold_dir,exist_ok=True)

    ## Extract current training, validation, and testing set GUPIs
    curr_fold_splits = cv_splits[(cv_splits.REPEAT==curr_repeat)&(cv_splits.FOLD==curr_fold)].reset_index(drop=True)
    curr_train_GUPIs = curr_fold_splits[curr_fold_splits.SET=='train'].GUPI.unique()
    curr_val_GUPIs = curr_fold_splits[curr_fold_splits.SET=='val'].GUPI.unique()
    curr_test_GUPIs = curr_fold_splits[curr_fold_splits.SET=='test'].GUPI.unique()

    ## Numeric baseline variables
    # Extract unique names of numeric baseline variables from the training set
    unique_numeric_baseline_variables = numeric_baseline_variables[numeric_baseline_variables.GUPI.isin(curr_train_GUPIs)].VARIABLE.unique()
    
    # Create column for storing bin value
    numeric_baseline_variables['BIN'] = ''
    
    # For missing values, assign 'NAN' to bin value
    numeric_baseline_variables.BIN[numeric_baseline_variables.VALUE.isna()] = '_NAN'
    
    # Iterate through unique numeric baseline variables and tokenise
    for curr_variable in tqdm(unique_numeric_baseline_variables,'Tokenising numeric baseline variables for repeat '+str(curr_repeat)+' fold '+str(curr_fold)):
        
        # Create a `KBinsDiscretizer` object for discretising the current variable
        curr_nbp_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
        
        # Train cuts for discretisation of the current variable
        curr_nbp_kbd.fit(np.expand_dims(numeric_baseline_variables[(numeric_baseline_variables.VARIABLE==curr_variable)&(numeric_baseline_variables.GUPI.isin(curr_train_GUPIs))&(~numeric_baseline_variables.VALUE.isna())].VALUE.values,1))
        
        # Discretise current variable into bins
        numeric_baseline_variables.BIN[(numeric_baseline_variables.VARIABLE==curr_variable)&(~numeric_baseline_variables.VALUE.isna())] = (categorizer(pd.Series((curr_nbp_kbd.transform(np.expand_dims(numeric_baseline_variables[(numeric_baseline_variables.VARIABLE==curr_variable)&(~numeric_baseline_variables.VALUE.isna())].VALUE.values,1))+1).squeeze()),100)).str.replace(r'\s+','',regex=True).values
        
    # If a variable has been neglected, replace with value
    numeric_baseline_variables.BIN[numeric_baseline_variables.BIN==''] = numeric_baseline_variables.VALUE[numeric_baseline_variables.BIN==''].astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)
    
    # Create tokens from each variable and bin value
    numeric_baseline_variables['TOKEN'] = numeric_baseline_variables.VARIABLE + '_BIN' + numeric_baseline_variables.BIN
    
    # Concatenate tokens from each GUPI into a combined baseline numeric variable token set
    numeric_baseline_variables = numeric_baseline_variables.drop_duplicates(subset=['GUPI','TOKEN'],ignore_index=True).groupby('GUPI',as_index=False).TOKEN.aggregate(lambda x: ' '.join(x)).rename(columns={'TOKEN':'NumericBaselineTokens'})
    
    # Merge baseline numeric variables with `cleaned_study_tokens_df`
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(numeric_baseline_variables,how='left',on=['GUPI'])
    cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericBaselineTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericBaselineTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericBaselineTokens[~cleaned_study_tokens_df.NumericBaselineTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericBaselineTokens')

    ## Numeric baseline physician impressions
    # Extract unique names of numeric baseline physician impressions from the training set
    unique_numeric_baseline_physician_impressions = numeric_baseline_physician_impressions[numeric_baseline_physician_impressions.GUPI.isin(curr_train_GUPIs)].VARIABLE.unique()

    # Create column for storing bin value
    numeric_baseline_physician_impressions['BIN'] = ''

    # For missing values, assign 'NAN' to bin value
    numeric_baseline_physician_impressions.BIN[numeric_baseline_physician_impressions.VALUE.isna()] = '_NAN'

    # Iterate through unique numeric baseline physician impressions and tokenise
    for curr_physician_impression in tqdm(unique_numeric_baseline_physician_impressions,'Tokenising numeric baseline physician impressions for repeat '+str(curr_repeat)+' fold '+str(curr_fold)):
        
        # Create a `KBinsDiscretizer` object for discretising the current physician impression
        curr_nbp_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
        
        # Train cuts for discretisation of the current physician impression
        curr_nbp_kbd.fit(np.expand_dims(numeric_baseline_physician_impressions[(numeric_baseline_physician_impressions.VARIABLE==curr_physician_impression)&(numeric_baseline_physician_impressions.GUPI.isin(curr_train_GUPIs))&(~numeric_baseline_physician_impressions.VALUE.isna())].VALUE.values,1))
        
        # Discretise current physician impression into bins
        numeric_baseline_physician_impressions.BIN[(numeric_baseline_physician_impressions.VARIABLE==curr_physician_impression)&(~numeric_baseline_physician_impressions.VALUE.isna())] = (categorizer(pd.Series((curr_nbp_kbd.transform(np.expand_dims(numeric_baseline_physician_impressions[(numeric_baseline_physician_impressions.VARIABLE==curr_physician_impression)&(~numeric_baseline_physician_impressions.VALUE.isna())].VALUE.values,1))+1).squeeze()),100)).str.replace(r'\s+','',regex=True).values
        
    # If a physician impression has been neglected, replace with value
    numeric_baseline_physician_impressions.BIN[numeric_baseline_physician_impressions.BIN==''] = numeric_baseline_physician_impressions.VALUE[numeric_baseline_physician_impressions.BIN==''].astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)

    # Create tokens from each physician impression and bin value
    numeric_baseline_physician_impressions['TOKEN'] = numeric_baseline_physician_impressions.VARIABLE + '_BIN' + numeric_baseline_physician_impressions.BIN

    # Concatenate tokens from each GUPI into a combined baseline numeric physician impression token set
    numeric_baseline_physician_impressions = numeric_baseline_physician_impressions.drop_duplicates(subset=['GUPI','TOKEN'],ignore_index=True).groupby('GUPI',as_index=False).TOKEN.aggregate(lambda x: ' '.join(x)).rename(columns={'TOKEN':'NumericBaselineTokens'})

    # Merge baseline numeric physician impressions with `cleaned_study_tokens_df`
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(numeric_baseline_physician_impressions,how='left',on=['GUPI'])
    cleaned_study_tokens_df.PHYSIMPRESSIONTOKENS[~cleaned_study_tokens_df.NumericBaselineTokens.isna()] = cleaned_study_tokens_df.PHYSIMPRESSIONTOKENS[~cleaned_study_tokens_df.NumericBaselineTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericBaselineTokens[~cleaned_study_tokens_df.NumericBaselineTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericBaselineTokens')

    ## Numeric discharge variables
    # Extract unique names of numeric discharge variables from the training set
    unique_numeric_discharge_variables = numeric_discharge_variables[numeric_discharge_variables.GUPI.isin(curr_train_GUPIs)].VARIABLE.unique()
    
    # Create column for storing bin value
    numeric_discharge_variables['BIN'] = ''
    
    # For missing values, assign 'NAN' to bin value
    numeric_discharge_variables.BIN[numeric_discharge_variables.VALUE.isna()] = '_NAN'
    
    # Iterate through unique numeric discharge variables and tokenise
    for curr_variable in tqdm(unique_numeric_discharge_variables,'Tokenising numeric discharge variables for repeat '+str(curr_repeat)+' fold '+str(curr_fold)):
        
        # Create a `KBinsDiscretizer` object for discretising the current variable
        curr_ndp_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
        
        # Train cuts for discretisation of the current variable
        curr_ndp_kbd.fit(np.expand_dims(numeric_discharge_variables[(numeric_discharge_variables.VARIABLE==curr_variable)&(numeric_discharge_variables.GUPI.isin(curr_train_GUPIs))&(~numeric_discharge_variables.VALUE.isna())].VALUE.values,1))
        
        # Discretise current variable into bins
        numeric_discharge_variables.BIN[(numeric_discharge_variables.VARIABLE==curr_variable)&(~numeric_discharge_variables.VALUE.isna())] = (categorizer(pd.Series((curr_ndp_kbd.transform(np.expand_dims(numeric_discharge_variables[(numeric_discharge_variables.VARIABLE==curr_variable)&(~numeric_discharge_variables.VALUE.isna())].VALUE.values,1))+1).squeeze()),100)).str.replace(r'\s+','',regex=True).values
        
    # If a variable has been neglected, replace with value
    numeric_discharge_variables.BIN[numeric_discharge_variables.BIN==''] = numeric_discharge_variables.VALUE[numeric_discharge_variables.BIN==''].astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)
    
    # Create tokens from each variable and bin value
    numeric_discharge_variables['TOKEN'] = numeric_discharge_variables.VARIABLE + '_BIN' + numeric_discharge_variables.BIN
    
    # Concatenate tokens from each GUPI into a combined discharge numeric variable token set
    numeric_discharge_variables = numeric_discharge_variables.drop_duplicates(subset=['GUPI','TOKEN'],ignore_index=True).groupby('GUPI',as_index=False).TOKEN.aggregate(lambda x: ' '.join(x)).rename(columns={'TOKEN':'NumericDischargeTokens'})
    
    # Add last window index information to discharge variable dataframe
    numeric_discharge_variables = numeric_discharge_variables.merge(cleaned_study_tokens_df[['GUPI','WindowTotal']].drop_duplicates().rename(columns={'WindowTotal':'WindowIdx'}),how='left',on='GUPI')

    # Merge discharge numeric variables with `cleaned_study_tokens_df`
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(numeric_discharge_variables,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericDischargeTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericDischargeTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericDischargeTokens[~cleaned_study_tokens_df.NumericDischargeTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericDischargeTokens')

    ## Numeric date-intervalled variables
    # Extract unique names of numeric date-intervalled variables from the training set
    unique_numeric_date_interval_variables = numeric_date_interval_variables[numeric_date_interval_variables.GUPI.isin(curr_train_GUPIs)].VARIABLE.unique()
    
    # Create column for storing bin value
    numeric_date_interval_variables['BIN'] = ''
    
    # For missing values, assign 'NAN' to bin value
    numeric_date_interval_variables.BIN[numeric_date_interval_variables.VALUE.isna()] = '_NAN'
    
    # Iterate through unique numeric date-intervalled variables and tokenise
    for curr_variable in tqdm(unique_numeric_date_interval_variables,'Tokenising numeric date-intervalled variables for repeat '+str(curr_repeat)+' fold '+str(curr_fold)):
        
        # Create a `KBinsDiscretizer` object for discretising the current variable
        curr_ndiv_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
        
        # Train cuts for discretisation of the current variable
        curr_ndiv_kbd.fit(np.expand_dims(numeric_date_interval_variables[(numeric_date_interval_variables.VARIABLE==curr_variable)&(numeric_date_interval_variables.GUPI.isin(curr_train_GUPIs))&(~numeric_date_interval_variables.VALUE.isna())].VALUE.values,1))
        
        # Discretise current variable into bins
        numeric_date_interval_variables.BIN[(numeric_date_interval_variables.VARIABLE==curr_variable)&(~numeric_date_interval_variables.VALUE.isna())] = (categorizer(pd.Series((curr_ndiv_kbd.transform(np.expand_dims(numeric_date_interval_variables[(numeric_date_interval_variables.VARIABLE==curr_variable)&(~numeric_date_interval_variables.VALUE.isna())].VALUE.values,1))+1).squeeze()),100)).str.replace(r'\s+','',regex=True).values
        
    # If a variable has been neglected, replace with value
    numeric_date_interval_variables.BIN[numeric_date_interval_variables.BIN==''] = numeric_date_interval_variables.VALUE[numeric_date_interval_variables.BIN==''].astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)
    
    # Create tokens from each variable and bin value
    numeric_date_interval_variables['TOKEN'] = numeric_date_interval_variables.VARIABLE + '_BIN' + numeric_date_interval_variables.BIN
    
    # Concatenate tokens from each GUPI and date into a combined date-intervalled numeric variable token set
    numeric_date_interval_variables = numeric_date_interval_variables.drop_duplicates(subset=['GUPI','StartDate','StopDate','TOKEN'],ignore_index=True).groupby(['GUPI','StartDate','StopDate'],as_index=False).TOKEN.aggregate(lambda x: ' '.join(x)).rename(columns={'TOKEN':'NumericTimeIntervalTokens'})
    
    # Merge window date starts and ends to formatted variable dataframe
    numeric_date_interval_variables = numeric_date_interval_variables.merge(cleaned_study_tokens_df[['GUPI','TimeStampStart','TimeStampEnd','WindowIdx']],how='left',on='GUPI')

    # First, isolate events which finish before the ICU admission date and combine end tokens
    baseline_numeric_date_interval_variables = numeric_date_interval_variables[numeric_date_interval_variables.WindowIdx == 1]
    baseline_numeric_date_interval_variables = baseline_numeric_date_interval_variables[baseline_numeric_date_interval_variables.StopDate < baseline_numeric_date_interval_variables.TimeStampStart].reset_index(drop=True)
    baseline_numeric_date_interval_variables = baseline_numeric_date_interval_variables.drop(columns=['StartDate','StopDate','TimeStampStart','TimeStampEnd'])
    # baseline_numeric_date_interval_variables = baseline_numeric_date_interval_variables.groupby(['GUPI','WindowIdx'],as_index=False).NumericTimeIntervalTokens.aggregate(lambda x: ' '.join(x))

    # # Merge event tokens which finish before the date of ICU admission onto study tokens dataframe
    # cleaned_study_tokens_df = cleaned_study_tokens_df.merge(baseline_numeric_date_interval_variables,how='left',on=['GUPI','WindowIdx'])
    # cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericTimeIntervalTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericTimeIntervalTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericTimeIntervalTokens[~cleaned_study_tokens_df.NumericTimeIntervalTokens.isna()]
    # cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericTimeIntervalTokens')

    # Then, isolate the events that fit within the given window
    numeric_date_interval_variables = numeric_date_interval_variables[(numeric_date_interval_variables.StartDate <= numeric_date_interval_variables.TimeStampEnd)&(numeric_date_interval_variables.StopDate >= numeric_date_interval_variables.TimeStampStart)].reset_index(drop=True)

    # Merge dated event tokens onto study tokens dataframe
    numeric_date_interval_variables = numeric_date_interval_variables.groupby(['GUPI','WindowIdx'],as_index=False).NumericTimeIntervalTokens.aggregate(lambda x: ' '.join(x))
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(numeric_date_interval_variables,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericTimeIntervalTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericTimeIntervalTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericTimeIntervalTokens[~cleaned_study_tokens_df.NumericTimeIntervalTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericTimeIntervalTokens')

    ## Numeric timestamp-intervalled physician impressions
    # Extract unique names of numeric timestamp-intervalled physician impressions from the training set
    unique_numeric_time_interval_physician_impressions = numeric_time_interval_physician_impressions[numeric_time_interval_physician_impressions.GUPI.isin(curr_train_GUPIs)].VARIABLE.unique()
    
    # Create column for storing bin value
    numeric_time_interval_physician_impressions['BIN'] = ''
    
    # For missing values, assign 'NAN' to bin value
    numeric_time_interval_physician_impressions.BIN[numeric_time_interval_physician_impressions.VALUE.isna()] = '_NAN'
    
    # Iterate through unique numeric timestamp-intervalled physician impressions and tokenise
    for curr_physician_impression in tqdm(unique_numeric_time_interval_physician_impressions,'Tokenising numeric timestamp-intervalled physician impressions for repeat '+str(curr_repeat)+' fold '+str(curr_fold)):
        
        # Create a `KBinsDiscretizer` object for discretising the current physician impression
        curr_ntep_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
        
        # Train cuts for discretisation of the current physician impression
        curr_ntep_kbd.fit(np.expand_dims(numeric_time_interval_physician_impressions[(numeric_time_interval_physician_impressions.VARIABLE==curr_physician_impression)&(numeric_time_interval_physician_impressions.GUPI.isin(curr_train_GUPIs))&(~numeric_time_interval_physician_impressions.VALUE.isna())].VALUE.values,1))
        
        # Discretise current physician impression into bins
        numeric_time_interval_physician_impressions.BIN[(numeric_time_interval_physician_impressions.VARIABLE==curr_physician_impression)&(~numeric_time_interval_physician_impressions.VALUE.isna())] = (categorizer(pd.Series((curr_ntep_kbd.transform(np.expand_dims(numeric_time_interval_physician_impressions[(numeric_time_interval_physician_impressions.VARIABLE==curr_physician_impression)&(~numeric_time_interval_physician_impressions.VALUE.isna())].VALUE.values,1))+1).squeeze()),100)).str.replace(r'\s+','',regex=True).values
        
    # If a physician impression has been neglected, replace with value
    numeric_time_interval_physician_impressions.BIN[numeric_time_interval_physician_impressions.BIN==''] = numeric_time_interval_physician_impressions.VALUE[numeric_time_interval_physician_impressions.BIN==''].astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)
    
    # Create tokens from each physician impression and bin value
    numeric_time_interval_physician_impressions['TOKEN'] = numeric_time_interval_physician_impressions.VARIABLE + '_BIN' + numeric_time_interval_physician_impressions.BIN
    
    # Concatenate tokens from each GUPI and timestamp into a combined timestamp-intervalled numeric physician impression token set
    numeric_time_interval_physician_impressions = numeric_time_interval_physician_impressions.drop_duplicates(subset=['GUPI','StartTimeStamp','StopTimeStamp','TOKEN'],ignore_index=True).groupby(['GUPI','StartTimeStamp','StopTimeStamp'],as_index=False).TOKEN.aggregate(lambda x: ' '.join(x)).rename(columns={'TOKEN':'NumericTimeIntervalTokens'})
    
    # Merge window timestamp starts and ends to formatted physician impression dataframe
    numeric_time_interval_physician_impressions = numeric_time_interval_physician_impressions.merge(cleaned_study_tokens_df[['GUPI','TimeStampStart','TimeStampEnd','WindowIdx']],how='left',on='GUPI')

    # First, isolate events which finish before the ICU admission timestamp and combine end tokens
    baseline_numeric_time_interval_physician_impressions = numeric_time_interval_physician_impressions[numeric_time_interval_physician_impressions.WindowIdx == 1]
    baseline_numeric_time_interval_physician_impressions = baseline_numeric_time_interval_physician_impressions[baseline_numeric_time_interval_physician_impressions.StopTimeStamp < baseline_numeric_time_interval_physician_impressions.TimeStampStart].reset_index(drop=True)
    baseline_numeric_time_interval_physician_impressions = baseline_numeric_time_interval_physician_impressions.drop(columns=['StartTimeStamp','StopTimeStamp','TimeStampStart','TimeStampEnd'])
    baseline_numeric_time_interval_physician_impressions = baseline_numeric_time_interval_physician_impressions.groupby(['GUPI','WindowIdx'],as_index=False).NumericTimeIntervalTokens.aggregate(lambda x: ' '.join(x))

    # Merge event tokens which finish before the date of ICU admission onto study tokens dataframe
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(baseline_numeric_time_interval_physician_impressions,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.PHYSIMPRESSIONTOKENS[~cleaned_study_tokens_df.NumericTimeIntervalTokens.isna()] = cleaned_study_tokens_df.PHYSIMPRESSIONTOKENS[~cleaned_study_tokens_df.NumericTimeIntervalTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericTimeIntervalTokens[~cleaned_study_tokens_df.NumericTimeIntervalTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericTimeIntervalTokens')

    # Then, isolate the events that fit within the given window
    numeric_time_interval_physician_impressions = numeric_time_interval_physician_impressions[(numeric_time_interval_physician_impressions.StartTimeStamp <= numeric_time_interval_physician_impressions.TimeStampEnd)&(numeric_time_interval_physician_impressions.StopTimeStamp >= numeric_time_interval_physician_impressions.TimeStampStart)].reset_index(drop=True)

    # Merge timestamped event tokens onto study tokens dataframe
    numeric_time_interval_physician_impressions = numeric_time_interval_physician_impressions.groupby(['GUPI','WindowIdx'],as_index=False).NumericTimeIntervalTokens.aggregate(lambda x: ' '.join(x))
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(numeric_time_interval_physician_impressions,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.PHYSIMPRESSIONTOKENS[~cleaned_study_tokens_df.NumericTimeIntervalTokens.isna()] = cleaned_study_tokens_df.PHYSIMPRESSIONTOKENS[~cleaned_study_tokens_df.NumericTimeIntervalTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericTimeIntervalTokens[~cleaned_study_tokens_df.NumericTimeIntervalTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericTimeIntervalTokens')

    ## Numeric dated single-event variables
    # Extract unique names of numeric dated single-event variables from the training set
    unique_numeric_date_event_variables = numeric_date_event_variables[numeric_date_event_variables.GUPI.isin(curr_train_GUPIs)].VARIABLE.unique()
    
    # Create column for storing bin value
    numeric_date_event_variables['BIN'] = ''
    
    # For missing values, assign 'NAN' to bin value
    numeric_date_event_variables.BIN[numeric_date_event_variables.VALUE.isna()] = '_NAN'
    
    # Iterate through unique numeric dated single-event variables and tokenise
    for curr_variable in tqdm(unique_numeric_date_event_variables,'Tokenising numeric dated single-event variables for repeat '+str(curr_repeat)+' fold '+str(curr_fold)):
        
        # Create a `KBinsDiscretizer` object for discretising the current variable
        curr_ndep_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
        
        # Train cuts for discretisation of the current variable
        curr_ndep_kbd.fit(np.expand_dims(numeric_date_event_variables[(numeric_date_event_variables.VARIABLE==curr_variable)&(numeric_date_event_variables.GUPI.isin(curr_train_GUPIs))&(~numeric_date_event_variables.VALUE.isna())].VALUE.values,1))
        
        # Discretise current variable into bins
        numeric_date_event_variables.BIN[(numeric_date_event_variables.VARIABLE==curr_variable)&(~numeric_date_event_variables.VALUE.isna())] = (categorizer(pd.Series((curr_ndep_kbd.transform(np.expand_dims(numeric_date_event_variables[(numeric_date_event_variables.VARIABLE==curr_variable)&(~numeric_date_event_variables.VALUE.isna())].VALUE.values,1))+1).squeeze()),100)).str.replace(r'\s+','',regex=True).values
        
    # If a variable has been neglected, replace with value
    numeric_date_event_variables.BIN[numeric_date_event_variables.BIN==''] = numeric_date_event_variables.VALUE[numeric_date_event_variables.BIN==''].astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)
    
    # Create tokens from each variable and bin value
    numeric_date_event_variables['TOKEN'] = numeric_date_event_variables.VARIABLE + '_BIN' + numeric_date_event_variables.BIN
    
    # Concatenate tokens from each GUPI and date into a combined dated single-event numeric variable token set
    numeric_date_event_variables = numeric_date_event_variables.drop_duplicates(subset=['GUPI','Date','TOKEN'],ignore_index=True).groupby(['GUPI','Date'],as_index=False).TOKEN.aggregate(lambda x: ' '.join(x)).rename(columns={'TOKEN':'NumericDateEventTokens'})
    
    # Merge window timestamp starts and ends to formatted variable dataframe
    numeric_date_event_variables = numeric_date_event_variables.merge(cleaned_study_tokens_df[['GUPI','TimeStampStart','TimeStampEnd','WindowIdx']],how='left',on='GUPI')

    # First, isolate events which finish before the date ICU admission and combine end tokens
    baseline_numeric_date_event_variables = numeric_date_event_variables[numeric_date_event_variables.WindowIdx == 1]
    baseline_numeric_date_event_variables = baseline_numeric_date_event_variables[baseline_numeric_date_event_variables.Date.dt.date < baseline_numeric_date_event_variables.TimeStampStart.dt.date].reset_index(drop=True)
    baseline_numeric_date_event_variables = baseline_numeric_date_event_variables.drop(columns=['Date','TimeStampStart','TimeStampEnd'])
    baseline_numeric_date_event_variables = baseline_numeric_date_event_variables.groupby(['GUPI','WindowIdx'],as_index=False).NumericDateEventTokens.aggregate(lambda x: ' '.join(x))

    # Merge event tokens which finish before the date of ICU admission onto study tokens dataframe
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(baseline_numeric_date_event_variables,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericDateEventTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericDateEventTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericDateEventTokens[~cleaned_study_tokens_df.NumericDateEventTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericDateEventTokens')

    # Then, isolate the events that fit within the given window
    numeric_date_event_variables = numeric_date_event_variables[(numeric_date_event_variables.Date.dt.date <= numeric_date_event_variables.TimeStampEnd.dt.date)&(numeric_date_event_variables.Date.dt.date >= numeric_date_event_variables.TimeStampStart.dt.date)].reset_index(drop=True)

    # Merge dated event tokens onto study tokens dataframe
    numeric_date_event_variables = numeric_date_event_variables.groupby(['GUPI','WindowIdx'],as_index=False).NumericDateEventTokens.aggregate(lambda x: ' '.join(x))
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(numeric_date_event_variables,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericDateEventTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericDateEventTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericDateEventTokens[~cleaned_study_tokens_df.NumericDateEventTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericDateEventTokens')

    ## Numeric dated single-event physician impressions
    # Extract unique names of numeric dated single-event physician impressions from the training set
    unique_numeric_date_event_physician_impressions = numeric_date_event_physician_impressions[numeric_date_event_physician_impressions.GUPI.isin(curr_train_GUPIs)].VARIABLE.unique()
    
    # Create column for storing bin value
    numeric_date_event_physician_impressions['BIN'] = ''
    
    # For missing values, assign 'NAN' to bin value
    numeric_date_event_physician_impressions.BIN[numeric_date_event_physician_impressions.VALUE.isna()] = '_NAN'
    
    # Iterate through unique numeric dated single-event physician impressions and tokenise
    for curr_physician_impression in tqdm(unique_numeric_date_event_physician_impressions,'Tokenising numeric dated single-event physician impressions for repeat '+str(curr_repeat)+' fold '+str(curr_fold)):
        
        # Create a `KBinsDiscretizer` object for discretising the current physician impression
        curr_ndep_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
        
        # Train cuts for discretisation of the current physician impression
        curr_ndep_kbd.fit(np.expand_dims(numeric_date_event_physician_impressions[(numeric_date_event_physician_impressions.VARIABLE==curr_physician_impression)&(numeric_date_event_physician_impressions.GUPI.isin(curr_train_GUPIs))&(~numeric_date_event_physician_impressions.VALUE.isna())].VALUE.values,1))
        
        # Discretise current physician impression into bins
        numeric_date_event_physician_impressions.BIN[(numeric_date_event_physician_impressions.VARIABLE==curr_physician_impression)&(~numeric_date_event_physician_impressions.VALUE.isna())] = (categorizer(pd.Series((curr_ndep_kbd.transform(np.expand_dims(numeric_date_event_physician_impressions[(numeric_date_event_physician_impressions.VARIABLE==curr_physician_impression)&(~numeric_date_event_physician_impressions.VALUE.isna())].VALUE.values,1))+1).squeeze()),100)).str.replace(r'\s+','',regex=True).values
        
    # If a physician impression has been neglected, replace with value
    numeric_date_event_physician_impressions.BIN[numeric_date_event_physician_impressions.BIN==''] = numeric_date_event_physician_impressions.VALUE[numeric_date_event_physician_impressions.BIN==''].astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)
    
    # Create tokens from each physician impression and bin value
    numeric_date_event_physician_impressions['TOKEN'] = numeric_date_event_physician_impressions.VARIABLE + '_BIN' + numeric_date_event_physician_impressions.BIN
    
    # Concatenate tokens from each GUPI and date into a combined dated single-event numeric physician impression token set
    numeric_date_event_physician_impressions = numeric_date_event_physician_impressions.drop_duplicates(subset=['GUPI','Date','TOKEN'],ignore_index=True).groupby(['GUPI','Date'],as_index=False).TOKEN.aggregate(lambda x: ' '.join(x)).rename(columns={'TOKEN':'NumericDateEventTokens'})
    
    # Merge window timestamp starts and ends to formatted physician impression dataframe
    numeric_date_event_physician_impressions = numeric_date_event_physician_impressions.merge(cleaned_study_tokens_df[['GUPI','TimeStampStart','TimeStampEnd','WindowIdx']],how='left',on='GUPI')

    # First, isolate events which finish before the date ICU admission and combine end tokens
    baseline_numeric_date_event_physician_impressions = numeric_date_event_physician_impressions[numeric_date_event_physician_impressions.WindowIdx == 1]
    baseline_numeric_date_event_physician_impressions = baseline_numeric_date_event_physician_impressions[baseline_numeric_date_event_physician_impressions.Date.dt.date < baseline_numeric_date_event_physician_impressions.TimeStampStart.dt.date].reset_index(drop=True)
    baseline_numeric_date_event_physician_impressions = baseline_numeric_date_event_physician_impressions.drop(columns=['Date','TimeStampStart','TimeStampEnd'])
    baseline_numeric_date_event_physician_impressions = baseline_numeric_date_event_physician_impressions.groupby(['GUPI','WindowIdx'],as_index=False).NumericDateEventTokens.aggregate(lambda x: ' '.join(x))

    # Merge event tokens which finish before the date of ICU admission onto study tokens dataframe
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(baseline_numeric_date_event_physician_impressions,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.PHYSIMPRESSIONTOKENS[~cleaned_study_tokens_df.NumericDateEventTokens.isna()] = cleaned_study_tokens_df.PHYSIMPRESSIONTOKENS[~cleaned_study_tokens_df.NumericDateEventTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericDateEventTokens[~cleaned_study_tokens_df.NumericDateEventTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericDateEventTokens')

    # Then, isolate the events that fit within the given window
    numeric_date_event_physician_impressions = numeric_date_event_physician_impressions[(numeric_date_event_physician_impressions.Date.dt.date <= numeric_date_event_physician_impressions.TimeStampEnd.dt.date)&(numeric_date_event_physician_impressions.Date.dt.date >= numeric_date_event_physician_impressions.TimeStampStart.dt.date)].reset_index(drop=True)

    # Merge dated event tokens onto study tokens dataframe
    numeric_date_event_physician_impressions = numeric_date_event_physician_impressions.groupby(['GUPI','WindowIdx'],as_index=False).NumericDateEventTokens.aggregate(lambda x: ' '.join(x))
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(numeric_date_event_physician_impressions,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.PHYSIMPRESSIONTOKENS[~cleaned_study_tokens_df.NumericDateEventTokens.isna()] = cleaned_study_tokens_df.PHYSIMPRESSIONTOKENS[~cleaned_study_tokens_df.NumericDateEventTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericDateEventTokens[~cleaned_study_tokens_df.NumericDateEventTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericDateEventTokens')

    ## Numeric timestamped single-event variables
    # Extract unique names of numeric timestamped single-event variables from the training set
    unique_numeric_timestamp_event_variables = numeric_timestamp_event_variables[numeric_timestamp_event_variables.GUPI.isin(curr_train_GUPIs)].VARIABLE.unique()
    
    # Create column for storing bin value
    numeric_timestamp_event_variables['BIN'] = ''
    
    # For missing values, assign 'NAN' to bin value
    numeric_timestamp_event_variables.BIN[numeric_timestamp_event_variables.VALUE.isna()] = '_NAN'
    
    # Iterate through unique numeric timestamped single-event variables and tokenise
    for curr_variable in tqdm(unique_numeric_timestamp_event_variables,'Tokenising numeric timestamped single-event variables for repeat '+str(curr_repeat)+' fold '+str(curr_fold)):
        
        # Create a `KBinsDiscretizer` object for discretising the current variable
        curr_ntep_kbd = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')
        
        # Train cuts for discretisation of the current variable
        curr_ntep_kbd.fit(np.expand_dims(numeric_timestamp_event_variables[(numeric_timestamp_event_variables.VARIABLE==curr_variable)&(numeric_timestamp_event_variables.GUPI.isin(curr_train_GUPIs))&(~numeric_timestamp_event_variables.VALUE.isna())].VALUE.values,1))
        
        # Discretise current variable into bins
        numeric_timestamp_event_variables.BIN[(numeric_timestamp_event_variables.VARIABLE==curr_variable)&(~numeric_timestamp_event_variables.VALUE.isna())] = (categorizer(pd.Series((curr_ntep_kbd.transform(np.expand_dims(numeric_timestamp_event_variables[(numeric_timestamp_event_variables.VARIABLE==curr_variable)&(~numeric_timestamp_event_variables.VALUE.isna())].VALUE.values,1))+1).squeeze()),100)).str.replace(r'\s+','',regex=True).values
        
    # If a variable has been neglected, replace with value
    numeric_timestamp_event_variables.BIN[numeric_timestamp_event_variables.BIN==''] = numeric_timestamp_event_variables.VALUE[numeric_timestamp_event_variables.BIN==''].astype(str).str.upper().str.replace('[^a-zA-Z0-9]','').str.replace(r'^\s*$','NAN',regex=True)
    
    # Create tokens from each variable and bin value
    numeric_timestamp_event_variables['TOKEN'] = numeric_timestamp_event_variables.VARIABLE + '_BIN' + numeric_timestamp_event_variables.BIN
    
    # Concatenate tokens from each GUPI and timestamp into a combined timestamped single-event numeric variable token set
    numeric_timestamp_event_variables = numeric_timestamp_event_variables.drop_duplicates(subset=['GUPI','TimeStamp','TOKEN'],ignore_index=True).groupby(['GUPI','TimeStamp'],as_index=False).TOKEN.aggregate(lambda x: ' '.join(x)).rename(columns={'TOKEN':'NumericTimeStampEventTokens'})
    
    # Merge window timestamp starts and ends to formatted variable dataframe
    numeric_timestamp_event_variables = numeric_timestamp_event_variables.merge(cleaned_study_tokens_df[['GUPI','TimeStampStart','TimeStampEnd','WindowIdx']],how='left',on='GUPI')

    # First, isolate events which finish before the ICU admission timestamp and combine end tokens
    baseline_numeric_timestamp_event_variables = numeric_timestamp_event_variables[numeric_timestamp_event_variables.WindowIdx == 1]
    baseline_numeric_timestamp_event_variables = baseline_numeric_timestamp_event_variables[baseline_numeric_timestamp_event_variables.TimeStamp < baseline_numeric_timestamp_event_variables.TimeStampStart].reset_index(drop=True)
    baseline_numeric_timestamp_event_variables = baseline_numeric_timestamp_event_variables.drop(columns=['TimeStamp','TimeStampStart','TimeStampEnd'])
    baseline_numeric_timestamp_event_variables = baseline_numeric_timestamp_event_variables.groupby(['GUPI','WindowIdx'],as_index=False).NumericTimeStampEventTokens.aggregate(lambda x: ' '.join(x))

    # Merge event tokens which finish before the date of ICU admission onto study tokens dataframe
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(baseline_numeric_timestamp_event_variables,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericTimeStampEventTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericTimeStampEventTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericTimeStampEventTokens[~cleaned_study_tokens_df.NumericTimeStampEventTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericTimeStampEventTokens')

    # Then, isolate the events that fit within the given window
    numeric_timestamp_event_variables = numeric_timestamp_event_variables[(numeric_timestamp_event_variables.TimeStamp <= numeric_timestamp_event_variables.TimeStampEnd)&(numeric_timestamp_event_variables.TimeStamp >= numeric_timestamp_event_variables.TimeStampStart)].reset_index(drop=True)

    # Merge timestamped event tokens onto study tokens dataframe
    numeric_timestamp_event_variables = numeric_timestamp_event_variables.groupby(['GUPI','WindowIdx'],as_index=False).NumericTimeStampEventTokens.aggregate(lambda x: ' '.join(x))
    cleaned_study_tokens_df = cleaned_study_tokens_df.merge(numeric_timestamp_event_variables,how='left',on=['GUPI','WindowIdx'])
    cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericTimeStampEventTokens.isna()] = cleaned_study_tokens_df.TOKENS[~cleaned_study_tokens_df.NumericTimeStampEventTokens.isna()] + ' ' + cleaned_study_tokens_df.NumericTimeStampEventTokens[~cleaned_study_tokens_df.NumericTimeStampEventTokens.isna()]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns ='NumericTimeStampEventTokens')

    ## Iterate through and clean tokens
    # Inspect each token row in parallel to ensure unique tokens
    cleaned_study_tokens_df = clean_token_rows(cleaned_study_tokens_df,True,'Cleaning token dataframe for repeat '+str(curr_repeat)+' fold '+str(curr_fold)).sort_values(by=['GUPI','WindowIdx']).reset_index(drop=True)

    # Create an ordered dictionary to create a token vocabulary from the training set
    training_token_list = (' '.join(cleaned_study_tokens_df[cleaned_study_tokens_df.GUPI.isin(curr_train_GUPIs)].TOKENS)).split(' ') + (' '.join(cleaned_study_tokens_df[cleaned_study_tokens_df.GUPI.isin(curr_train_GUPIs)].PHYSIMPRESSIONTOKENS)).split(' ')
    if ('' in training_token_list):
        training_token_list = list(filter(lambda a: a != '', training_token_list))
    train_token_freqs = OrderedDict(Counter(training_token_list).most_common())

    # Build and save vocabulary (PyTorch Text) from training set tokens
    curr_vocab = vocab(train_token_freqs, min_freq=1)
    null_token = ''
    unk_token = '<unk>'
    if null_token not in curr_vocab: curr_vocab.insert_token(null_token, 0)
    if unk_token not in curr_vocab: curr_vocab.insert_token(unk_token, len(curr_vocab))
    curr_vocab.set_default_index(curr_vocab[unk_token])
    cp.dump(curr_vocab, open(os.path.join(fold_dir,'TILTomorrow_token_dictionary.pkl'), "wb" ))
    
    # Convert token set to indices
    cleaned_study_tokens_df['VocabIndex'] = [curr_vocab.lookup_indices(cleaned_study_tokens_df.TOKENS[curr_row].split(' ')) for curr_row in tqdm(range(cleaned_study_tokens_df.shape[0]),desc='Converting study tokens to indices for repeat '+str(curr_repeat)+' fold '+str(curr_fold))]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns='TOKENS')
    cleaned_study_tokens_df['VocabPhysImpressionIndex'] = [curr_vocab.lookup_indices(cleaned_study_tokens_df.PHYSIMPRESSIONTOKENS[curr_row].split(' ')) for curr_row in tqdm(range(cleaned_study_tokens_df.shape[0]),desc='Converting study physician impressions tokens to indices for repeat '+str(curr_repeat)+' fold '+str(curr_fold))]
    cleaned_study_tokens_df = cleaned_study_tokens_df.drop(columns=['PHYSIMPRESSIONTOKENS'])

    # Reorder token columns
    cleaned_study_tokens_df.insert(5,'VocabIndex',cleaned_study_tokens_df.pop('VocabIndex'))
    cleaned_study_tokens_df.insert(6,'VocabPhysImpressionIndex',cleaned_study_tokens_df.pop('VocabPhysImpressionIndex'))

    # Split token set into training, validation, and testing sets
    train_tokens = cleaned_study_tokens_df[cleaned_study_tokens_df.GUPI.isin(curr_train_GUPIs)].reset_index(drop=True)
    val_tokens = cleaned_study_tokens_df[cleaned_study_tokens_df.GUPI.isin(curr_val_GUPIs)].reset_index(drop=True)
    test_tokens = cleaned_study_tokens_df[cleaned_study_tokens_df.GUPI.isin(curr_test_GUPIs)].reset_index(drop=True)

    # Save index sets
    train_tokens.to_pickle(os.path.join(fold_dir,'TILTomorrow_training_indices.pkl'))
    val_tokens.to_pickle(os.path.join(fold_dir,'TILTomorrow_validation_indices.pkl'))
    test_tokens.to_pickle(os.path.join(fold_dir,'TILTomorrow_testing_indices.pkl'))

if __name__ == '__main__':

    array_task_id = int(sys.argv[1])
    main(array_task_id)