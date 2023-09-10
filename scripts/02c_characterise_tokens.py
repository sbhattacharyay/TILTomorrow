#### Master Script 2c: Create a dictionary of all tokens in study and characterise tokens in each patient's ICU stay ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Create a full dictionary of tokens for exploration
# III. Categorize tokens from each patient's ICU stay

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
from functions.token_preparation import categorizer, clean_token_rows, get_token_info, count_token_incidences

## Define and create relevant directories
# Define directory in which CENTER-TBI data is stored
dir_CENTER_TBI = '../../center_tbi/CENTER-TBI'

# Define subdirectory in which formatted TIL values are stored
form_TIL_dir = os.path.join(dir_CENTER_TBI,'FormattedTIL')

# Define directory for storing tokens for each partition
tokens_dir = '../tokens'

## Load fundamental information for variable tokenisation
# Load cross-validation splits of study population
cv_splits = pd.read_csv('../cross_validation_splits.csv')

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

## Define parameters for token characterisation
# Set version code
VERSION = 'v1-0'

### II. Create a full dictionary of tokens for exploration
## Identify and characterise all TILTomorrow token dictionaries
# Search for all dictionary files
token_dict_files = []
for path in Path(tokens_dir).rglob('TILTomorrow_token_dictionary.pkl'):
    token_dict_files.append(str(path.resolve()))

# Characterise the dictionary files found
token_dict_file_info_df = pd.DataFrame({'FILE':token_dict_files,
                                        'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in token_dict_files],
                                        'FOLD':[int(re.search('/fold(.*)/TILTomorrow', curr_file).group(1)) for curr_file in token_dict_files]
                                        }).sort_values(by=['REPEAT','FOLD']).reset_index(drop=True)

## Extract all tokens across repeated cross-validation partitions
# Initialize empty list for storing tokens
compiled_tokens_list = []

# Iterate through identified dictionary files
for curr_dict_idx in tqdm(range(token_dict_file_info_df.shape[0]),'Iterating through available dictionary files for vocabulary collection'):

    # Load current fold vocabulary
    curr_vocab = cp.load(open(token_dict_file_info_df.FILE[curr_dict_idx],"rb"))
    
    # Append tokens from current vocabulary to running list
    compiled_tokens_list.append(curr_vocab.get_itos())

# Flatten list of token lists
compiled_tokens_list = np.unique(list(itertools.chain.from_iterable(compiled_tokens_list)))

## Create characterised dataframe of all possible tokens
# Initialise dataframe
full_token_keys = pd.DataFrame({'Token':compiled_tokens_list})

# Parse out `BaseToken` and `Value` from `Token`
full_token_keys['BaseToken'] = full_token_keys.Token.str.split('_').str[0]
full_token_keys['Value'] = full_token_keys.Token.str.split('_',n=1).str[1].fillna('')

# Determine wheter tokens represent missing values
full_token_keys['Missing'] = full_token_keys.Token.str.endswith('_NAN')

# Determine whether tokens are numeric
full_token_keys['Numeric'] = full_token_keys.Token.str.contains('_BIN')

# Determine whether tokens are baseline or discharge
full_token_keys['Baseline'] = full_token_keys.Token.str.startswith('Baseline')
full_token_keys['Discharge'] = full_token_keys.Token.str.startswith('Discharge')

# For baseline and discharge tokens, remove prefix from `BaseToken` entry
full_token_keys.BaseToken[full_token_keys.Baseline] = full_token_keys.BaseToken[full_token_keys.Baseline].str.replace('Baseline','',1,regex=False)
full_token_keys.BaseToken[full_token_keys.Discharge] = full_token_keys.BaseToken[full_token_keys.Discharge].str.replace('Discharge','',1,regex=False)

## Extract token information for prior (dynamic GOSE) study
# Load token keys from prior study (v1) and filter to those found in current study
legacy_v1_token_key = pd.read_excel(os.path.join(tokens_dir,'legacy_full_token_keys.xlsx'))
legacy_v1_token_key['BaseToken'] = legacy_v1_token_key['BaseToken'].fillna('')
legacy_v1_token_key = legacy_v1_token_key[legacy_v1_token_key.BaseToken.isin(full_token_keys.BaseToken)].reset_index(drop=True)
legacy_v1_token_key['Binary'] = False

# Load token keys from prior study (v2) and filter to those found in current study
legacy_v2_token_key = pd.read_excel(os.path.join(tokens_dir,'full_token_keys.xlsx'))
legacy_v2_token_key['BaseToken'] = legacy_v2_token_key['BaseToken'].fillna('')
legacy_v2_token_key = legacy_v2_token_key[legacy_v2_token_key.BaseToken.isin(full_token_keys.BaseToken)].reset_index(drop=True)

# Based on hierarchy, remove all v1 legacy token keys that are in v2 legacy token keys
legacy_v1_token_key = legacy_v1_token_key[~legacy_v1_token_key.BaseToken.isin(legacy_v2_token_key.BaseToken)].reset_index(drop=True)

# Add a new column in v2 legacy token keys designating unknown non-missing values and drop missingness indicator
legacy_v2_token_key['UnknownNonMissing'] = (legacy_v2_token_key.Missing) & ~(legacy_v2_token_key.Value.isin(['NAN','BIN_NAN']))
legacy_v2_token_key = legacy_v2_token_key.drop(columns='Missing')
legacy_v1_token_key = legacy_v1_token_key.drop(columns='Missing')

# In the v1 legacy token keys, fix token types to match those of v2 legacy token keys
legacy_v1_token_key.Type[legacy_v1_token_key.Type=='Surgery and Neuromonitoring'] = 'Surgery'
legacy_v1_token_key.Type[(legacy_v1_token_key.Type=='ICU Medications and Management')&(legacy_v1_token_key.BaseToken=='TransReason')] = 'Transitions of Care'
legacy_v1_token_key.Type[(legacy_v1_token_key.Type=='ICU Medications and Management')&(legacy_v1_token_key.BaseToken=='HVTILChangeReason')] = 'Bihourly Assessments'
legacy_v1_token_key.Type[(legacy_v1_token_key.Type=='ICU Medications and Management')] = 'ICU Monitoring and Management'

# Concatenate base-token information from both legacy version token keys
compiled_legacy_base_token_key = pd.concat([legacy_v1_token_key.drop(columns=['Token','Numeric','Baseline']).drop_duplicates(ignore_index=True),
                                            legacy_v2_token_key.drop(columns=['Token','Value','OrderIdx','Numeric','Baseline','Discharge','UnknownNonMissing']).drop_duplicates(ignore_index=True)],ignore_index=True).drop_duplicates(ignore_index=True)

# Merge base-token information onto current compiled token vocabulary
full_token_keys = full_token_keys.merge(compiled_legacy_base_token_key,how='left')

# Merge token-unique information (of unknown but not missing) from v2 legacy token keys onto current compiled token vocabulary
full_token_keys = full_token_keys.merge(legacy_v2_token_key[['Token','BaseToken','UnknownNonMissing']].drop_duplicates(),how='left')

## Manually add 'UnknownNonMissing' status for other unknown but not missing tokens
# Identify tokens with unknown, but not missing, values
unk_nonmiss_tokens = full_token_keys[full_token_keys.Value.isin(['088','UNK','077','UNINTERPRETABLE','INDETEMINATE','NOTINTERPRETED','UNKNOWN','NOTKNOWN','UNKNWON'])&(full_token_keys.UnknownNonMissing.isna())&(~full_token_keys.BaseToken.isin(['ERaAngleExtem','DayOfICUStay','ERCTExtem','APAggreg']))].reset_index(drop=True)

# If convert all indicator values for tokens in the list to true, and all other tokens of the BaseTokens to false
full_token_keys.UnknownNonMissing[full_token_keys.Token.isin(unk_nonmiss_tokens.Token)] = True
full_token_keys.UnknownNonMissing[full_token_keys.BaseToken.isin(unk_nonmiss_tokens.BaseToken)&(~full_token_keys.Token.isin(unk_nonmiss_tokens.Token))&(full_token_keys.UnknownNonMissing.isna())] = False

# If a BaseToken has any nonmissing `UnknownNonMissing` indicators, then impute all missing `UnknownNonMissing` indicators with True
full_token_keys.UnknownNonMissing[(full_token_keys.BaseToken.isin(full_token_keys[full_token_keys.UnknownNonMissing.notna()].BaseToken.unique()))&(full_token_keys.UnknownNonMissing.isna())] = False

# Do manual inspection of all remaining tokens with missing `UnknownNonMissing` indicators
missing_unk_nonmiss_indicator_tokens = full_token_keys[full_token_keys.UnknownNonMissing.isna()].reset_index(drop=True)

# Based on manual inspection, all remaining tokens are not `UnknownNonMissing`. Therefore, impute with false
full_token_keys.UnknownNonMissing[full_token_keys.UnknownNonMissing.isna()] = False

## Fix missing token `Type` designations
# If BaseToken starts with 'ERJSON' or 'JSON', categorise into Brain imaging type
full_token_keys.Type[full_token_keys.Type.isna()&(full_token_keys.BaseToken.str.startswith('ERJSON')|full_token_keys.BaseToken.str.startswith('JSON'))] = 'Brain Imaging'

# If BaseToken starts with 'TIL' or 'TotalTIL', categorise into ICU Monitoring and Management type
full_token_keys.Type[full_token_keys.Type.isna()&(full_token_keys.BaseToken.str.startswith('TIL')|full_token_keys.BaseToken.str.contains('TotalTIL')|full_token_keys.BaseToken.isin(['HighIntensityTherapy','TimepointDiff']))] = 'ICU Monitoring and Management'

# If BaseToken contains 'ICP', categorise into ICU Monitoring and Management type
full_token_keys.Type[full_token_keys.Type.isna()&(full_token_keys.BaseToken.str.contains('ICP'))] = 'ICU Monitoring and Management'

# If BaseToken contains 'SixMonthOutcome', categorise into Discharge Assessment and ICU Stay Summary type
full_token_keys.Type[full_token_keys.Type.isna()&(full_token_keys.BaseToken.str.contains('SixMonthOutcome'))] = 'Discharge Assessment and ICU Stay Summary'

# Create new type corresponding to temporal metadata tokens
full_token_keys.Type[full_token_keys.Type.isna()&full_token_keys.BaseToken.isin(['DayOfICUStay'])] = 'Day of ICU Stay'

## Fix missing metadata for variables
# If a variable is numeric, then it should automatically be ordered and non-binary
full_token_keys.Ordered[full_token_keys.Ordered.isna()&full_token_keys.Numeric] = True
full_token_keys.Binary[full_token_keys.Binary.isna()&full_token_keys.Numeric] = False

# For each variable with missing `Ordered` or `Binary` indicators, determine the unique number of non-missing tokens
CountPerBaseToken = full_token_keys[full_token_keys.Ordered.isna()|full_token_keys.Binary.isna()].groupby(['BaseToken'],as_index=False).Missing.aggregate({'Missings':'sum','ValueOptions':'count'}).merge(full_token_keys[full_token_keys.Ordered.isna()|full_token_keys.Binary.isna()].groupby(['BaseToken'],as_index=False).UnknownNonMissing.aggregate({'Unknowns':'sum','ValueOptions':'count'}))
CountPerBaseToken['NonMissings'] = CountPerBaseToken.ValueOptions.astype(int) - CountPerBaseToken.Missings.astype(int) - CountPerBaseToken.Unknowns.astype(int)

# If variable has 2 or fewer unique non-missing tokens, disqualify it from being ordered
full_token_keys.Ordered[full_token_keys.Ordered.isna()&full_token_keys.BaseToken.isin(CountPerBaseToken.BaseToken[CountPerBaseToken.NonMissings <= 2])] = False

# If variable has exactly 2 unique non-missing tokens, mark it as binary. Otherwise, not binary
full_token_keys.Binary[full_token_keys.Binary.isna()&full_token_keys.BaseToken.isin(CountPerBaseToken.BaseToken[CountPerBaseToken.NonMissings == 2])] = True
full_token_keys.Binary[full_token_keys.Binary.isna()&full_token_keys.BaseToken.isin(CountPerBaseToken.BaseToken[CountPerBaseToken.NonMissings != 2])] = False

# Based on manual inspection: All other tokens are ordered, unless in specified list
full_token_keys.Ordered[full_token_keys.Ordered.isna()&full_token_keys.BaseToken.isin(['ERJSONIncidentalFindingsOtherComments','ICUDisSixMonthOutcomeType','ICUReasonNoICP','JSONIncidentalFindingsOtherComments','SixMonthOutcomeType'])] = False
full_token_keys.Ordered[full_token_keys.Ordered.isna()] = True

# Among variables with missing `ClinicianInput` markers, only prognosis variables and ICP reasons are True
full_token_keys.ClinicianInput[full_token_keys.ClinicianInput.isna()&(full_token_keys.BaseToken.str.contains('SixMonthOutcome')|full_token_keys.BaseToken.str.startswith('ICUReasonNoICP'))] = True
full_token_keys.ClinicianInput[full_token_keys.ClinicianInput.isna()] = False

# Among variables with missing `ICUIntervention` markers, only variables containing 'TIL' and 'HighIntensityTherapy' are True 
full_token_keys.ICUIntervention[full_token_keys.ICUIntervention.isna()&(full_token_keys.BaseToken.str.contains('TIL')|(full_token_keys.BaseToken=='HighIntensityTherapy'))] = True
full_token_keys.ICUIntervention[full_token_keys.ICUIntervention.isna()] = False

## Add ordering index to Binary and Ordered variables
# Correct boolean variable types
full_token_keys[['Ordered','Binary','ICUIntervention','ClinicianInput','UnknownNonMissing']] = full_token_keys[['Ordered','Binary','ICUIntervention','ClinicianInput','UnknownNonMissing']].astype(bool)

# If Binary or Ordered variables have less than 2 nonmissing options in dataset, remove Binary or Ordered label
CountPerBaseToken = full_token_keys.groupby(['BaseToken','Ordered','Binary'],as_index=False).Missing.aggregate({'Missings':'sum','ValueOptions':'count'}).merge(full_token_keys.groupby(['BaseToken','Ordered','Binary'],as_index=False).UnknownNonMissing.aggregate({'Unknowns':'sum','ValueOptions':'count'}))
CountPerBaseToken['NonMissings'] = CountPerBaseToken.ValueOptions.astype(int) - CountPerBaseToken.Missings.astype(int) - CountPerBaseToken.Unknowns.astype(int)
full_token_keys.Binary[full_token_keys.BaseToken.isin(CountPerBaseToken[CountPerBaseToken.Binary&(CountPerBaseToken.NonMissings != 2)].BaseToken.unique())] = False
full_token_keys.Ordered[full_token_keys.BaseToken.isin(CountPerBaseToken[CountPerBaseToken.Ordered&(CountPerBaseToken.NonMissings == 1)].BaseToken.unique())] = False

# Initialise column for storing ordering index for Binary or Ordered variables
full_token_keys['OrderIdx'] = np.nan

# Sort full token dataframe alphabetically
full_token_keys = full_token_keys.sort_values(by=['BaseToken','Token'],ignore_index=True)

# Iterate through Numeric variables and order values alphabetically
numeric_vars = full_token_keys[full_token_keys.Numeric].BaseToken.unique()

# Iterate through Numeric variables and order values alphabetically
for curr_var in tqdm(numeric_vars,'Iterating through Numeric variables for ordering'):    
    full_token_keys.OrderIdx[(full_token_keys.BaseToken==curr_var)&(~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing)] = np.arange(full_token_keys[(full_token_keys.BaseToken==curr_var)&(~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing)].shape[0])

# Identify orderings in legacy dictionary which would match ordering in new dictionary
legacy_ordering_token_ct = legacy_v2_token_key[(legacy_v2_token_key.OrderIdx.notna())&(~legacy_v2_token_key.Numeric)].groupby('BaseToken',as_index=False).Token.count().rename(columns={'Token':'LegacyCount'})
new_ordering_token_ct = full_token_keys[(full_token_keys.BaseToken.isin(legacy_ordering_token_ct.BaseToken))&(~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing)].groupby('BaseToken',as_index=False).Token.count().rename(columns={'Token':'NewCount'})
ordering_token_ct = new_ordering_token_ct.merge(legacy_ordering_token_ct,how='left')
ordering_token_ct = ordering_token_ct[ordering_token_ct.NewCount == ordering_token_ct.LegacyCount].reset_index(drop=True)

# Place legacy ordering indices onto current dictionary
full_token_keys = full_token_keys.merge(legacy_v2_token_key[(legacy_v2_token_key.BaseToken.isin(ordering_token_ct.BaseToken))&(legacy_v2_token_key.OrderIdx.notna())][['BaseToken','Value','OrderIdx']].rename(columns={'OrderIdx':'LegacyOrderIdx'}),how='left')
full_token_keys.OrderIdx[(full_token_keys.OrderIdx.isna())&(full_token_keys.LegacyOrderIdx.notna())] = full_token_keys.LegacyOrderIdx[(full_token_keys.OrderIdx.isna())&(full_token_keys.LegacyOrderIdx.notna())]
full_token_keys = full_token_keys.drop(columns='LegacyOrderIdx')

# Iterate through Binary variables with missing ordering and order values alphabetically
for curr_var in tqdm(full_token_keys[(full_token_keys.Binary)&((~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing))&(full_token_keys.OrderIdx.isna())].BaseToken.unique(),'Iterating through remaining Binary variables for ordering'):    
    full_token_keys.OrderIdx[(full_token_keys.BaseToken==curr_var)&(~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing)] = np.arange(full_token_keys[(full_token_keys.BaseToken==curr_var)&(~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing)].shape[0])

# Order JSON compression variables
compression_vars = full_token_keys[(full_token_keys.Ordered)&((~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing))&(full_token_keys.OrderIdx.isna())&(full_token_keys.Value=='COMPTRESSED')].BaseToken.unique()
full_token_keys.OrderIdx[(full_token_keys.OrderIdx.isna())&(full_token_keys.BaseToken.isin(compression_vars))&(full_token_keys.Value=='NORMAL')] = 0
full_token_keys.OrderIdx[(full_token_keys.OrderIdx.isna())&(full_token_keys.BaseToken.isin(compression_vars))&(full_token_keys.Value=='COMPTRESSED')] = 1
full_token_keys.OrderIdx[(full_token_keys.OrderIdx.isna())&(full_token_keys.BaseToken.isin(compression_vars))&(full_token_keys.Value=='ABSENT')] = 2

# Order JSON TSAH variables
tsah_vars = full_token_keys[(full_token_keys.Ordered)&((~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing))&(full_token_keys.OrderIdx.isna())&(full_token_keys.Value=='MODERATE')].BaseToken.unique()
full_token_keys.OrderIdx[(full_token_keys.OrderIdx.isna())&(full_token_keys.BaseToken.isin(tsah_vars))&(full_token_keys.Value=='TRACE')] = 0
full_token_keys.OrderIdx[(full_token_keys.OrderIdx.isna())&(full_token_keys.BaseToken.isin(tsah_vars))&(full_token_keys.Value=='MODERATE')] = 1
full_token_keys.OrderIdx[(full_token_keys.OrderIdx.isna())&(full_token_keys.BaseToken.isin(tsah_vars))&(full_token_keys.Value=='FULL')] = 2

# Order variables with only numbers alphabetically
alpha_vars = full_token_keys[(full_token_keys.Ordered)&((~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing))&(full_token_keys.OrderIdx.isna())&(full_token_keys.Value.apply(lambda x: re.search('[a-zA-Z]', x)))].BaseToken.unique()
num_vars = np.setdiff1d(full_token_keys[(full_token_keys.Ordered)&((~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing))&(full_token_keys.OrderIdx.isna())].BaseToken.unique(),alpha_vars)
for curr_var in tqdm(num_vars,'Iterating through Ordered variables only containing numeric characters for ordering'):    
    full_token_keys.OrderIdx[(full_token_keys.BaseToken==curr_var)&(~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing)] = np.arange(full_token_keys[(full_token_keys.BaseToken==curr_var)&(~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing)].shape[0])

# Order variables with "DEC" for decimal encoding
dec_vars = full_token_keys[(full_token_keys.Ordered)&((~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing))&(full_token_keys.OrderIdx.isna())&(full_token_keys.Value.str.contains('DEC'))].BaseToken.unique()
for curr_var in tqdm(dec_vars,'Iterating through Ordered variables with DEC encoding for ordering'):    
    full_token_keys.OrderIdx[(full_token_keys.BaseToken==curr_var)&(~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing)] = (full_token_keys[(full_token_keys.BaseToken==curr_var)&(~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing)].Value.str.replace('DEC','.').astype(float).rank()-1)

# Order outcome prognosis variables based on outcome types
prog_vars = full_token_keys[(full_token_keys.Ordered)&((~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing))&(full_token_keys.OrderIdx.isna())&(full_token_keys.BaseToken.str.contains('Outcome'))].BaseToken.unique()
for curr_var in tqdm(prog_vars,'Iterating through Ordered variables with outcome prognoses for ordering'):    
    full_token_keys.OrderIdx[(full_token_keys.BaseToken==curr_var)&(~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing)] = (full_token_keys[(full_token_keys.BaseToken==curr_var)&(~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing)].Value.map({'D':0,'V':1,'SD':2,'MD':3,'GR':4}).rank()-1)

# Rank remaining variables alphabetically
remaining_vars = full_token_keys[(full_token_keys.Ordered)&((~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing))&(full_token_keys.OrderIdx.isna())].BaseToken.unique()
for curr_var in tqdm(remaining_vars,'Iterating through remaining Ordered variables for ordering'):    
    full_token_keys.OrderIdx[(full_token_keys.BaseToken==curr_var)&(~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing)] = np.arange(full_token_keys[(full_token_keys.BaseToken==curr_var)&(~full_token_keys.Missing)&(~full_token_keys.UnknownNonMissing)].shape[0])

## Save fully formatted token dictionary onto token directory
# Reorder columns
full_token_keys.insert(3,'OrderIdx',full_token_keys.pop('OrderIdx'))
full_token_keys.insert(5,'UnknownNonMissing',full_token_keys.pop('UnknownNonMissing'))

# Save file
full_token_keys.to_excel(os.path.join(tokens_dir,'pre_check_TILTomorrow_full_token_keys_'+VERSION+'.xlsx'),index=False)

### III. Categorize tokens from each patient's ICU stay
## Identify and characterise all TILTomorrow token dictionaries
# Load formatted full token dictionary
full_token_keys = pd.read_excel(os.path.join(tokens_dir,'TILTomorrow_full_token_keys_'+VERSION+'.xlsx'))
full_token_keys.Token = full_token_keys.Token.fillna('')
full_token_keys.BaseToken = full_token_keys.BaseToken.fillna('')

# Search for all dictionary files
token_dict_files = []
for path in Path(tokens_dir).rglob('TILTomorrow_token_dictionary.pkl'):
    token_dict_files.append(str(path.resolve()))

# Characterise the dictionary files found
token_dict_file_info_df = pd.DataFrame({'FILE':token_dict_files,
                                        'REPEAT':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in token_dict_files],
                                        'FOLD':[int(re.search('/fold(.*)/TILTomorrow', curr_file).group(1)) for curr_file in token_dict_files]
                                        }).sort_values(by=['REPEAT','FOLD']).reset_index(drop=True)

## Characterise tokens across repeated cross-validation partitions
# Initialize empty list for storing tokens
compiled_tokens_list = []

# Iterate through identified dictionary files
for curr_dict_idx in tqdm(range(token_dict_file_info_df.shape[0]),'Iterating through available dictionary files for patient token characterisation'):

    ## Extract parameters corresponding to current dictionary file
    # Current repeat
    curr_repeat = token_dict_file_info_df.REPEAT[curr_dict_idx]

    # Current fold
    curr_fold = token_dict_file_info_df.FOLD[curr_dict_idx]

    # Current file
    curr_file = token_dict_file_info_df.FILE[curr_dict_idx]

    # Define subdirectory for the current repeated cross-validation partition based on extracted parameters
    fold_dir = os.path.join(tokens_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold).zfill(1))

    ## Extract current training, validation, and testing set GUPIs
    curr_fold_splits = cv_splits[(cv_splits.REPEAT==curr_repeat)&(cv_splits.FOLD==curr_fold)].reset_index(drop=True)
    curr_train_GUPIs = curr_fold_splits[curr_fold_splits.SET=='train'].GUPI.unique()
    curr_val_GUPIs = curr_fold_splits[curr_fold_splits.SET=='val'].GUPI.unique()
    curr_test_GUPIs = curr_fold_splits[curr_fold_splits.SET=='test'].GUPI.unique()

    ## Categorize token vocabulary from current fold
    # Load current fold vocabulary
    curr_vocab = cp.load(open(curr_file,"rb"))
    
    # Create dataframe version of vocabulary
    curr_vocab_df = pd.DataFrame({'VocabIndex':list(range(len(curr_vocab))),'Token':curr_vocab.get_itos()})

    # Merge token dictionary information onto current vocabulary
    curr_vocab_df = curr_vocab_df.merge(full_token_keys,how='left')

    # Load index sets for current fold
    train_indices = pd.read_pickle(os.path.join(fold_dir,'TILTomorrow_training_indices.pkl'))
    val_indices = pd.read_pickle(os.path.join(fold_dir,'TILTomorrow_validation_indices.pkl'))
    test_indices = pd.read_pickle(os.path.join(fold_dir,'TILTomorrow_testing_indices.pkl'))
    
    # Add set indicator and combine index sets for current fold
    train_indices['Set'] = 'train'
    val_indices['Set'] = 'val'
    test_indices['Set'] = 'test'
    indices_df = pd.concat([train_indices,val_indices,test_indices],ignore_index=True)

    # For the purposes of characterisation, combine VocabIndex and VocabPhysImpressionIndex
    indices_df.VocabIndex = indices_df.VocabIndex + indices_df.VocabPhysImpressionIndex

    # Calculate token information for each study window row
    study_window_token_info = get_token_info(indices_df,curr_vocab_df,False,True,'Characterising tokens in study windows for repeat '+str(curr_repeat)+' fold '+str(curr_fold))

    # Add repeat and fold information
    study_window_token_info.insert(0,'REPEAT',curr_repeat)
    study_window_token_info.insert(1,'FOLD',curr_fold)

    # Save calculated token information into current fold directory
    study_window_token_info.to_pickle(os.path.join(fold_dir,'TILTomorrow_token_type_counts.pkl'))

    # Calculate count information per patient-token combination
    token_patient_incidences = count_token_incidences(indices_df,curr_vocab,curr_vocab_df,False,True,'Characterising tokens in study windows for repeat '+str(curr_repeat)+' fold '+str(curr_fold))

    # Add repeat and fold information
    token_patient_incidences.insert(0,'REPEAT',curr_repeat)
    token_patient_incidences.insert(1,'FOLD',curr_fold)
    
    # Save token incidence information into current fold directory
    token_patient_incidences.to_pickle(os.path.join(fold_dir,'TILTomorrow_token_incidences_per_patient.pkl'))