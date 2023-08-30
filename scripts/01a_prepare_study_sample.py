#### Master Script 1a: Extract and prepare study sample from CENTER-TBI dataset ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Apply initial inclusion criteria
# III. Load and prepare TIL information
# IV. Partition study set for repeated stratified k-fold cross-validation

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
from tqdm import tqdm
import seaborn as sns
from scipy import stats
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

## Define and create relevant directories
# Define directory in which CENTER-TBI data is stored
dir_CENTER_TBI = '../../center_tbi/CENTER-TBI'

# Define and create subdirectory to store formatted TIL values
form_TIL_dir = os.path.join(dir_CENTER_TBI,'FormattedTIL')
os.makedirs(form_TIL_dir,exist_ok=True)

### II. Apply initial inclusion criteria
## Load and prepare basic patient information
# Load patient ICU admission/discharge information
CENTER_TBI_ICU_timestamps = pd.read_csv(os.path.join(dir_CENTER_TBI,'adm_disch_timestamps.csv'),na_values = ["NA","NaN","NaT"," ", ""])[['GUPI','ICUAdmTimeStamp','ICUDischTimeStamp','ICUDurationHours']]

# Convert admission and discharge timestamps to proper format
CENTER_TBI_ICU_timestamps['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_timestamps['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )
CENTER_TBI_ICU_timestamps['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_timestamps['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# Load patient age information
CENTER_TBI_age_info = pd.read_csv(os.path.join(dir_CENTER_TBI,'DemoInjHospMedHx','data.csv'),na_values = ["NA","NaN","NaT"," ", ""])[['GUPI','PatientType','Age']]

# Load patient ICP monitoring information
CENTER_TBI_ICP_mx_info = pd.read_csv(os.path.join(dir_CENTER_TBI,'ICP_monitored_patients.csv'),na_values = ["NA","NaN","NaT"," ", ""])

# Load patient withdrawal-of-life-sustaining-therapies (WLST) information
CENTER_TBI_WLST_info = pd.read_csv(os.path.join(dir_CENTER_TBI,'WLST_patients.csv'),na_values = ["NA","NaN","NaT"," ", ""])

# Load patient death information
CENTER_TBI_death_info = pd.read_csv(os.path.join(dir_CENTER_TBI,'death_patients.csv'),na_values = ["NA","NaN","NaT"," ", ""])

# Merge ICU discharge and death information to WLST dataframe
CENTER_TBI_WLST_info = CENTER_TBI_WLST_info.merge(CENTER_TBI_ICU_timestamps[['GUPI','ICUDischTimeStamp']],how='left').merge(CENTER_TBI_death_info[['GUPI','ICUDischargeStatus','DeathTimeStamp']],how='left')

# Convert WLST dataframe to long form
CENTER_TBI_WLST_info = CENTER_TBI_WLST_info.melt(id_vars=['GUPI','PatientType','DeathERWithdrawalLifeSuppForSeverityOfTBI','WithdrawalTreatmentDecision','DeadSeverityofTBI','DeadAge','DeadCoMorbidities','DeadRequestRelatives','DeadDeterminationOfBrainDeath','ICUDisWithdrawlTreatmentDecision','ICUDischargeStatus','ICUDischTimeStamp'],value_name='TimeStamp',var_name='TimeStampType')

# Sort WLST dataframe by date(time) value per patient
CENTER_TBI_WLST_info = CENTER_TBI_WLST_info.sort_values(['GUPI','TimeStamp'],ignore_index=True)

# Find patients who have no non-missing timestamps
no_non_missing_timestamp_patients = CENTER_TBI_WLST_info.groupby(['GUPI'],as_index=False)['TimeStamp'].agg('count')
no_non_missing_timestamp_patients = no_non_missing_timestamp_patients[no_non_missing_timestamp_patients['TimeStamp']==0].reset_index(drop=True)

# Drop missing values entries if the paient has at least one non-missing vlaue
CENTER_TBI_WLST_info = CENTER_TBI_WLST_info[(~CENTER_TBI_WLST_info['TimeStamp'].isna())|(CENTER_TBI_WLST_info.GUPI.isin(no_non_missing_timestamp_patients.GUPI))].reset_index(drop=True)

# Extract first timestamp (chronologically) from each patient
CENTER_TBI_WLST_info = CENTER_TBI_WLST_info.groupby('GUPI',as_index=False).first()

# Extract 'DateComponent' from timestamp values
CENTER_TBI_WLST_info['DateComponent'] = pd.to_datetime(CENTER_TBI_WLST_info['TimeStamp'].str[:10],format = '%Y-%m-%d').dt.date

# Convert timestamp to proper format
CENTER_TBI_WLST_info['TimeStamp'] = pd.to_datetime(CENTER_TBI_WLST_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

## Define study inclusion criteria parameters
# Minimum age (>=)
MIN_AGE = 16

# Minimum duration of ICU stay in hours (>=)
MIN_STAY = 24

## Apply initial study inclusion criteria parameters
# Inclusion criteria no. 1: admission to ICU (i.e., included in `CENTER_TBI_ICU_timestamps` dataframe)
study_included_set = CENTER_TBI_ICU_timestamps.merge(CENTER_TBI_age_info,how='left')

# Inclusion criteria no. 2: patients of at least minimum age (>=)
study_included_set = study_included_set[study_included_set.Age>=MIN_AGE].reset_index(drop=True)

# Inclusion criteria no. 3: ICU stay for at least minimum ICU stay duration (>=)
study_included_set = study_included_set[study_included_set.ICUDurationHours>=MIN_STAY].reset_index(drop=True)

# Inclusion criteria no. 4: ICP monitored during ICU stay
study_included_set = study_included_set[study_included_set.GUPI.isin(CENTER_TBI_ICP_mx_info.GUPI)].reset_index(drop=True)

# Inclusion criteria no. 5: no WLST decision within minimum ICU stay duration
study_included_set = study_included_set.merge(CENTER_TBI_WLST_info[['GUPI','TimeStamp']].rename(columns={'TimeStamp':'WLSTDecisionTimeStamp'}),how='left')
study_included_set['TimeToWLSTDecisionHours'] = (study_included_set['WLSTDecisionTimeStamp']-study_included_set['ICUAdmTimeStamp']).dt.total_seconds() / 60 / 60
study_included_set = study_included_set[(study_included_set.TimeToWLSTDecisionHours.isna())|(study_included_set.TimeToWLSTDecisionHours>=MIN_STAY)].reset_index(drop=True)

# Define end timestamp based on ICU discharge timestamp or WLST decision timestamp
study_included_set['EndTimeStamp'] = study_included_set['ICUDischTimeStamp']
study_included_set.EndTimeStamp[(study_included_set.WLSTDecisionTimeStamp.notna()) & (study_included_set.WLSTDecisionTimeStamp<study_included_set.ICUDischTimeStamp)] = study_included_set.WLSTDecisionTimeStamp[(study_included_set.WLSTDecisionTimeStamp.notna()) & (study_included_set.WLSTDecisionTimeStamp<study_included_set.ICUDischTimeStamp)]

### III. Load and prepare TIL information
## Fix timestamp inaccuracies in DailyTIL dataframe based on TILTimepoint
# Load DailyTIL dataframe
daily_TIL_info = pd.read_csv(os.path.join(dir_CENTER_TBI,'DailyTIL','data.csv'),na_values = ["NA","NaN"," ", ""])

# Filter dataframe to current included study set
daily_TIL_info = daily_TIL_info[daily_TIL_info.GUPI.isin(study_included_set.GUPI)].reset_index(drop=True)

# Remove all entries without date or `TILTimepoint`
daily_TIL_info = daily_TIL_info[(daily_TIL_info.TILTimepoint!='None')|(~daily_TIL_info.TILDate.isna())].reset_index(drop=True)

# Remove all TIL entries with NA for all data variable columns
meta_columns = ['GUPI', 'TILTimepoint', 'TILDate', 'TILTime','DailyTILCompleteStatus','TotalTIL','TILFluidCalcStartDate','TILFluidCalcStartTime','TILFluidCalcStopDate','TILFluidCalcStopTime']
true_var_columns = daily_TIL_info.columns.difference(meta_columns)
daily_TIL_info = daily_TIL_info.dropna(axis=1,how='all').dropna(subset=true_var_columns,how='all').reset_index(drop=True)

# Remove all TIL entries marked as "Not Performed"
daily_TIL_info = daily_TIL_info[daily_TIL_info.DailyTILCompleteStatus!='INCPT']

# Convert dates from string to date format
daily_TIL_info.TILDate = pd.to_datetime(daily_TIL_info.TILDate,format = '%Y-%m-%d')

# For each patient, and for the overall set, calculate median TIL evaluation time
median_TILTime = daily_TIL_info.copy().dropna(subset=['GUPI','TILTime'])
median_TILTime['TILTime'] = pd.to_datetime(median_TILTime.TILTime,format = '%H:%M:%S')
median_TILTime = median_TILTime.groupby(['GUPI'],as_index=False).TILTime.aggregate('median')
overall_median_TILTime = median_TILTime.TILTime.median().strftime('%H:%M:%S')
median_TILTime['TILTime'] = median_TILTime['TILTime'].dt.strftime('%H:%M:%S')
median_TILTime = median_TILTime.rename(columns={'TILTime':'medianTILTime'})

# Iterate through GUPIs and fix `TILDate` based on `TILTimepoint` information if possible
problem_GUPIs = []
for curr_GUPI in tqdm(daily_TIL_info.GUPI.unique(),'Fixing daily TIL dates if possible'):
    curr_GUPI_daily_TIL = daily_TIL_info[(daily_TIL_info.GUPI==curr_GUPI)&(daily_TIL_info.TILTimepoint!='None')].reset_index(drop=True)
    if curr_GUPI_daily_TIL.TILDate.isna().all():
        print('Problem GUPI: '+curr_GUPI)
        problem_GUPIs.append(curr_GUPI)
        continue
    curr_date_diff = int((curr_GUPI_daily_TIL.TILDate.dt.day - curr_GUPI_daily_TIL.TILTimepoint.astype(float)).mode()[0])
    fixed_date_vector = pd.Series([pd.Timestamp('1970-01-01') + pd.DateOffset(days=dt+curr_date_diff) for dt in (curr_GUPI_daily_TIL.TILTimepoint.astype(float)-1)],index=daily_TIL_info[(daily_TIL_info.GUPI==curr_GUPI)&(daily_TIL_info.TILTimepoint!='None')].index)
    daily_TIL_info.TILDate[(daily_TIL_info.GUPI==curr_GUPI)&(daily_TIL_info.TILTimepoint!='None')] = fixed_date_vector    
    
# Replace 'None' timepoints with NaN
daily_TIL_info.TILTimepoint[daily_TIL_info.TILTimepoint=='None'] = np.nan

# Determine GUPIs with 'None' timepoints
none_GUPIs = daily_TIL_info[daily_TIL_info.TILTimepoint.isna()].GUPI.unique()

# Iterate through 'None' GUPIs and impute missing timepoint values
for curr_GUPI in none_GUPIs:
    curr_GUPI_TIL_scores = daily_TIL_info[daily_TIL_info.GUPI==curr_GUPI].reset_index(drop=True)
    non_missing_timepoint_mask = ~curr_GUPI_TIL_scores.TILTimepoint.isna()
    if non_missing_timepoint_mask.sum() != 1:
        curr_default_date = (curr_GUPI_TIL_scores.TILDate[non_missing_timepoint_mask] - pd.to_timedelta(curr_GUPI_TIL_scores.TILTimepoint.astype(float)[non_missing_timepoint_mask],unit='d')).mode()[0]
    else:
        curr_default_date = (curr_GUPI_TIL_scores.TILDate[non_missing_timepoint_mask] - timedelta(days=curr_GUPI_TIL_scores.TILTimepoint.astype(float)[non_missing_timepoint_mask].values[0])).mode()[0]
    fixed_timepoints_vector = ((curr_GUPI_TIL_scores.TILDate - curr_default_date)/np.timedelta64(1,'D')).astype(int).astype(str)
    fixed_timepoints_vector.index=daily_TIL_info[daily_TIL_info.GUPI==curr_GUPI].index
    daily_TIL_info.TILTimepoint[daily_TIL_info.GUPI==curr_GUPI] = fixed_timepoints_vector

# Convert TILTimepoint variable from string to integer
daily_TIL_info.TILTimepoint = daily_TIL_info.TILTimepoint.astype(int)

# Merge median TIL time to daily TIL dataframe
daily_TIL_info = daily_TIL_info.merge(median_TILTime,how='left',on='GUPI')

# If daily TIL assessment time is missing, first impute with patient-specific median time
daily_TIL_info.TILTime[daily_TIL_info.TILTime.isna()&~daily_TIL_info.medianTILTime.isna()] = daily_TIL_info.medianTILTime[daily_TIL_info.TILTime.isna()&~daily_TIL_info.medianTILTime.isna()]

# If daily TIL assessment time is still missing, then impute with overall-set median time
daily_TIL_info.TILTime[daily_TIL_info.TILTime.isna()] = overall_median_TILTime

# Construct daily TIL assessment timestamp
daily_TIL_info['TimeStamp'] = daily_TIL_info[['TILDate', 'TILTime']].astype(str).agg(' '.join, axis=1)
daily_TIL_info['TimeStamp'] = pd.to_datetime(daily_TIL_info['TimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

# Add WLST timestamp to daily TIL dataframe
daily_TIL_info = daily_TIL_info.merge(CENTER_TBI_WLST_info[['GUPI','TimeStamp']].rename(columns={'TimeStamp':'WLSTDecisionTimeStamp'}),how='left')

# Remove all TIL dataframe rows following WLST decision timestamp
daily_TIL_info = daily_TIL_info[daily_TIL_info.WLSTDecisionTimeStamp.isna() | (daily_TIL_info.TimeStamp<=daily_TIL_info.WLSTDecisionTimeStamp)].reset_index(drop=True)

# Sort dataframe by GUPI and TILTimepoint
daily_TIL_info = daily_TIL_info.sort_values(by=['GUPI','TILTimepoint'],ignore_index=True).drop(columns='medianTILTime')

## Corroborate and repair DailyTIL values based on other information
# Merge daily TIL dataframe onto CENTER-TBI timestamp dataframe
mod_daily_TIL_info = CENTER_TBI_ICU_timestamps[['GUPI','ICUAdmTimeStamp','ICUDischTimeStamp']].merge(daily_TIL_info,how='inner')

# Sort dataframe by GUPI and TILTimepoint
mod_daily_TIL_info = mod_daily_TIL_info.sort_values(by=['GUPI','TILTimepoint'],ignore_index=True)

# Rearrange columns for convenient viewing
first_cols = ['GUPI','TILTimepoint','TILDate','TILTime','TimeStamp','TotalTIL','ICUAdmTimeStamp','ICUDischTimeStamp','WLSTDecisionTimeStamp']
other_cols = mod_daily_TIL_info.columns.difference(first_cols).to_list()
mod_daily_TIL_info = mod_daily_TIL_info[first_cols+other_cols]

# Fix volume and dose variables if incorrectly casted as character types
fix_TIL_columns = [col for col, dt in mod_daily_TIL_info.dtypes.items() if (col.endswith('Dose')|('Volume' in col))&(dt == object)]
mod_daily_TIL_info[fix_TIL_columns] = mod_daily_TIL_info[fix_TIL_columns].replace(to_replace='^\D*$', value=np.nan, regex=True)
mod_daily_TIL_info[fix_TIL_columns] = mod_daily_TIL_info[fix_TIL_columns].apply(lambda x: x.str.replace(',','.',regex=False))
mod_daily_TIL_info[fix_TIL_columns] = mod_daily_TIL_info[fix_TIL_columns].apply(lambda x: x.str.replace('[^0-9\\.]','',regex=True))
mod_daily_TIL_info[fix_TIL_columns] = mod_daily_TIL_info[fix_TIL_columns].apply(lambda x: x.str.replace('\\.\\.','.',regex=True))
mod_daily_TIL_info[fix_TIL_columns] = mod_daily_TIL_info[fix_TIL_columns].apply(pd.to_numeric)

# Load and filter cranial surgeries dataset information to decompressive craniectomies
cran_surgeries_info = pd.read_csv(os.path.join(dir_CENTER_TBI,'SurgeriesCranial','data.csv'),na_values = ["NA","NaN"," ", ""])[['GUPI','SurgeryStartDate','SurgeryStartTime','SurgeryEndDate','SurgeryEndTime','SurgeryDescCranial','SurgeryCranialReason']]
cran_surgeries_info = cran_surgeries_info[(cran_surgeries_info.GUPI.isin(mod_daily_TIL_info.GUPI))&(cran_surgeries_info.SurgeryDescCranial.isin([7,71,72]))].reset_index(drop=True)

# Format date timestamps of cranial surgeries dataframe to match TILDate on DailyTIL dataframe
cran_surgeries_info['TILDate'] = pd.to_datetime(cran_surgeries_info['SurgeryStartDate'],format = '%Y-%m-%d')

# Find instances of cranial surgery that coincide with DailyTIL dataframe
cran_surgeries_info = cran_surgeries_info[['GUPI','TILDate']].drop_duplicates(ignore_index=True).merge(mod_daily_TIL_info[['GUPI','TILDate']].drop_duplicates(ignore_index=True),how='inner')

# Iterate through cranial surgery dataframe to document instances of decompressive craniectomy if missing
for curr_cs_row in tqdm(range(cran_surgeries_info.shape[0]), 'Fixing decompressive craniectomy instances if missing'):
    # Extract current GUPI and date
    curr_GUPI = cran_surgeries_info.GUPI[curr_cs_row]
    curr_date = cran_surgeries_info.TILDate[curr_cs_row]

    # Current TIL row corresponding to the current date and GUPI
    curr_TIL_row = mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)&(mod_daily_TIL_info.TILDate==curr_date)]

    # Ensure decompressive craniectomy markers if date and GUPI match
    if curr_TIL_row.shape[0] != 0:
        mod_daily_TIL_info.TILICPSurgeryDecomCranectomy[curr_TIL_row.index] = 1

# Load CENTER-TBI dataset decompressive craniectomy information from baseline dataset
CENTER_TBI_DC_info = pd.read_csv(os.path.join(dir_CENTER_TBI,'DemoInjHospMedHx','data.csv'),na_values = ["NA","NaN"," ", ""])[['GUPI','DecompressiveCran','DecompressiveCranLocation','DecompressiveCranReason','DecompressiveCranType','DecompressiveSize']]

# Filter rows that indicate pertinent decompresive craniectomy information
CENTER_TBI_DC_info = CENTER_TBI_DC_info[(CENTER_TBI_DC_info.GUPI.isin(mod_daily_TIL_info.GUPI))&(CENTER_TBI_DC_info.DecompressiveCran==1)].reset_index(drop=True)

# Identify GUPIs by the categorized type of decompressive craniectomy
gupis_with_DecomCranectomy = mod_daily_TIL_info[(mod_daily_TIL_info.TILICPSurgeryDecomCranectomy==1)].GUPI.unique()
gupis_with_initial_DecomCranectomy = mod_daily_TIL_info[(mod_daily_TIL_info.TILICPSurgeryDecomCranectomy==1)&(mod_daily_TIL_info.TILTimepoint==1)].GUPI.unique()
gupis_with_noninitial_DecomCranectomy = np.setdiff1d(gupis_with_DecomCranectomy, gupis_with_initial_DecomCranectomy)
gupis_with_refract_DecomCranectomy_1 = CENTER_TBI_DC_info[CENTER_TBI_DC_info.DecompressiveCranReason==2].GUPI.unique()
gupis_with_refract_DecomCranectomy_2 = np.intersect1d(np.setdiff1d(gupis_with_DecomCranectomy, CENTER_TBI_DC_info.GUPI.unique()),gupis_with_noninitial_DecomCranectomy)
gupis_with_refract_DecomCranectomy = np.union1d(gupis_with_refract_DecomCranectomy_1,gupis_with_refract_DecomCranectomy_2)
gupis_with_nonrefract_DecomCranectomy = np.setdiff1d(gupis_with_DecomCranectomy, gupis_with_refract_DecomCranectomy)

# Iterate through GUPIs with initial decompressive craniectomies and correct surgery indicators and the total TIL score
for curr_GUPI in tqdm(gupis_with_nonrefract_DecomCranectomy, 'Fixing TotalTIL and initial decompressive craniectomy indicators'):

    # Extract TIL assessments of current patient
    curr_GUPI_daily_TIL = mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].reset_index(drop=True)

    # Extract total TIL scores of current patient
    curr_TotalTIL = curr_GUPI_daily_TIL.TotalTIL

    # Extract decompressive craniectomy indicators of current patient
    curr_TILICPSurgeryDecomCranectomy = curr_GUPI_daily_TIL.TILICPSurgeryDecomCranectomy

    # Identify first TIL instance of surgery
    firstSurgInstance = curr_TILICPSurgeryDecomCranectomy.index[curr_TILICPSurgeryDecomCranectomy==1].tolist()[0]

    # Fix the decompressive craniectomy indicators of current patient
    fix_TILICPSurgeryDecomCranectomy = curr_TILICPSurgeryDecomCranectomy.copy()
    if firstSurgInstance != (len(fix_TILICPSurgeryDecomCranectomy)-1):
        fix_TILICPSurgeryDecomCranectomy[range(firstSurgInstance+1,len(fix_TILICPSurgeryDecomCranectomy))] = 0
    fix_TILICPSurgeryDecomCranectomy.index=mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].index

    # Fix the total TIL score of current patient
    fix_TotalTIL = curr_TotalTIL.copy()
    if firstSurgInstance != (len(fix_TotalTIL)-1):
        fix_TotalTIL[(fix_TILICPSurgeryDecomCranectomy.reset_index(drop=True) - curr_TILICPSurgeryDecomCranectomy).astype('bool')] = fix_TotalTIL[(fix_TILICPSurgeryDecomCranectomy.reset_index(drop=True) - curr_TILICPSurgeryDecomCranectomy).astype('bool')]-5
    fix_TotalTIL.index=mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].index

    # Place fixed vectors into modified daily TIL dataframe
    mod_daily_TIL_info.TotalTIL[(mod_daily_TIL_info.GUPI==curr_GUPI)] = fix_TotalTIL    
    mod_daily_TIL_info.TILICPSurgeryDecomCranectomy[(mod_daily_TIL_info.GUPI==curr_GUPI)] = fix_TILICPSurgeryDecomCranectomy

# Iterate through GUPIs with refractory decompressive craniectomies and correct surgery indicators and the total TIL score
for curr_GUPI in tqdm(gupis_with_refract_DecomCranectomy, 'Fixing TotalTIL and refractory decompressive craniectomy indicators'):

    # Extract TIL assessments of current patient
    curr_GUPI_daily_TIL = mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].reset_index(drop=True)

    # Extract total TIL scores of current patient
    curr_TotalTIL = curr_GUPI_daily_TIL.TotalTIL

    # Extract decompressive craniectomy indicators of current patient
    curr_TILICPSurgeryDecomCranectomy = curr_GUPI_daily_TIL.TILICPSurgeryDecomCranectomy

    # Identify first TIL instance of surgery
    try:
        firstSurgInstance = curr_TILICPSurgeryDecomCranectomy.index[curr_TILICPSurgeryDecomCranectomy==1].tolist()[0]
    except:
        firstSurgInstance = 1
        
    # Fix the decompressive craniectomy indicators of current patient
    fix_TILICPSurgeryDecomCranectomy = curr_TILICPSurgeryDecomCranectomy.copy()
    if firstSurgInstance != (len(fix_TILICPSurgeryDecomCranectomy)-1):
        fix_TILICPSurgeryDecomCranectomy[range(firstSurgInstance+1,len(fix_TILICPSurgeryDecomCranectomy))] = 1
    fix_TILICPSurgeryDecomCranectomy.index=mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].index

    # Fix the total TIL score of current patient
    fix_TotalTIL = curr_TotalTIL.copy()
    if firstSurgInstance != (len(fix_TotalTIL)-1):
        fix_TotalTIL[(fix_TILICPSurgeryDecomCranectomy.reset_index(drop=True) - curr_TILICPSurgeryDecomCranectomy).astype('bool')] = fix_TotalTIL[(fix_TILICPSurgeryDecomCranectomy.reset_index(drop=True) - curr_TILICPSurgeryDecomCranectomy).astype('bool')]+5
    fix_TotalTIL.index=mod_daily_TIL_info[(mod_daily_TIL_info.GUPI==curr_GUPI)].index

    # Place fixed vectors into modified daily TIL dataframe
    mod_daily_TIL_info.TotalTIL[(mod_daily_TIL_info.GUPI==curr_GUPI)] = fix_TotalTIL    
    mod_daily_TIL_info.TILICPSurgeryDecomCranectomy[(mod_daily_TIL_info.GUPI==curr_GUPI)] = fix_TILICPSurgeryDecomCranectomy

## Recalculate TotalTIL based on TIL item fixes
# Repair TIL item names if there are misspellings or ambiguities
old_TIL_names = ['TILHyperosmolarThearpy','TILHyperosomolarTherapyMannitolGreater2g','TILHyperosomolarTherapyHypertonicLow','TILHyperosomolarTherapyHigher','TILICPSurgeryDecomCranectomy']
new_TIL_names = ['TILMannitolLowDose','TILMannitolHighDose','TILHypertonicSalineLowDose','TILHypertonicSalineHighDose','TILICPSurgeryDecomCraniectomy']
mod_daily_TIL_info = mod_daily_TIL_info.rename(columns=dict(zip(old_TIL_names, new_TIL_names)))

# Create new column to categorise high-volume CSF drainage based on fluid drainage values
mod_daily_TIL_info['TILCSFDrainageHighVolume'] = np.where(mod_daily_TIL_info['TILFluidOutCSFDrain']>=120, 1,
                                                          np.where(mod_daily_TIL_info['TILCCSFDrainageVolume']>=120, 1,
                                                                   np.where(mod_daily_TIL_info['TILFluidOutCSFDrain']<120, 0,
                                                                            np.where(mod_daily_TIL_info['TILCCSFDrainageVolume']<120, 0,
                                                                                     np.where(mod_daily_TIL_info['TILCSFDrainage'].notna(), 0,np.nan)))))

# Create new column to categorise low-volume CSF drainage based on fluid drainage values
mod_daily_TIL_info['TILCSFDrainageLowVolume'] = np.where((mod_daily_TIL_info['TILCSFDrainageHighVolume']==0)&(mod_daily_TIL_info['TILCSFDrainage']==1), 1,
                                                         np.where((mod_daily_TIL_info['TILCSFDrainageHighVolume']==1)&(mod_daily_TIL_info['TILCSFDrainage']==1), 0,
                                                                  np.where((mod_daily_TIL_info['TILFluidOutCSFDrain']<120)&(mod_daily_TIL_info['TILFluidOutCSFDrain']!=0), 0,
                                                                           np.where((mod_daily_TIL_info['TILCCSFDrainageVolume']<120)&(mod_daily_TIL_info['TILCCSFDrainageVolume']!=0), 0,
                                                                                    np.where(mod_daily_TIL_info['TILCSFDrainage']==0, 0,
                                                                                             np.where(mod_daily_TIL_info['TILFluidOutCSFDrain']==0, 0,
                                                                                                      np.where(mod_daily_TIL_info['TILCCSFDrainageVolume']==0, 0,np.nan)))))))

# Load TIL scoring weighted key
TIL_scoring_key = pd.read_excel(os.path.join(dir_CENTER_TBI,'TIL_scoring_key.xlsx'),na_values = ["NA","NaN"," ", ""])

# Extract TIL subitem columns from Daily TIL dataframe and pivot to longer form
long_TIL_item_info = mod_daily_TIL_info[['GUPI','TILTimepoint']+TIL_scoring_key.SubItem.to_list()].melt(id_vars=['GUPI','TILTimepoint'],value_vars=TIL_scoring_key.SubItem.to_list(),var_name='SubItem',value_name='Incidence').fillna(0)

# Take maximum `Incidence` indicator per each GUPI-TILTimepoint-SubItem combination
long_TIL_item_info = long_TIL_item_info.groupby(['GUPI','TILTimepoint','SubItem'],as_index=False)['Incidence'].max()

# Merge long dataframe with TIL scoring key
long_TIL_item_info = long_TIL_item_info.merge(TIL_scoring_key,how='left')

# Calculate weigted sub-item score
long_TIL_item_info['Score'] = long_TIL_item_info['Incidence'] * long_TIL_item_info['Weight']

# Recalculate TotalTIL scores
recalc_TotalTIL_info = long_TIL_item_info.groupby(['GUPI','TILTimepoint','Item'],as_index=False)['Score'].max().groupby(['GUPI','TILTimepoint'],as_index=False)['Score'].sum().rename(columns={'Score':'TotalTIL'})

# Reformat long TIL item dataframe to wider format and append recalculated total TIL scores
clean_TIL_item_info = pd.pivot_table(long_TIL_item_info, values = 'Incidence', index=['GUPI','TILTimepoint'], columns = 'SubItem').reset_index().merge(recalc_TotalTIL_info,how='left')

# Reorder columns in clean, wide dataframe to TIL components and sum
clean_TIL_item_info = clean_TIL_item_info[['GUPI','TILTimepoint','TotalTIL']+TIL_scoring_key.SubItem.to_list()]

## Calculate study endpoints from TIL dataframe
# Sort clean dataframe to ensure correct chronological order per patient
clean_TIL_item_info = clean_TIL_item_info.sort_values(by=['GUPI','TILTimepoint'],ignore_index=True)

# Add a column designating difference in consecutive TIL scores
clean_TIL_item_info.insert(column='TimepointDiff',value=clean_TIL_item_info.groupby(['GUPI']).TILTimepoint.diff(),loc=2)

# Add a column designating difference in consecutive TIL assessment days
clean_TIL_item_info.insert(column='TotalTILDiff',value=clean_TIL_item_info.groupby(['GUPI']).TotalTIL.diff(),loc=4)

# Add a column designating sign of change-in-TIL
clean_TIL_item_info.insert(column='SignTotalTILDiff',value=clean_TIL_item_info.TotalTILDiff.apply(np.sign),loc=5)

# Add a column designating incidence of high-intensity therapies
clean_TIL_item_info.insert(column='HighIntensityTherapy',value=(clean_TIL_item_info[['TILSedationMetabolic','TILHyperventilationIntensive','TILFeverHypothermia','TILICPSurgery','TILICPSurgeryDecomCraniectomy']].sum(axis=1)>0).astype(int),loc=6)

# Add a column designating TILBasic score
clean_TIL_item_info['TILBasic'] = np.where(clean_TIL_item_info[['TILSedationMetabolic','TILHyperventilationIntensive','TILFeverHypothermia','TILICPSurgery','TILICPSurgeryDecomCraniectomy']].sum(axis=1)>0, 4,
                                           np.where(clean_TIL_item_info[['TILCSFDrainageHighVolume','TILHyperventilationModerate','TILMannitolHighDose','TILHypertonicSalineHighDose','TILFeverMildHypothermia']].sum(axis=1)>0, 3,
                                                    np.where(clean_TIL_item_info[['TILSedationHigher','TILCSFDrainageLowVolume','TILFluidLoading','TILFluidLoadingVasopressor','TILHyperventilation','TILMannitolLowDose','TILHypertonicSalineLowDose']].sum(axis=1)>0, 2,
                                                             np.where(clean_TIL_item_info[['TILPosition','TILPositionNursedFlat','TILSedation']].sum(axis=1)>0, 1,0))))

# Move TILBasic score column to front
clean_TIL_item_info.insert(column='TILBasic',value=clean_TIL_item_info.pop('TILBasic'),loc=7)

# Add TILDate to clean TIL dataframe
clean_TIL_item_info = clean_TIL_item_info.merge(mod_daily_TIL_info[['GUPI','TILTimepoint','TILDate']].drop_duplicates(),how='left')

# Move TILDate column to front
clean_TIL_item_info.insert(column='TILDate',value=clean_TIL_item_info.pop('TILDate'),loc=2)

# Convert integer column types to integer format
clean_TIL_item_info[['TILTimepoint','TotalTIL']+TIL_scoring_key.SubItem.to_list()] = clean_TIL_item_info[['TILTimepoint','TotalTIL']+TIL_scoring_key.SubItem.to_list()].astype(int)

# Inclusion criteria no. 6: Filter to patients who have non-first-day TIL values available
clean_TIL_item_info = clean_TIL_item_info[clean_TIL_item_info.GUPI.isin(clean_TIL_item_info[clean_TIL_item_info.TILTimepoint!=1].GUPI.unique())].reset_index(drop=True)
study_included_set = study_included_set[study_included_set.GUPI.isin(clean_TIL_item_info.GUPI)].sort_values(['GUPI'],ignore_index=True)

# Save cleaned TIL and study set dataframes
study_included_set.to_csv(os.path.join(form_TIL_dir,'study_included_set.csv'),index=False)
clean_TIL_item_info.to_csv(os.path.join(form_TIL_dir,'formatted_TIL_values.csv'),index=False)

### IV. Partition study set for repeated stratified k-fold cross-validation
## Prepare environment for repeated stratified k-fold cross-validation
# Establish number of repeats and folds
REPEATS = 20
FOLDS = 5

# Establish proportion of training set to set aside for validation
PROP_VALIDATION = 0.15

# Load formatted TIL dataframe
formatted_TIL_values = pd.read_csv(os.path.join(form_TIL_dir,'formatted_TIL_values.csv'))

# Create stratification label based on high-intensity treatment on days 2 - 7
outcome_stratification = formatted_TIL_values[(formatted_TIL_values.TILTimepoint>1)&(formatted_TIL_values.TILTimepoint<=7)].groupby('GUPI',as_index=False).HighIntensityTherapy.max()

## Resampling of dataset for training/testing splits
# Initialize repeated stratified k-fold cross-validator with fixed random seed
rskf = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats = REPEATS, random_state = 2023)

# Initialize empty dataframe to store repeated cross-validation information
cv_splits = pd.DataFrame(np.empty((0,5)),columns = ['REPEAT','FOLD','SET','GUPI','HighIntensityTherapy'])

# Store vectors of site codes and count quantiles
study_GUPIs = outcome_stratification.GUPI.values
study_HighIntensityTherapies = outcome_stratification.HighIntensityTherapy.values

# Iterate through cross-validator and store splits in the dataframe
iter_no = 0
for train_index, test_index in rskf.split(study_GUPIs, study_HighIntensityTherapies):
    
    # Extract current partition indices
    fold_no = (iter_no % FOLDS) + 1
    repeat_no = np.floor(iter_no/FOLDS) + 1
    
    # Extract current training and testing splits from partitioner
    GUPI_train, GUPI_test = study_GUPIs[train_index], study_GUPIs[test_index]
    HighIntensityTherapy_train, HighIntensityTherapy_test = study_HighIntensityTherapies[train_index], study_HighIntensityTherapies[test_index]

    # Organise training and testing set patients into dataframe
    train_df = pd.DataFrame({'REPEAT':int(repeat_no),'FOLD':int(fold_no),'SET':'train','GUPI':GUPI_train,'HighIntensityTherapy':HighIntensityTherapy_train})
    test_df = pd.DataFrame({'REPEAT':int(repeat_no),'FOLD':int(fold_no),'SET':'test','GUPI':GUPI_test,'HighIntensityTherapy':HighIntensityTherapy_test})
    
    # Append orgainsed partition dataframes onto compiled dataframe
    cv_splits = cv_splits.append(train_df, ignore_index = True)
    cv_splits = cv_splits.append(test_df, ignore_index = True)
    
    # Add one to iteration number
    iter_no += 1

## Resampling of training sets to set aside validation set
# Get unique repeat-fold combinations
partition_combos = cv_splits[['REPEAT','FOLD']].drop_duplicates(ignore_index=True)

# Iterate through repeat-fold combinations to access individual training sets
for curr_partition_idx in range(partition_combos.shape[0]):
    
    # Extract current repeat and fold numbers
    curr_repeat = partition_combos['REPEAT'][curr_partition_idx]
    curr_fold = partition_combos['FOLD'][curr_partition_idx]

    # Extract current training set
    curr_training_set = cv_splits[(cv_splits.SET == 'train')&(cv_splits.REPEAT == curr_repeat)&(cv_splits.FOLD == curr_fold)].reset_index(drop=True)
    
    # Initialize stratified splitter with fixed random seed
    sss = StratifiedShuffleSplit(n_splits=1, test_size=PROP_VALIDATION, random_state=int(curr_partition_idx))
    
    # Extract indices from split
    for train_index, val_index in sss.split(curr_training_set.GUPI,curr_training_set.HighIntensityTherapy):
        train_GUPIs, val_GUPIs = curr_training_set.GUPI[train_index], curr_training_set.GUPI[val_index]
    
    # Assign chosen training set GUPIs to 'val'
    cv_splits.SET[(cv_splits.REPEAT == curr_repeat)&(cv_splits.FOLD == curr_fold)&(cv_splits.SET == 'train')&(cv_splits.GUPI.isin(val_GUPIs))] = 'val'
    
# Sort cross-validation splits and force datatypes
cv_splits = cv_splits.sort_values(by=['REPEAT','FOLD','SET','GUPI'],ignore_index=True)
cv_splits[['REPEAT','FOLD','HighIntensityTherapy']] = cv_splits[['REPEAT','FOLD','HighIntensityTherapy']].astype('int')

# Save repeated cross-validation partitions
cv_splits.to_csv('../cross_validation_splits.csv',index=False)