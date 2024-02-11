#### Master Script 1b: Calculate summary statistics of study population ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate the representation of TILBasic scores and treatments
# III. Calculate summary characteristics and p-values for manuscript table

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

# Custom methods
from functions.analysis import prepare_df

## Define relevant directories
# Define directory in which CENTER-TBI data is stored
dir_CENTER_TBI = '../../center_tbi/CENTER-TBI'

# Define and create subdirectory to store formatted TIL values
form_TIL_dir = os.path.join(dir_CENTER_TBI,'FormattedTIL')

# Define and create directory to store formatted tables for manuscript
table_dir = os.path.join('../','tables')
os.makedirs(table_dir,exist_ok=True)

## Define parameters for statistic calculation
# Window indices of TIL(Basic) to consider
PERF_WINDOW_INDICES = [1,2,3,4,5,6,7,10,14]

### II. Calculate the representation of TILBasic scores and treatments
## Load, prepare, and merge relevant information
# Load formatted TIL values
formatted_TIL_values = pd.read_csv(os.path.join(form_TIL_dir,'formatted_TIL_values.csv'),na_values = ["NA","NaN","NaT"," ", ""])

# Select dataframe columns corresponding to TILBasic and TILBasic treatments
formatted_TIL_values = formatted_TIL_values.drop(columns=['TimepointDiff','TotalTIL','TotalTILDiff','SignTotalTILDiff','HighIntensityTherapy','TILSedationNeuromuscular','TILFever'])

# Load patient center affiliation
CENTER_TBI_center_info = pd.read_csv(os.path.join(dir_CENTER_TBI,'DemoInjHospMedHx','data.csv'),na_values = ["NA","NaN","NaT"," ", ""])[['GUPI','SiteCode']]

# Merge patient center affiliation onto TIL dataframe
formatted_TIL_values = formatted_TIL_values.merge(CENTER_TBI_center_info,how='left')

# Melt dataframe to long form
formatted_TIL_values = formatted_TIL_values.melt(id_vars=['GUPI','TILTimepoint','TILDate','SiteCode'],var_name='Treatment',value_name='Value')

# Remove any rows with treatment value 0
formatted_TIL_values = formatted_TIL_values[(formatted_TIL_values.Treatment=='TILBasic')|(formatted_TIL_values.Value!=0)].reset_index(drop=True)

## Calculate representation counts per treatment
# Calculate number of unique patients per treatment
treatment_patient_count = formatted_TIL_values.groupby(['Treatment','Value'],as_index=False)['GUPI'].nunique()

# Add formatted label with percentage of patient population
treatment_patient_count['PatientCount'] = treatment_patient_count['GUPI'].astype(str)+' ('+((treatment_patient_count.GUPI*100)/844).round().astype(int).astype(str)+'%)'

# Calculate number of unique centres per treatment
treatment_centre_count = formatted_TIL_values.groupby(['Treatment','Value'],as_index=False)['SiteCode'].nunique()

# Add formatted label with percentage of centre number
treatment_centre_count['CentreCount'] = treatment_centre_count['SiteCode'].astype(str)+' ('+((treatment_centre_count.SiteCode*100)/51).round().astype(int).astype(str)+'%)'

# Merge patient and centre count information into single dataframe
treatment_pt_ct_count = treatment_patient_count[['Treatment','Value','PatientCount']].merge(treatment_centre_count[['Treatment','Value','CentreCount']],how='outer')

# Save to CSV for easy formatting
treatment_pt_ct_count.to_csv(os.path.join(table_dir,'patient_centre_count_per_treatment.csv'),index=False)

### III. Calculate summary characteristics and p-values for manuscript table
## Load, prepare, and merge relevant information
# Load formatted TIL values
formatted_TIL_values = pd.read_csv(os.path.join(form_TIL_dir,'formatted_TIL_values.csv'),na_values = ["NA","NaN","NaT"," ", ""])

# Filter formatted TIL values to the first week of values
formatted_TIL_values = formatted_TIL_values[(formatted_TIL_values.TILTimepoint<=7)&(formatted_TIL_values.TILTimepoint>=1)].reset_index(drop=True)

# Calculate TILmedian scores from the first week of TIL data
TILmedian_values = formatted_TIL_values.dropna(subset=['TILBasic']).groupby('GUPI',as_index=False).TILBasic.median().rename(columns={'TILBasic':'TILmedian'})

# Calculate number of ICP-therapy days in first week of ICU stay per patient
ICPTx_days = formatted_TIL_values.dropna(subset=['TILBasic']).groupby('GUPI',as_index=False).TILBasic.apply(lambda x: len(x[x!=0])).rename(columns={'TILBasic':'DaysOfTIL'})

# Load baseline patient information
CENTER_TBI_baseline_info = pd.read_csv(os.path.join(dir_CENTER_TBI,'DemoInjHospMedHx','data.csv'),na_values = ["NA","NaN","NaT"," ", ""])[['GUPI','SiteCode','Age','Sex','GCSScoreBaselineDerived','GOSE6monthEndpointDerived','ICURaisedICP','DecompressiveCranReason']]

# Add marker of refractory IC hypertension
CENTER_TBI_baseline_info['RefractoryICP'] = np.nan
CENTER_TBI_baseline_info['RefractoryICP'][(~CENTER_TBI_baseline_info.ICURaisedICP.isna())|(~CENTER_TBI_baseline_info.DecompressiveCranReason.isna())] = ((CENTER_TBI_baseline_info[(~CENTER_TBI_baseline_info.ICURaisedICP.isna())|(~CENTER_TBI_baseline_info.DecompressiveCranReason.isna())].ICURaisedICP==2)|(CENTER_TBI_baseline_info[(~CENTER_TBI_baseline_info.ICURaisedICP.isna())|(~CENTER_TBI_baseline_info.DecompressiveCranReason.isna())].DecompressiveCranReason==2)).astype(int)

# Categorise GCS into severity
CENTER_TBI_baseline_info['GCSSeverity'] = np.nan
CENTER_TBI_baseline_info.GCSSeverity[CENTER_TBI_baseline_info.GCSScoreBaselineDerived<=8] = '3-8'
CENTER_TBI_baseline_info.GCSSeverity[(CENTER_TBI_baseline_info.GCSScoreBaselineDerived>=9)&(CENTER_TBI_baseline_info.GCSScoreBaselineDerived<=12)] = '9-12'
CENTER_TBI_baseline_info.GCSSeverity[CENTER_TBI_baseline_info.GCSScoreBaselineDerived>=13] = '13-15'

# Drop unused columns
CENTER_TBI_baseline_info = CENTER_TBI_baseline_info.drop(columns=['ICURaisedICP','DecompressiveCranReason','GCSScoreBaselineDerived'])

# Load ordinal, baseline prognoses of GOSE
CENTER_TBI_baseline_prognoses = pd.read_csv(os.path.join(dir_CENTER_TBI,'APM_deepMN_compiled_test_predictions.csv'),na_values = ["NA","NaN","NaT"," ", ""]).drop(columns='Unnamed: 0')

# Convert multiclass prognoses to ordinal, threshold-level scores
prob_cols = [col for col in CENTER_TBI_baseline_prognoses if col.startswith('Pr(GOSE=')]
prob_matrix = CENTER_TBI_baseline_prognoses[prob_cols]
thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
for thresh in range(1,len(prob_cols)):
    cols_gt = prob_cols[thresh:]
    prob_gt = CENTER_TBI_baseline_prognoses[cols_gt].sum(1).values
    CENTER_TBI_baseline_prognoses['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
CENTER_TBI_baseline_prognoses = CENTER_TBI_baseline_prognoses.drop(columns=prob_cols+['TrueLabel','TUNE_IDX'])
CENTER_TBI_baseline_prognoses = CENTER_TBI_baseline_prognoses.melt(id_vars=['GUPI'],var_name='Threshold',value_name='Probability').groupby(['GUPI','Threshold'],as_index=False).Probability.mean()
CENTER_TBI_baseline_prognoses = pd.pivot_table(CENTER_TBI_baseline_prognoses, values = 'Probability', index=['GUPI'], columns = 'Threshold').reset_index()

# Convert prognostic probabilities to percentages
prog_cols = [col for col in CENTER_TBI_baseline_prognoses if col.startswith('Pr(GOSE>')]
CENTER_TBI_baseline_prognoses[prog_cols] = CENTER_TBI_baseline_prognoses[prog_cols]*100

# Load imaging lesion data from JSON extracts
CENTER_TBI_lesion_data = pd.read_csv(os.path.join(dir_CENTER_TBI,'Imaging','image_json_data','all_lesions.24.07.2023.csv'),na_values = ["NA","NaN","NaT"," ", ""]).drop(columns='Unnamed: 0').rename(columns={'subjectId':'GUPI'})

# Extract columns pertinent to desired lesion types
cols_lesion = CENTER_TBI_lesion_data.columns
desired_lesions = [col for col in cols_lesion if (col.startswith('epidural_hematoma') | col.startswith('subdural_hematoma') | col.startswith('tsah') | col.startswith('intraparenchymal_hemorrhage') | col.startswith('intraventricular_hemorrhage'))]
CENTER_TBI_lesion_data = CENTER_TBI_lesion_data[['GUPI','Imaging.Timepoint']+desired_lesions].fillna(0)

# Summarise lesion data into single marker per lesion
CENTER_TBI_lesion_data['EDH'] = (CENTER_TBI_lesion_data[[col for col in cols_lesion if col.startswith('epidural_hematoma')]].sum(axis=1) > 0).astype(int)
CENTER_TBI_lesion_data['SDH'] = (CENTER_TBI_lesion_data[[col for col in cols_lesion if col.startswith('subdural_hematoma')]].sum(axis=1) > 0).astype(int)
CENTER_TBI_lesion_data['tSAH'] = (CENTER_TBI_lesion_data[[col for col in cols_lesion if col.startswith('tsah')]].sum(axis=1) > 0).astype(int)
CENTER_TBI_lesion_data['ICH'] = (CENTER_TBI_lesion_data[[col for col in cols_lesion if (col.startswith('intraparenchymal_hemorrhage') | col.startswith('intraventricular_hemorrhage'))]].sum(axis=1) > 0).astype(int)

# Drop unused lesion information
CENTER_TBI_lesion_data = CENTER_TBI_lesion_data[CENTER_TBI_lesion_data['Imaging.Timepoint'].str.endswith('Early')][['GUPI','Imaging.Timepoint','EDH','SDH','tSAH','ICH']].reset_index(drop=True)

# Group incidences by patient to collapse lesion data
CENTER_TBI_lesion_data = CENTER_TBI_lesion_data.melt(id_vars=['GUPI','Imaging.Timepoint'],var_name='Lesion',value_name='Incidence').groupby(['GUPI','Lesion'],as_index=False).Incidence.max()
CENTER_TBI_lesion_data = pd.pivot_table(CENTER_TBI_lesion_data, values = 'Incidence', index=['GUPI'], columns = 'Lesion').reset_index()

# Load ICU admission and discharge timestamps
CENTER_TBI_ICU_adm_disch_timestamps = pd.read_csv(os.path.join(dir_CENTER_TBI,'adm_disch_timestamps.csv'),na_values = ["NA","NaN","NaT"," ", ""])[['GUPI','ICUDurationHours']]

# Convert hours to days for ICU stay duration
CENTER_TBI_ICU_adm_disch_timestamps['ICUDurationDays'] = CENTER_TBI_ICU_adm_disch_timestamps['ICUDurationHours']/24
CENTER_TBI_ICU_adm_disch_timestamps = CENTER_TBI_ICU_adm_disch_timestamps[['GUPI','ICUDurationDays']]

# Load and prepare low-resolution ICP dataframe
lo_res_ICP = pd.read_csv(os.path.join(dir_CENTER_TBI,'DailyHourlyValues','data.csv'),na_values = ["NA","NaN"," ", ""])[['GUPI','HourlyValueTimePoint','HVHour','HVICP']]

# Filter patients to study set and remove missing ICP values
lo_res_ICP = lo_res_ICP[(lo_res_ICP.GUPI.isin(formatted_TIL_values.GUPI))&(lo_res_ICP.HourlyValueTimePoint!='None')].dropna(subset=['HVICP']).reset_index(drop=True)

# Identify first day of ICP monitoring per patient
lo_res_ICP.HourlyValueTimePoint = lo_res_ICP.HourlyValueTimePoint.astype(int)
first_day_ICP_Mx = lo_res_ICP.groupby('GUPI',as_index=False).HourlyValueTimePoint.min()

# Calculate first-day mean ICP values
first_day_ICP_values = lo_res_ICP.merge(first_day_ICP_Mx,how='inner').groupby('GUPI',as_index=False).HVICP.mean()

# Merge static characteristics into single dataframe
formatted_baseline_outcome_vars = TILmedian_values.merge(ICPTx_days,how='left').merge(first_day_ICP_values,how='left').merge(CENTER_TBI_baseline_info,how='left').merge(CENTER_TBI_baseline_prognoses,how='left').merge(CENTER_TBI_lesion_data,how='left').merge(CENTER_TBI_ICU_adm_disch_timestamps,how='left')

## Characterise patients by TIL dynamism
# Ensure formatted TIL is sorted properly for difference calculation
formatted_TIL_values = formatted_TIL_values.sort_values(by=['GUPI','TILTimepoint'],ignore_index=True)

# Calculate difference in TIL in successive days
formatted_TIL_values['dTIL'] = formatted_TIL_values.groupby(['GUPI'],as_index=False).TILBasic.diff()

# Determine GUPIs of patients with and without TIL dynamism in the first week
dynamism_GUPIs = formatted_TIL_values[formatted_TIL_values.dTIL!=0].dropna(subset='dTIL').GUPI.unique()
nondynamism_GUPIs = formatted_TIL_values[~formatted_TIL_values.GUPI.isin(dynamism_GUPIs)].GUPI.unique()

# Concatenate lists of GUPIs per analysis groups
GUPI_group_keys = pd.concat([pd.DataFrame({'GUPI':dynamism_GUPIs,'Group':'Dynamic'}),pd.DataFrame({'GUPI':nondynamism_GUPIs,'Group':'Nondynamic'}),pd.DataFrame({'GUPI':formatted_TIL_values.GUPI.unique(),'Group':'Overall'})])

# Expand formatted characteristic dataframe per GUPI-group key
expanded_baseline_outcome_vars = GUPI_group_keys.merge(formatted_baseline_outcome_vars,how='left')

## Define summary characteristic columns for analysis
# Define numeric summary statistics
num_cols = ['Age','ICUDurationDays','DaysOfTIL','TILmedian','HVICP'] + prog_cols

# Define categorical summary statistics
cat_cols = ['Sex','GOSE6monthEndpointDerived','GCSSeverity','RefractoryICP','EDH','ICH','SDH','tSAH']

## Calculate basic count stats
# Population by group
n_by_group = expanded_baseline_outcome_vars.groupby('Group',as_index=False).GUPI.nunique()

# Centre count by group
centres_by_group = expanded_baseline_outcome_vars.groupby('Group',as_index=False).SiteCode.nunique()

## Calculate summary statistics for numerical variables
# Create a long dataframe of the numerical variable values
num_charset = expanded_baseline_outcome_vars[['GUPI','Group']+num_cols].melt(id_vars=['GUPI','Group'],var_name='Variable',value_name='Value').dropna().reset_index(drop=True)

# First, calculate summary statistics for each numeric variable
num_summary_stats = num_charset.groupby(['Variable','Group'],as_index=False)['Value'].aggregate({'q1':lambda x: np.quantile(x,.25),'median':np.median,'q3':lambda x: np.quantile(x,.75),'n':'count'}).reset_index(drop=True)

# Add a formatted confidence interval
num_summary_stats['FormattedCI'] = num_summary_stats['median'].round(1).astype(str)+' ('+num_summary_stats['q1'].round(1).astype(str)+'–'+num_summary_stats['q3'].round(1).astype(str)+')'

# Pivot table to wide form
num_summary_stats = num_summary_stats[['Variable','Group','FormattedCI']].pivot(columns='Group',index='Variable').reset_index().droplevel(0, axis=1).rename(columns={'':'Variable'})

# Second, calculate p-value for each numeric variable comparison and add to dataframe
num_summary_stats = num_summary_stats.merge(num_charset.groupby(['Variable'],as_index=False).apply(lambda x: stats.ttest_ind(x['Value'][x.Group=='Dynamic'].values,x['Value'][x.Group=='Nondynamic'].values,equal_var=False).pvalue).rename(columns={None:'p_val'}),how='left')

## Calculate summary statistics for categorical variables
# Create a long dataframe of the numerical variable values
cat_charset = expanded_baseline_outcome_vars[['GUPI','Group']+cat_cols].melt(id_vars=['GUPI','Group'],var_name='Variable',value_name='Value').dropna().reset_index(drop=True)
cat_charset['Value'] = cat_charset['Value'].astype(str)

# First, calculate summary characteristics for each categorical variable
cat_summary_stats = cat_charset.groupby(['Variable','Group','Value'],as_index=False).GUPI.count().rename(columns={'GUPI':'n'}).merge(cat_charset.groupby(['Variable','Group'],as_index=False).GUPI.count().rename(columns={'GUPI':'n_total'}),how='left')
cat_summary_stats['proportion'] = 100*(cat_summary_stats['n']/cat_summary_stats['n_total'])

# Add a formatted proportion entry
cat_summary_stats['FormattedProp'] = cat_summary_stats['n'].astype(str)+' ('+cat_summary_stats['proportion'].round().astype(int).astype(str)+'%)'

# Pivot table to wide form
cat_summary_stats = cat_summary_stats[['Variable','Group','Value','FormattedProp']].pivot(columns='Group',index=['Variable','Value']).reset_index().droplevel(0, axis=1)
cat_summary_stats.columns = ['Variable','Value','Dynamic','Nondynamic','Overall']

# Then, calculate p-value for each categorical variable comparison and add to dataframe
cat_summary_stats = cat_summary_stats.merge(cat_charset[cat_charset.Group.isin(['Dynamic','Nondynamic'])].groupby(['Variable'],as_index=False).apply(lambda x: stats.chi2_contingency(pd.crosstab(x["Value"],x["Group"])).pvalue).rename(columns={None:'p_val'}),how='left')

## Concatenate table results from numerical and categorical variables
# Concatenate numerical and categorical summary statistics
summary_stats = pd.concat([num_summary_stats,cat_summary_stats],ignore_index=True)

# Change the order of the columns
summary_stats = summary_stats[['Variable','Value','Overall','Dynamic','Nondynamic','p_val']]

# Save concatenated characteristic results to table dataframe as CSV for easy formatting
summary_stats.to_csv(os.path.join(table_dir,'summary_statistics.csv'),index=False)