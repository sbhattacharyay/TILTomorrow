#### Master Script 1b: Calculate summary statistics of study population ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate the representation of TILBasic scores and treatments

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
formatted_TIL_values = pd.read_csv(os.path.join(form_TIL_dir,'formatted_TIL_values.csv'))

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

