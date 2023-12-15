#### Master Script 06a: Calculate testing set calibration and discrimination performance metrics at focused transition points ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Create grid of post-hoc analysis parameters if it does not yet exist
# III. Calculate testing set calibration and discrimination based on provided bootstrapping resample row index

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
import multiprocessing
from scipy import stats
from pathlib import Path
from shutil import rmtree
from ast import literal_eval
import matplotlib.pyplot as plt
from scipy.special import logit
from collections import Counter
from argparse import ArgumentParser
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

# StatsModel methods
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from statsmodels.nonparametric.smoothers_lowess import lowess

# Custom methods
from functions.model_building import load_model_outputs
from functions.analysis import thresh_trans, prepare_df, calc_ORC, calc_Somers_D, calc_thresh_AUC, calc_thresh_calibration, calc_test_thresh_calib_curves

## Define parameters for model training
# Set version code
VERSION = 'v2-0'

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Window indices at which to calculate performance metrics
PERF_WINDOW_INDICES = [1,2,3,4,5,6,9,13]

## Define and create relevant directories
# Define directory in which CENTER-TBI data is stored
dir_CENTER_TBI = '/home/sb2406/rds/hpc-work/CENTER-TBI'

# Define and create subdirectory to store formatted TIL values
form_TIL_dir = os.path.join(dir_CENTER_TBI,'FormattedTIL')

# Define model output directory based on version code
model_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_outputs',VERSION)

# Define model performance directory based on version code
model_perf_dir = os.path.join('/home/sb2406/rds/hpc-work','TILTomorrow_model_performance',VERSION)

# Define and create subdirectory to store testing set bootstrapping results
test_bs_dir = os.path.join(model_perf_dir,'testing_set_bootstrapping')

# Define and create subdirectory to store testing set bootstrapping results for post-hoc analysis
post_hoc_bs_dir = os.path.join(model_perf_dir,'post_hoc_bootstrapping')
os.makedirs(post_hoc_bs_dir,exist_ok=True)

## Load fundamental information for model training
# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
partitions = cv_splits[['REPEAT','FOLD']].drop_duplicates().reset_index(drop=True)

# Load parametric grid corresponding to sensitivity analysis
sens_analysis_grid = pd.read_csv(os.path.join(model_dir,'sens_analysis_grid.csv'))[['SENS_IDX','DROPOUT_VARS']].drop_duplicates(ignore_index=True)

# Load bootstrapping resample dataframe for testing set performance
bs_resamples = pd.read_pickle(os.path.join(model_perf_dir,'test_performance_bs_resamples.pkl'))

### II. Create grid of post-hoc analysis parameters if it does not yet exist
# If post-hoc analysis dataframe doesn't exist, create it
if not (os.path.exists(os.path.join(model_perf_dir,'post_hoc_grid.csv'))):
    
    # Create parameters for post-hoc analysis
    post_hoc_parameters = {'TRANSITION_TYPE':['Decrease','Stasis','Increase'],
                           'THRESHOLD_LEVEL':[False,True]}

    # Convert parameter dictionary to dataframe
    post_hoc_grid = pd.DataFrame([row for row in itertools.product(*post_hoc_parameters.values())],columns=post_hoc_parameters.keys())

    # Add a single row corresponding to threshold-level performance across all transitions
    post_hoc_grid = pd.concat([post_hoc_grid,pd.DataFrame({'TRANSITION_TYPE':'All_transitions','THRESHOLD_LEVEL':True},index=[0])],ignore_index=True)

    # Filter out rows corresponding to implausible combinations
    post_hoc_grid = post_hoc_grid[(post_hoc_grid.TRANSITION_TYPE!='Stasis')|(~post_hoc_grid.THRESHOLD_LEVEL)].reset_index(drop=True)

    # Define an index for each post-hoc analysis parametric combination
    post_hoc_grid.insert(0,'POST_HOC_IDX',list(range(1,post_hoc_grid.shape[0]+1)))

    # Save parametrics grid to model performance directory
    post_hoc_grid.to_csv(os.path.join(model_perf_dir,'post_hoc_grid.csv'),index=False)

else:
    
    # Load prepared post-hoc parametric grid
    post_hoc_grid = pd.read_csv(os.path.join(model_perf_dir,'post_hoc_grid.csv'))    

### III. Calculate testing set calibration and discrimination based on provided bootstrapping resample row index
# Argument-induced bootstrapping functions
def main(array_task_id):
    
    # Extract current bootstrapping resample parameters
    curr_rs_idx = bs_resamples.RESAMPLE_IDX[array_task_id]
    curr_GUPIs = bs_resamples.GUPIs[array_task_id]
    
    ## Prepare and compile testing set outputs from different variable-set models
    # Load full-model compiled testing set outputs
    calib_TILBasic_test_outputs = pd.read_pickle(os.path.join(model_dir,'TomorrowTILBasic_compiled_test_calibrated_outputs.pkl'))
    calib_TILBasic_test_outputs['DROPOUT_VARS'] = 'none'

    # Load dropped-variable compiled testing set outputs
    sens_calib_TILBasic_test_outputs = pd.read_pickle(os.path.join(model_dir,'sens_analysis_TomorrowTILBasic_compiled_test_calibrated_outputs.pkl')).merge(sens_analysis_grid,how='left').drop(columns='SENS_IDX')

    # Compile full-model and dropped-variable outputs
    compiled_test_outputs = pd.concat([calib_TILBasic_test_outputs,sens_calib_TILBasic_test_outputs],ignore_index=True)

    # Drop logit columns
    logit_cols = [col for col in compiled_test_outputs if col.startswith('z_TILBasic=')]
    compiled_test_outputs = compiled_test_outputs.drop(columns=logit_cols)

    # Calculate intermediate values for TomorrowTILBasic testing set outputs
    prob_cols = [col for col in compiled_test_outputs if col.startswith('Pr(TILBasic=')]
    prob_matrix = compiled_test_outputs[prob_cols]
    prob_matrix.columns = list(range(prob_matrix.shape[1]))
    index_vector = np.array(list(range(prob_matrix.shape[1])), ndmin=2).T
    compiled_test_outputs['ExpectedValue'] = np.matmul(prob_matrix.values,index_vector)
    compiled_test_outputs['PredLabel'] = prob_matrix.idxmax(axis=1)

    # Load trivial, no information outputs
    no_info_TILBasic_outputs = pd.read_pickle(os.path.join(model_dir,'no_info_TomorrowTILBasic_compiled_outputs.pkl'))
    no_info_TILBasic_outputs['DROPOUT_VARS'] = 'last_TIL_only'

    # Compile full-model, dropped-variable, and last-TIL outputs
    compiled_test_outputs = pd.concat([compiled_test_outputs,no_info_TILBasic_outputs],ignore_index=True)

    # Filter testing set outputs to current GUPI set
    compiled_test_outputs = compiled_test_outputs[compiled_test_outputs.GUPI.isin(curr_GUPIs)].reset_index(drop=True)

    # Filter testing set outputs to optimal tuning configuration
    compiled_test_outputs = compiled_test_outputs[compiled_test_outputs.TUNE_IDX==332].reset_index(drop=True)

    ## Determine points of model outputs corresponding to a transition in TILBasic
    # Load formatted TIL values
    formatted_TIL_values = pd.read_csv(os.path.join(form_TIL_dir,'formatted_TIL_values.csv'))[['GUPI','TILTimepoint','TILBasic']].rename(columns={'TILTimepoint':'WindowIdx'})

    # Merge available TIL values onto study windows
    merged_TIL_values = compiled_test_outputs[['GUPI','WindowIdx','TrueLabel']].drop_duplicates(ignore_index=True).merge(formatted_TIL_values,how='left')

    # Fill in missing current TILBasic values by using the last available TILBasic assessment
    merged_TIL_values['TILBasic'] = merged_TIL_values.groupby(['GUPI'],as_index=False).TILBasic.ffill()

    # Mark points of stasis, decrease, and increase
    merged_TIL_values['Stasis'] = (merged_TIL_values['TrueLabel'] == merged_TIL_values['TILBasic']).astype(int)
    merged_TIL_values['Decrease'] = (merged_TIL_values['TrueLabel'] < merged_TIL_values['TILBasic']).astype(int)
    merged_TIL_values['Increase'] = (merged_TIL_values['TrueLabel'] > merged_TIL_values['TILBasic']).astype(int)

    # Apply function to determine transitions across specific thresholds
    merged_TIL_values = thresh_trans(merged_TIL_values)

    # Add transition indicators to the test-set outputs
    compiled_test_outputs = compiled_test_outputs.merge(merged_TIL_values.drop(columns=['TILBasic']),how='left')

    ## Prepare model outputs based on window indices
    # Prepare testing set output dataframes for performance calculation
    filt_test_outputs = prepare_df(compiled_test_outputs,PERF_WINDOW_INDICES)

    ## Iterate through post-hoc parametric grid and calculate testing set performance metrics
    # Create empty lists to store performance metrics
    test_scalar_metrics = []
    test_thresh_calibration_curves = []

    # Iterate through all unique post-hoc analysis parameters
    for curr_ph_idx in tqdm(post_hoc_grid.POST_HOC_IDX.unique(),'Calculating testing set performance metrics for each post-hoc analysis index'):

        # Extract parameters corresponding to current post-hoc analysis index
        curr_trans_type = post_hoc_grid[(post_hoc_grid.POST_HOC_IDX==curr_ph_idx)].TRANSITION_TYPE.values[0]
        curr_thresh_level = post_hoc_grid[(post_hoc_grid.POST_HOC_IDX==curr_ph_idx)].THRESHOLD_LEVEL.values[0]

        # Filter testing set outputs based on current transition type
        if curr_trans_type == 'Decrease':
            curr_test_outputs = filt_test_outputs[filt_test_outputs.Decrease==1].reset_index(drop=True)

        elif curr_trans_type == 'Stasis':
            curr_test_outputs = filt_test_outputs[filt_test_outputs.Stasis==1].reset_index(drop=True)

        elif curr_trans_type == 'Increase':
            curr_test_outputs = filt_test_outputs[filt_test_outputs.Increase==1].reset_index(drop=True)

        elif curr_trans_type == 'All_transitions':
            curr_test_outputs = filt_test_outputs[filt_test_outputs.Stasis!=1].reset_index(drop=True)

        # Calculate performance metrics based on levels of analysis
        if curr_thresh_level:
            # Extract names of trans-threshold markers in current test output dataframe
            trans_cols = [col for col in curr_test_outputs if col.startswith('TransTILBasic>')]
            
            # Iterate through trans-threhsold markers
            for curr_trans_marker in trans_cols:

                # Calculate ORCs of TIL-Basic model on testing set outputs
                curr_ORC = curr_test_outputs[curr_test_outputs[curr_trans_marker]==1].groupby('DROPOUT_VARS',as_index=True).apply(lambda x: calc_ORC(x,PERF_WINDOW_INDICES,True,'Calculating testing set ORC at points of transition')).reset_index().drop(columns=['level_1'])
                curr_ORC.insert(3,'THRESHOLD',['None' for idx in range(curr_ORC.shape[0])])

                # Calculate Somers' D of TIL-Basic model on testing set outputs
                curr_Somers_D = curr_test_outputs[curr_test_outputs[curr_trans_marker]==1].groupby('DROPOUT_VARS',as_index=True).apply(lambda x: calc_Somers_D(x,PERF_WINDOW_INDICES,True,'Calculating testing set Somers D at points of transition')).reset_index().drop(columns=['level_1'])
                curr_Somers_D.insert(3,'THRESHOLD',['None' for idx in range(curr_Somers_D.shape[0])])

                # Calculate threshold-level AUC of TIL-Basic model on testing set outputs
                curr_thresh_AUC = curr_test_outputs[curr_test_outputs[curr_trans_marker]==1].groupby('DROPOUT_VARS',as_index=True).apply(lambda x: calc_thresh_AUC(x,PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level AUC at points of transition')).reset_index().drop(columns=['level_1'])

                # Calculate threshold-level calibration metrics of TIL-Basic model on testing set outputs
                curr_thresh_calibration = curr_test_outputs[curr_test_outputs[curr_trans_marker]==1].groupby('DROPOUT_VARS',as_index=True).apply(lambda x: calc_thresh_calibration(x,PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level calibration metrics at points of transition')).reset_index().drop(columns=['level_1'])

                # Calculate threshold-level calibration curves of TIL-Basic model on testing set outputs
                curr_thresh_calibration_curves = curr_test_outputs[curr_test_outputs[curr_trans_marker]==1].groupby('DROPOUT_VARS',as_index=True).apply(lambda x: calc_test_thresh_calib_curves(x,PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level calibration metrics at points of transition')).reset_index().drop(columns=['level_1'])
                
                # Add macro-averages to threshold-level AUCs
                macro_average_thresh_AUCs = curr_thresh_AUC.groupby(['DROPOUT_VARS','TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
                macro_average_thresh_AUCs.insert(3,'THRESHOLD',['Average' for idx in range(macro_average_thresh_AUCs.shape[0])])
                curr_thresh_AUC = pd.concat([curr_thresh_AUC,macro_average_thresh_AUCs],ignore_index=True).sort_values(by=['DROPOUT_VARS','TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)

                # Add macro-averages to threshold-level calibration metrics
                macro_average_thresh_calibration = curr_thresh_calibration.groupby(['DROPOUT_VARS','TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
                macro_average_thresh_calibration.insert(3,'THRESHOLD',['Average' for idx in range(macro_average_thresh_calibration.shape[0])])
                curr_thresh_calibration = pd.concat([curr_thresh_calibration,macro_average_thresh_calibration],ignore_index=True).sort_values(by=['DROPOUT_VARS','TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)
            
                # Compile scalar metrics into single dataframe and label
                curr_scalars = pd.concat([curr_ORC,curr_Somers_D,curr_thresh_AUC,curr_thresh_calibration],ignore_index=True)
                curr_scalars['THRESHOLD_LEVEL'] = curr_thresh_level
                curr_scalars['TRANS_THRESHOLD'] = curr_trans_marker
                curr_scalars['TRANSITION_TYPE'] = curr_trans_type

                # Label current calibration curves
                curr_thresh_calibration_curves['THRESHOLD_LEVEL'] = curr_thresh_level
                curr_thresh_calibration_curves['TRANS_THRESHOLD'] = curr_trans_marker
                curr_thresh_calibration_curves['TRANSITION_TYPE'] = curr_trans_type

                # Append dataframes to running lists
                test_scalar_metrics.append(curr_scalars)
                test_thresh_calibration_curves.append(curr_thresh_calibration_curves)

        else:
            # Calculate ORCs of TIL-Basic model on testing set outputs
            curr_ORC = curr_test_outputs.groupby('DROPOUT_VARS',as_index=True).apply(lambda x: calc_ORC(x,PERF_WINDOW_INDICES,True,'Calculating testing set ORC at points of transition')).reset_index().drop(columns=['level_1'])
            curr_ORC.insert(3,'THRESHOLD',['None' for idx in range(curr_ORC.shape[0])])

            # Calculate Somers' D of TIL-Basic model on testing set outputs
            curr_Somers_D = curr_test_outputs.groupby('DROPOUT_VARS',as_index=True).apply(lambda x: calc_Somers_D(x,PERF_WINDOW_INDICES,True,'Calculating testing set Somers D at points of transition')).reset_index().drop(columns=['level_1'])
            curr_Somers_D.insert(3,'THRESHOLD',['None' for idx in range(curr_Somers_D.shape[0])])

            # Calculate threshold-level AUC of TIL-Basic model on testing set outputs
            curr_thresh_AUC = curr_test_outputs.groupby('DROPOUT_VARS',as_index=True).apply(lambda x: calc_thresh_AUC(x,PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level AUC at points of transition')).reset_index().drop(columns=['level_1'])

            # Calculate threshold-level calibration metrics of TIL-Basic model on testing set outputs
            curr_thresh_calibration = curr_test_outputs.groupby('DROPOUT_VARS',as_index=True).apply(lambda x: calc_thresh_calibration(x,PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level calibration metrics at points of transition')).reset_index().drop(columns=['level_1'])

            # Calculate threshold-level calibration curves of TIL-Basic model on testing set outputs
            curr_thresh_calibration_curves = curr_test_outputs.groupby('DROPOUT_VARS',as_index=True).apply(lambda x: calc_test_thresh_calib_curves(x,PERF_WINDOW_INDICES,True,'Calculating testing set threshold-level calibration metrics at points of transition')).reset_index().drop(columns=['level_1'])
            
            # Add macro-averages to threshold-level AUCs
            macro_average_thresh_AUCs = curr_thresh_AUC.groupby(['DROPOUT_VARS','TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
            macro_average_thresh_AUCs.insert(3,'THRESHOLD',['Average' for idx in range(macro_average_thresh_AUCs.shape[0])])
            curr_thresh_AUC = pd.concat([curr_thresh_AUC,macro_average_thresh_AUCs],ignore_index=True).sort_values(by=['DROPOUT_VARS','TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)

            # Add macro-averages to threshold-level calibration metrics
            macro_average_thresh_calibration = curr_thresh_calibration.groupby(['DROPOUT_VARS','TUNE_IDX','WINDOW_IDX','METRIC'],as_index=False).VALUE.mean()
            macro_average_thresh_calibration.insert(3,'THRESHOLD',['Average' for idx in range(macro_average_thresh_calibration.shape[0])])
            curr_thresh_calibration = pd.concat([curr_thresh_calibration,macro_average_thresh_calibration],ignore_index=True).sort_values(by=['DROPOUT_VARS','TUNE_IDX','WINDOW_IDX','THRESHOLD']).reset_index(drop=True)
        
            # Compile scalar metrics into single dataframe and label
            curr_scalars = pd.concat([curr_ORC,curr_Somers_D,curr_thresh_AUC,curr_thresh_calibration],ignore_index=True)
            curr_scalars['THRESHOLD_LEVEL'] = curr_thresh_level
            curr_scalars['TRANS_THRESHOLD'] = 'None'
            curr_scalars['TRANSITION_TYPE'] = curr_trans_type

            # Label current calibration curves
            curr_thresh_calibration_curves['THRESHOLD_LEVEL'] = curr_thresh_level
            curr_thresh_calibration_curves['TRANS_THRESHOLD'] = 'None'
            curr_thresh_calibration_curves['TRANSITION_TYPE'] = curr_trans_type

            # Append dataframes to running lists
            test_scalar_metrics.append(curr_scalars)
            test_thresh_calibration_curves.append(curr_thresh_calibration_curves)

    # Concatenate lists of dataframes to produce single 
    test_scalar_metrics = pd.concat(test_scalar_metrics,ignore_index=True)
    test_thresh_calibration_curves = pd.concat(test_thresh_calibration_curves,ignore_index=True)

    ## Save performance metrics from current resample's testing outputs
    # Save scalar metrics of TIL-Basic model from testing set outputs
    test_scalar_metrics.to_pickle(os.path.join(post_hoc_bs_dir,'post_hoc_test_calibrated_metrics_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

    # Save threshold-level calibration curves of TIL-Basic model from testing set outputs
    test_thresh_calibration_curves.to_pickle(os.path.join(post_hoc_bs_dir,'post_hoc_test_calibrated_calibration_curves_rs_'+str(curr_rs_idx).zfill(4)+'.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)
