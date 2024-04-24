# TILTomorrow: dynamic, ordinal, and full-context prediction of next-day treatment intensity for TBI patients in the ICU
[TILTomorrow today: dynamic factors predicting changes in intracranial pressure treatment intensity after traumatic brain injury](LINK)

## Contents

- [Overview](#overview)
- [Abstract](#abstract)
- [Code](#code)
- [License](./LICENSE)
- [Citation](#citation)

## Overview

This repository contains the code underlying the article entitled **TILTomorrow today: dynamic factors predicting changes in intracranial pressure treatment intensity after traumatic brain injury** from the Collaborative European NeuroTrauma Effectiveness Research in TBI ([CENTER-TBI](https://www.center-tbi.eu/)) consortium. In this file, we present the abstract, to outline the motivation for the work and the findings, and then a brief description of the code with which we generate these finding and achieve this objective.\
\
The code in this repository is commented throughout to provide a description of each step alongside the code which achieves it.

## Abstract
Practices for controlling intracranial pressure (ICP) in traumatic brain injury (TBI) patients admitted to the intensive care unit (ICU) vary considerably between centres. To help understand the rational basis for such variance in care, this study aims to identify the patient-level determinants of changes in ICP management. We extract all heterogeneous data (2,008 pre-ICU and ICU variables) collected from a prospective cohort (*n*=844, 51 ICUs) of ICP-monitored TBI patients in the Collaborative European NeuroTrauma Effectiveness Research in TBI (CENTER-TBI) study. We develop the TILTomorrow modelling strategy, which leverages recurrent neural networks to map a token-embedded time series representation of all variables (including missing values) to an ordinal, dynamic prediction of the following day’s five-category therapy intensity level (TIL<sup>(Basic)</sup>) score. With 20 repeats of 5-fold cross-validation, we train TILTomorrow on different variable sets and apply the TimeSHAP (temporal extension of SHapley Additive exPlanations) algorithm to estimate variable contributions towards next-day changes in TIL<sup>(Basic)</sup>. Based on Somers’ Dxy, the full range of variables explains 68% (95% CI: 65–72%) of the ordinal variance in next-day changes in TIL<sup>(Basic)</sup> on day one and up to 51% (95% CI: 45–56%) thereafter, when changes in TIL<sup>(Basic)</sup> become less frequent. Up to 81% (95% CI: 78–85%) of this explanation can be derived from non-treatment variables (i.e., markers of pathophysiology and injury severity), but the prior trajectory of ICU management significantly improves prediction of future de-escalations in ICP-targeted treatment. Whilst there is no significant difference in the predictive discriminability (i.e., area under receiver operating characteristic curve [AUC]) between next-day escalations (0.80 [95% CI: 0.77–0.84]) and de-escalations (0.79 [95% CI: 0.76–0.82]) in TIL<sup>(Basic)</sup> after day two, we find specific variable associations to be more robust with de-escalations. The most important predictors of day-to-day changes in ICP management include preceding treatments, age, space-occupying lesions, ICP, metabolic derangements, and neurological function. Serial protein biomarkers were also important and offer a potential addition to the clinical armamentarium for assessing therapeutic needs. Approximately half of the ordinal variance in day-to-day changes in TIL<sup>(Basic)</sup> after day two remains unexplained, underscoring the significant contribution of unmeasured factors or clinicians’ personal preferences in ICP treatment. At the same time, specific dynamic markers of pathophysiology associate strongly with changes in treatment intensity and, upon mechanistic investigation, may improve the precision and timing of future care.


## Code
All of the code used in this work can be found in the `./scripts` directory as Python (`.py`), R (`.R`), or bash (`.sh`) scripts. Moreover, custom classes have been saved in the `./scripts/classes` sub-directory, custom functions have been saved in the `./scripts/functions` sub-directory, and custom PyTorch models have been saved in the `./scripts/models` sub-directory.

### 1. Extract and characterise study sample from CENTER-TBI dataset

<ol type="a">
  <li><h4><a href="scripts/01a_prepare_study_sample.py">Extract and prepare study sample from CENTER-TBI dataset</a></h4> In this <code>.py</code> file, we extract the study sample from the CENTER-TBI dataset and filter patients by our study criteria. We also clean and format information pertaining to the <a href="https://doi.org/10.1089/neu.2023.0377">Therapy Intensity Level (TIL)</a> scores. We also create 100 partitions for 20-repeats of 5-fold cross-validation, and save the splits into a dataframe for subsequent scripts.</li>
  <li><h4><a href="scripts/01b_calculate_summary_stats.py">Calculate summary statistics of study population</a></h4> In this <code>.py</code> file, we characterise the study dataset by calculating summary statistics for the manuscript. </li>
</ol>

### 2. Tokenise all CENTER-TBI variables and place into discretised ICU stay time windows

<ol type="a">
  <li><h4><a href="scripts/02a_format_CENTER_TBI_data_for_tokenisation.py">Format CENTER-TBI data for tokenisation</a></h4> In this <code>.py</code> file, we extract all heterogeneous types of variables from CENTER-TBI and fix erroneous timestamps and formats.</li>
  <li><h4><a href="scripts/02b_convert_ICU_stays_into_tokenised_sets.py">Convert full patient information from ICU stays into tokenised sets</a></h4> In this <code>.py</code> file, we convert all CENTER-TBI variables into tokens depending on variable type and compile full dictionaries of tokens across the full dataset. This is run, with multi-array indexing, on the HPC using a <a href="scripts/02b_convert_ICU_stays_into_tokenised_sets.sh">bash script</a>. </li>
  <li><h4><a href="scripts/02c_characterise_tokens.py">Create a dictionary of all tokens in study and characterise tokens in each patient's ICU stay</a></h4> In this <code>.py</code> file, we categorise and characterise the tokens in each patient's ICU stay in our study. </li>
</ol>

### 3. Train and evaluate full-context ordinal-trajectory-generating models

<ol type="a">
  <li><h4><a href="scripts/03a_train_models.py">Train dynamic all-variable-based TILTomorrow models</a></h4> In this <code>.py</code> file, we train the trajectory-generating models across the repeated cross-validation splits and the hyperparameter configurations. This is run, with multi-array indexing, on the HPC using a <a href="scripts/03a_train_models.sh">bash script</a>.</li>
  <li><h4><a href="scripts/03b_compile_model_outputs.py"> Extract outputs from TILTomorrow models and prepare for bootstrapping-based dropout</a></h4> In this <code>.py</code> file, we compile the validation and testing set trajectories generated by the models. We determine the top-performing tuning configurations based on validation set calibration and discrimination. We also create bootstrapping resamples for dropping out poorly calibrated configurations.</li>
  <li><h4><a href="scripts/03c_validation_set_bootstrapping_for_dropout.py"> Calculate validation set calibration and discrimination for dropout</a></h4> In this <code>.py</code> file, we calculate validation set trajectory calibration and discrimination based on provided bootstrapping resample row index. This is run, with multi-array indexing, on the HPC using a <a href="scripts/03c_validation_set_bootstrapping_for_dropout.sh">bash script</a>.</li>
  <li><h4><a href="scripts/03d_dropout_configurations.py"> Compile validation set performance results for configuration dropout of TILTomorrow models </a></h4> In this <code>.py</code> file, we compile and save bootstrapped validation set performance dataframes. Futhermore, the hyperparameter optimisation results are  Visualised.</li>
  <li><h4><a href="scripts/03e_model_calibration.py"> Assess post-processing calibration methods for remaining TILTomorrow configurations</a></h4> In this <code>.py</code> file, we recalibrate the TILTomorrow model and save the performance measure before and after. This is run, with multi-array indexing, on the HPC using a <a href="scripts/03e_model_calibration.sh">bash script</a>.</li>
  <li><h4><a href="scripts/03f_compile_calibration_results.py"> Compile and examine calibration performance metrics </a></h4> In this <code>.py</code> file, we compile all the calibration performance metrics and calculate the average effect of post-processing calibration methods on each tuning configuration. We also create bootstrapping resamples for calculating testing set performance metrics.</li>
  <li><h4><a href="scripts/03g_testing_set_performance_metrics.py">  Calculate calibrated testing set calibration and discrimination for statistical inference</a></h4> In this <code>.py</code> file, we calculate testing set trajectory calibration and discrimination based on provided bootstrapping resample row index. This is run, with multi-array indexing, on the HPC using a <a href="scripts/03g_testing_set_performance_metrics.sh">bash script</a>.</li>
  <li><h4><a href="scripts/03h_testing_set_confidence_intervals.py"> Compile testing set performance results for statistical inference of TILTomorrow models</a></h4> In this <code>.py</code> file, we compile the performance metrics and summarise them across bootstrapping resamples to define the 95% confidence intervals for statistical inference.</li>  
</ol>

### 4. Sensitivity analysis

<ol type="a">
  <li><h4><a href="scripts/04a_train_sensitivity_analysis_models.py">Train dynamic TILTomorrow models with focused input sets for sensitivity analysis</a></h4> In this <code>.py</code> file, train new models based on hyperparameters specific to the sensitivity analysis. This is run, with multi-array indexing, on the HPC using a <a href="scripts/04a_train_sensitivity_analysis_models.sh">bash script</a>.</li>
  <li><h4><a href="scripts/04b_sensitivity_analysis_performance_metrics.py">Compile generated trajectories across repeated cross-validation and different hyperparameter configurations</a></h4> In this <code>.py</code> file, we calculate test set calibration and discrimination based on provided bootstrapping resample row index. This is run, with multi-array indexing, on the HPC using a <a href="scripts/04b_sensitivity_analysis_performance_metrics.sh">bash script</a>. </li>
  <li><h4><a href="scripts/04c_sensitivity_analysis_confidence_intervals.py">Compile testing set performance results for statistical inference of sensitivity analysis TILTomorrow models </a></h4> In this <code>.py</code> file, we compile the performance metrics and summarise them across bootstrapping resamples to define the 95% confidence intervals for statistical inference. </li>
</ol>

### 5. Interpret variable effects 

<ol type="a">
  <li><h4><a href="scripts/05a_compile_relevance_layer_values.py">Extract relevance layer values from trained TILTomorrow models</a></h4> In this <code>.py</code> file, we extract and summarise the learned weights from the model relevance layers (trained as PyTorch Embedding layers).</li>
  <li><h4><a href="scripts/05b_prepare_for_TimeSHAP.py">Prepare environment to calculate TimeSHAP for TILTomorrow models </a></h4> In this <code>.py</code> file, we define and identify significant transitions in individual patient trajectories, partition them for bootstrapping, and calculate summarised testing set trajectory information in preparation for TimeSHAP feature contribution calculation. </li>
  <li><h4><a href="scripts/05c_calculate_TimeSHAP.py">Calculating TimeSHAP for TILTomorrow models in parallel</a></h4> In this <code>.py</code> file, we calculate variable and time-window TimeSHAP values for each individual's significant transitions. This is run, with multi-array indexing, on the HPC using a <a href="scripts/05c_calculate_TimeSHAP.sh">bash script</a>.</li>
  <li><h4><a href="scripts/05d_compile_TimeSHAP_values.py">Compile TimeSHAP values calculated in parallel </a></h4> In this <code>.py</code> file, we load all the calculated TimeSHAP values and summarise them for population-level variable and time-window analysis. </li>
  </ol>

### 6. Performance evaluation focused on prediction of transitions

<ol type="a">
  <li><h4><a href="scripts/06a_trans_prediction_performance_metrics.py">Calculate testing set calibration and discrimination performance metrics at prediction of transitions </a></h4> In this <code>.py</code> file, we calculate testing set calibration and discrimination based on provided bootstrapping resample row index. This is run, with multi-array indexing, on the HPC using a <a href="scripts/06a_trans_prediction_performance_metrics.sh">bash script</a>.</li>
  <li><h4><a href="scripts/06b_trans_prediction_confidence_intervals.py">Compile testing set performance results for statistical inference of transition prediction analysis TILTomorrow models</a></h4> In this <code>.py</code> file, we compile the saved bootstraped testing set performances and calculate 95% confidence intervals. </li>
  </ol>
  
### 7. [Visualise study results for manuscript](scripts/07_manuscript_visualisations.R)
In this `.R` file, we produce the figures for the manuscript and the supplementary figures. The large majority of the quantitative figures in the manuscript are produced using the `ggplot` package.

## Citation
```
```
