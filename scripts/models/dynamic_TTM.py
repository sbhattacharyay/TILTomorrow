# Fundamental libraries
import os
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
from collections import Counter
warnings.filterwarnings(action="ignore")

# PyTorch, PyTorch.Text, and Lightning-PyTorch methods
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# SciKit-Learn methods
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Define dynamic TILTomorrow model
class TILTomorrow_model(pl.LightningModule):
    def __init__(self,n_tokens,latent_dim,embed_dropout,rnn_type,hidden_dim,rnn_layers,outcome_name,learning_rate,class_weights,targets,mask_indices):
        """
        Args:
            n_tokens (int): Size of vocabulary
            latent_dim (int): Number of dimensions to which tokens are embedded
            embed_dropout (float): Probability of dropout layer on embedding vectors
            rnn_type (string, 'LSTM' or 'GRU'): Identify RNN architecture type
            hidden_dim (int): Number of dimensions in the RNN hidden state (output dimensionality of RNN)
            rnn_layers (int): Number of recurrent layers
            outcome_name (string, 'TomorrowHighIntensityTherapy' or 'TomorrowTILBasic'): Name of day-to-day outcome to use during model training
            learning_rate (float): Learning rate for ADAM optimizer
            class_weights (boolean): identifies whether loss should be weighted against class frequency
            targets (NumPy array): if class_weights == True, provides the class labels of the training set
            mask_indices (list): Provides indices to mask out from the embedding layer
        """
        super(TILTomorrow_model, self).__init__()
        
        self.save_hyperparameters()
        
        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        self.dropout = embed_dropout
        self.outcome_name = outcome_name
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.targets = targets
        self.mask_indices = mask_indices
        self.mask_indices.sort()
        
        self.embedX = nn.Embedding(n_tokens, latent_dim)
        self.embedW = nn.Embedding(n_tokens, 1)
        self.embed_Dropout = nn.Dropout(p = embed_dropout)
        
        if rnn_type == 'LSTM':
            self.rnn_module = nn.LSTM(input_size = latent_dim, hidden_size = hidden_dim, num_layers = rnn_layers)
        elif rnn_type == 'GRU':
            self.rnn_module = nn.GRU(input_size = latent_dim, hidden_size = hidden_dim, num_layers = rnn_layers)
        else:
            raise ValueError("Invalid RNN type. Must be 'LSTM' or 'GRU'")
        
        if self.outcome_name == 'TomorrowTILBasic': 
            self.hidden2tiltomorrow = nn.Linear(hidden_dim,5)
        elif self.outcome_name == 'TomorrowHighIntensityTherapy': 
            self.hidden2tiltomorrow = nn.Linear(hidden_dim,1)
        else:
            raise ValueError("Invalid outcome label type. Must be 'TomorrowTILBasic' or 'TomorrowHighIntensityTherapy'")
        
        ## Initialize learned parameters
        # First mask out all chosen embedding indices
        embedX_weight = torch.zeros(self.n_tokens, self.latent_dim)
        nn.init.xavier_uniform_(embedX_weight)
        
        embedW_weight = torch.zeros(self.n_tokens, 1)
        nn.init.xavier_uniform_(embedW_weight)
                
        if len(self.mask_indices) > 0:
            embedX_weight[self.mask_indices,:] = 0.0
            embedW_weight[self.mask_indices,:] = 0.0
            
        self.embedX.weight = nn.Parameter(embedX_weight)
        self.embedW.weight = nn.Parameter(embedW_weight)
        
        for name, param in self.rnn_module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.hidden2tiltomorrow.weight)
        nn.init.constant_(self.hidden2tiltomorrow.bias, 0.0)
        
    def forward(self,idx_list, bin_offsets, gupi_offsets):
        
        embeddedX = self.embedX(idx_list)
        
        # Constrain weights to be positive with exponentiation
        w = torch.exp(self.embedW(idx_list))
        
        # Iterate through infdividual bins and calculate weighted averages per bin
        embed_output = []
        for curr_bin_idx in torch.arange(0, len(bin_offsets), dtype=torch.long):
            if curr_bin_idx == (torch.LongTensor([len(bin_offsets) - 1])[0]):
                curr_bin_seq = torch.arange(bin_offsets[curr_bin_idx], embeddedX.shape[0], dtype=torch.long)
            else:
                curr_bin_seq = torch.arange(bin_offsets[curr_bin_idx], bin_offsets[curr_bin_idx+1], dtype=torch.long)
            embeddedX_avg = (embeddedX[curr_bin_seq,:] * w[curr_bin_seq]).sum(dim=0, keepdim=True) / (len(curr_bin_seq) + 1e-6)
            embed_output += [embeddedX_avg]
        embed_output = torch.cat(embed_output, dim=0)
        embed_output = self.embed_Dropout(F.relu(embed_output))
        
        # Iterate through unique patients to run sequences through LSTM and model networks
        rnn_linear_outputs = []
        for curr_gupi_idx in torch.arange(0, len(gupi_offsets), dtype=torch.long):
            if curr_gupi_idx == (torch.LongTensor([len(gupi_offsets) - 1])[0]):
                curr_gupi_seq = torch.arange(gupi_offsets[curr_gupi_idx], embed_output.shape[0], dtype=torch.long)
            else:
                curr_gupi_seq = torch.arange(gupi_offsets[curr_gupi_idx], gupi_offsets[curr_gupi_idx+1], dtype=torch.long)    
            curr_rnn_out, _ = self.rnn_module(embed_output[curr_gupi_seq,:].unsqueeze(1))
            curr_til_out = self.hidden2tiltomorrow(curr_rnn_out).squeeze(1)
            rnn_linear_outputs += [curr_til_out]
        rnn_linear_outputs = torch.cat(rnn_linear_outputs, dim=0)
        return rnn_linear_outputs, gupi_offsets
    
    def training_step(self, batch, batch_idx):
        
        # Get information from current batch
        curr_label_list, curr_idx_list, curr_bin_offsets, curr_gupi_offsets, curr_gupis = batch
        
        # Collect current model state outputs for the batch
        (yhat, out_gupi_offsets) = self(curr_idx_list, curr_bin_offsets, curr_gupi_offsets)
        
        # Calculate loss based on the outcome label type
        if self.outcome_name == 'TomorrowTILBasic': 
            
            if self.class_weights:
                bal_weights = torch.from_numpy(compute_class_weight(class_weight='balanced',
                                                                    classes=np.sort(self.targets.explode().dropna().unique()).astype(float),
                                                                    y=self.targets.explode().dropna().values.astype(float))).type_as(yhat)
                missing_outcome_mask = torch.isnan(curr_label_list)
                loss = F.cross_entropy(yhat[~missing_outcome_mask,:], curr_label_list[~missing_outcome_mask].type(torch.long), weight = bal_weights)
            else:
                missing_outcome_mask = torch.isnan(curr_label_list)
                loss = F.cross_entropy(yhat[~missing_outcome_mask,:], curr_label_list[~missing_outcome_mask].type(torch.long))
                
        elif self.outcome_name == 'TomorrowHighIntensityTherapy': 
            
            if self.class_weights:
                bal_weights = torch.tensor((self.targets.explode().dropna().shape[0]-self.targets.explode().dropna().sum())/self.targets.explode().dropna().sum()).type_as(yhat)
                missing_outcome_mask = torch.isnan(curr_label_list)
                loss = F.binary_cross_entropy_with_logits(yhat[~missing_outcome_mask,:].flatten(), curr_label_list[~missing_outcome_mask].type_as(yhat), pos_weight = bal_weights)
            else:
                missing_outcome_mask = torch.isnan(curr_label_list)
                loss = F.binary_cross_entropy_with_logits(yhat[~missing_outcome_mask,:].flatten(), curr_label_list[~missing_outcome_mask].type_as(yhat))
                
        else:
            raise ValueError("Invalid outcome label type. Must be 'TomorrowTILBasic' or 'TomorrowHighIntensityTherapy'")
        
        return {"loss": loss, "yhat": yhat, "curr_label_list": curr_label_list}
    
    def training_epoch_end(self, training_step_outputs):
        
        comp_loss = torch.tensor([x["loss"].detach() for x in training_step_outputs]).cpu().numpy().mean()
        comp_yhats = torch.vstack([x["yhat"].detach() for x in training_step_outputs])
        comp_label_list = torch.cat([x["curr_label_list"].detach() for x in training_step_outputs])
        curr_train_labels = torch.cat([comp_label_list],dim=0).cpu().numpy()
        
        if self.outcome_name == 'TomorrowTILBasic': 
            curr_train_probs = torch.cat([F.softmax(comp_yhats)],dim=0).cpu().numpy()
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(np.unique(curr_train_labels[~np.isnan(curr_train_labels)])), 2)):
                a_mask = curr_train_labels == a
                b_mask = curr_train_labels == b
                ab_mask = np.logical_or(a_mask,b_mask)
                condit_probs = curr_train_probs[ab_mask,int(b)]/(curr_train_probs[ab_mask,int(a)]+curr_train_probs[ab_mask,int(b)]) 
                condit_probs = np.nan_to_num(condit_probs,nan=.5,posinf=1,neginf=0)
                condit_labels = b_mask[ab_mask].astype(int)
                aucs.append(roc_auc_score(condit_labels,condit_probs))            
            train_metric = np.mean(aucs)
        elif self.outcome_name == 'TomorrowHighIntensityTherapy': 
            curr_train_probs = torch.cat([F.sigmoid(comp_yhats)],dim=0).flatten().cpu().numpy()    
            missing_outcome_mask = np.isnan(curr_train_labels)   
            train_metric = roc_auc_score(curr_train_labels[~missing_outcome_mask],curr_train_probs[~missing_outcome_mask])
                    
        self.log('train_metric', train_metric, prog_bar=True, logger=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('train_loss', comp_loss, prog_bar=False, logger=True, sync_dist=True, on_step=False, on_epoch=True)
    
    def validation_step(self, batch, batch_idx):
        
        # Get information from current batch
        curr_label_list, curr_idx_list, curr_bin_offsets, curr_gupi_offsets, curr_gupis = batch
        
        # Collect current model state outputs for the batch
        (yhat, out_gupi_offsets) = self(curr_idx_list, curr_bin_offsets, curr_gupi_offsets)
        
        curr_val_labels = torch.cat([curr_label_list],dim=0).cpu().numpy()

        # Calculate loss based on the outcome label type
        if self.outcome_name == 'TomorrowTILBasic': 
            
            curr_val_probs = torch.cat([F.softmax(yhat)],dim=0).cpu().numpy()
            missing_outcome_mask = torch.isnan(curr_label_list)
            val_loss = F.cross_entropy(yhat[~missing_outcome_mask,:], curr_label_list[~missing_outcome_mask].type(torch.long))
            
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(np.unique(curr_val_labels[~np.isnan(curr_val_labels)])), 2)):
                a_mask = curr_val_labels == a
                b_mask = curr_val_labels == b
                ab_mask = np.logical_or(a_mask,b_mask)
                condit_probs = curr_val_probs[ab_mask,int(b)]/(curr_val_probs[ab_mask,int(a)]+curr_val_probs[ab_mask,int(b)]) 
                condit_probs = np.nan_to_num(condit_probs,nan=.5,posinf=1,neginf=0)
                condit_labels = b_mask[ab_mask].astype(int)
                aucs.append(roc_auc_score(condit_labels,condit_probs))            
            val_metric = np.mean(aucs)
                            
        elif self.outcome_name == 'TomorrowHighIntensityTherapy': 
            
            curr_val_probs = torch.cat([F.sigmoid(yhat)],dim=0).flatten().cpu().numpy()
            missing_outcome_mask = torch.isnan(curr_label_list)
            val_loss = F.binary_cross_entropy_with_logits(yhat[~missing_outcome_mask,:].flatten(), curr_label_list[~missing_outcome_mask].type_as(yhat))
            val_metric = roc_auc_score(curr_val_labels[~missing_outcome_mask],curr_val_probs[~missing_outcome_mask])

        else:
            raise ValueError("Invalid outcome label type. Must be 'TomorrowTILBasic' or 'TomorrowHighIntensityTherapy'")
        
        self.log('val_metric', val_metric, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss', val_loss, prog_bar=False, logger=True, sync_dist=True)

        return val_loss
        
    def configure_optimizers(self):
        
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: 
                continue # frozen weights
            if (len(param.shape) == 1) or (".bias" in name): 
                no_decay.append(param)
            else:
                decay.append(param)
        params = [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': (10**(-3.5))}]
        
        optimizer = optim.Adam(params,lr=self.learning_rate)
        return optimizer

# dynamic TILTomorrow model modification for TimeSHAP calculation
class timeshap_TILTomorrow_model(nn.Module):
    def __init__(self,tiltomorrow_model,rnn_type,threshold_idx,unknown_index,cols_to_add):
        """
        Args:
            tiltomorrow_model (LightningModule): trained dynTTM model from which to extract layers
            rnn_type (string, 'LSTM' or 'GRU'): Identify RNN architecture type
            threshold_idx (int): index of the GOSE threshold to focus on for TimeSHAP. If -1, then represents expected GOSE outcome as TimeSHAP target
            unknown_index (int): Embedding layer index corresponding to '<unk>' token
            cols_to_add (int): Number of rows to add to embedding layer to account for unknown indices
        """
        super(timeshap_TILTomorrow_model, self).__init__()

        # Extract trained initial embedding layer and modify for TimeSHAP calculation 
        self.embedX = copy.deepcopy(tiltomorrow_model).embedX
        self.embedX.weight = nn.Parameter(torch.cat((self.embedX.weight,torch.tile(self.embedX.weight[unknown_index,:],(cols_to_add,1))),dim=0),requires_grad=False)

        # Extract trained weighting embedding layer and modify for TimeSHAP calculation 
        self.embedW = copy.deepcopy(tiltomorrow_model).embedW
        self.embedW.weight = nn.Parameter(torch.cat((self.embedW.weight,torch.tile(self.embedW.weight[unknown_index,:],(cols_to_add,1))),dim=0),requires_grad=False)
        
        # Combine 2 embedding layers into single transformation matrix
        self.comb_embedding = self.embedX.weight*torch.tile(torch.exp(self.embedW.weight),(1,self.embedX.weight.shape[1]))
        
        # Extract trained RNN module and modify for TimeSHAP calculation
        self.rnn_module = copy.deepcopy(tiltomorrow_model).rnn_module
        self.rnn_module.batch_first=True
        
        # Extract trained output layer and modify for TimeSHAP calculation
        self.hidden2tiltomorrow = copy.deepcopy(tiltomorrow_model).hidden2tiltomorrow
        
        # Save inputted RNN type and ensure it is one of 2 accepted options
        self.rnn_type = rnn_type
        if self.rnn_type not in ['LSTM','GRU']:
            raise ValueError("Invalid RNN type. Must be 'LSTM' or 'GRU'")
        
        # Save threshold idx of focus
        self.threshold_idx = threshold_idx
        
    # Define forward run function
    def forward(self,x: torch.Tensor, hidden_states:tuple = None):
        
        # Calculate number of tokens per row and fix zero-token rows to one
        row_sums = x.sum(-1)
        row_sums[row_sums == 0] = 1.0
        row_sums = torch.tile(row_sums.unsqueeze(-1),(1,1,self.comb_embedding.shape[-1]))
                   
        # Embed input and divide by row sums
        curr_embedding_out = F.relu(torch.matmul(x,self.comb_embedding) / row_sums)
        
        # Obtain RNN output and transform to GOSE space
        if hidden_states is None:
            curr_rnn_out, curr_rnn_hidden = self.rnn_module(curr_embedding_out)
        else:
            curr_rnn_out, curr_rnn_hidden = self.rnn_module(curr_embedding_out, hidden_states)
            
        # -1 on hidden, to select the last layer of the stacked gru/lstm
        if self.rnn_type == 'LSTM':
            assert torch.equal(curr_rnn_out[:,-1,:], curr_rnn_hidden[0][-1, :, :])
        elif self.rnn_type == 'GRU':
            assert torch.equal(curr_rnn_out[:,-1,:], curr_rnn_hidden[-1, :, :])
        else:
            raise ValueError("Invalid RNN type. Must be 'LSTM' or 'GRU'")
        
        # Calculate output values for TimeSHAP based on target and RNN type
        if self.rnn_type == 'LSTM':
            curr_til_out = F.softmax(self.hidden2tiltomorrow(curr_rnn_hidden[0][-1, :, :]))
        elif self.rnn_type == 'GRU':
            curr_til_out = F.softmax(self.hidden2tiltomorrow(curr_rnn_hidden[-1, :, :]))
        else:
            raise ValueError("Invalid RNN type. Must be 'LSTM' or 'GRU'")

        if self.threshold_idx == -1:
            curr_til_out = torch.matmul(curr_til_out,torch.arange(curr_til_out.shape[1],dtype=torch.float).unsqueeze(1))
        else:
            curr_til_out = (1-curr_til_out.cumsum(-1))[:,self.threshold_idx]
        
        # Return output value of focus and RNN hidden state
        return curr_til_out, curr_rnn_hidden