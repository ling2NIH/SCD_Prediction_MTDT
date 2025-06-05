import dash
from dash import html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import io
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import math
import shap_v2
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.io import DataLoader, TensorDataset, Subset

import numpy as np
import pandas as pd
import os
from paddle.static import InputSpec
import random
from paddle.jit import to_static
from scipy.interpolate import interp1d
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import random
from paddle.io import DataLoader
from visualdl import LogWriter

from sklearn.model_selection import KFold, train_test_split

def log(x):
    return paddle.log(x + 1e-08)

def div(x, y):
    return x / (y + 1e-08)

def get_activation_fn(name):
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "softmax": nn.Softmax(axis=-1),
    }
    return activations.get(name, None)  

def overall_cause_specific_c_index(pred, event, time, num_causes_idx):
    out = pred[:,num_causes_idx,:].numpy()
    
    time_points = np.arange(1, out.shape[1] + 1, dtype=np.float32)
    expected_time = np.sum(out * time_points, axis=1)
    risk_score = -expected_time
    event_indicator_bool = event.numpy().astype(bool)
    try:
        c_index = concordance_index_censored(
            event_indicator_bool.squeeze(), 
            time.numpy().squeeze(), 
            risk_score.squeeze()
        )[0]  
    except NoComparablePairException:
        c_index = -1  
    
    return c_index

def cause_specific_auc(pred, event, event_time, cause_idx, time_grid):
    out = pred[:, cause_idx, :].numpy()
    
    event_notensor = event.numpy()
    event_time_notensor = event_time.numpy()
    
    auc_scores = []
    
    for t in time_grid:
        # Define event indicator
        event_indicator_new = ((event_time_notensor <= t) & (event_notensor == 1)).astype(int)
        
        # Compute risk score
        risk_score = np.sum(out[:, :t], axis=1)  # Summing probabilities up to time t
        
        # Compute AUC if valid
        if len(np.unique(event_indicator_new)) > 1:
            auc_value = roc_auc_score(event_indicator_new, risk_score)
        else:
            auc_value = np.nan  # Store NaN if AUC cannot be computed
        
        auc_scores.append(auc_value)

    # Convert to numpy array
    auc_scores = np.array(auc_scores)

    # Compute iAUC only if valid values exist
    
    iauc = auc(time_grid, auc_scores) / (time_grid[-1] - time_grid[0])

    return auc_scores, iauc

def cause_specific_intergrated_brier_score(predictions, time_survival, event_type,  num_causes_idx):
    prediction = predictions[:,num_causes_idx,:].numpy()

    event_type = event_type.numpy()
    
    time_survival = time_survival.numpy()
    
    time_grid = np.arange(0, 6)
    
    brier_scores = []
    for time in time_grid:
        # when time == 0, we need it be 0
        if time ==0:
            pred_e = np.zeros(prediction.shape[0])
        else:
            pred_e = np.sum(prediction[:, :time], axis=1)  # Sum P(T = ti) for ti ≤ t
        
        y_true = ((time_survival <= time) & (event_type == 1)).astype(float)
        brier_scores.append(np.mean(np.array((pred_e - y_true) ** 2)))
    
    ibs = np.trapz(brier_scores, time_grid) / (time_grid[-1] - time_grid[0])
    return brier_scores, ibs



### CENSORING PROBABILITY USING KAPLAN-MEIER ESTIMATE
def censoring_prob(y, t):
    """
    Compute censoring probabilities using Kaplan-Meier estimate.
    
    Parameters:
        y (np.ndarray): Censoring indicator (1 = event, 0 = censored).
        t (np.ndarray): Observed times.

    Returns:
        np.ndarray: Censoring probabilities at observed times.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(t, event_observed=(y == 0).astype(int))  # Censoring probability as survival probability of censoring event
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  # Fill 0 with the last observed value
    return G

### C(t)-INDEX FOR ALL CAUSES
def cause_specific_c_index_all(predictions, time_survival, event_type, time, num_causes):
    """
    Computes cause-specific C-index for each cause in competing risks.

    Parameters:
        predictions (np.ndarray): Predicted risk scores for each cause, shape (num_samples, num_causes).
        time_survival (np.ndarray): Survival or censoring times, shape (num_samples,).
        event_type (np.ndarray): Event types (1, 2, ..., num_causes; 0 for censored), shape (num_samples,).
        time (int): Evaluation time horizon for the C-index.
        num_causes (int): Number of competing events.

    Returns:
        dict: Cause-specific C-index for each event.
    """
    c_index_scores = {}
    for cause in range(1, num_causes + 1):
        prediction = np.sum(predictions[:, cause - 1,:time], axis=1)
        N = len(prediction)
        A, Q, N_t = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))

        for i in range(N):
            A[i, np.where(time_survival[i] < time_survival)[0]] = 1
            Q[i, np.where(prediction[i] > prediction)[0]] = 1
            if time_survival[i] <= time and event_type[i] == cause:
                N_t[i, :] = 1
            #print(f"A: {A}")  # Debug print
            #print(f"Q: {Q}")  # Debug print

        Num = np.sum(A * N_t * Q)
        Den = np.sum(A * N_t)
        c_index_scores[f"Cause_{cause} at horizon_{time}"] = float(Num / Den) if Den != 0 else -1
    return c_index_scores

### BRIER SCORE FOR ALL CAUSES
def cause_specific_brier_score_all(predictions, time_survival, event_type, time, num_causes):
    """
    Computes cause-specific Brier score for each cause in competing risks.

    Parameters:
        predictions (np.ndarray): Predicted risk scores for each cause, shape (num_samples, num_causes, num_time_steps).
        time_survival (np.ndarray): Survival or censoring times, shape (num_samples,).
        event_type (np.ndarray): Event types (1, 2, ..., num_causes; 0 for censored), shape (num_samples,).
        time (int): Evaluation time horizon for the Brier score.
        num_causes (int): Number of competing events.

    Returns:
        dict: Cause-specific Brier score for each event.
    """
    brier_scores = {}
    for cause in range(1, num_causes + 1):
        pred_e = np.sum(predictions[:, cause - 1, :time], axis=1)
        y_true = ((time_survival <= time) & (event_type == cause)).astype(float)
        #print(f"y_true: {y_true}")  # Debug print
        brier_scores[f"Cause_{cause} at horizon_{time}"] = np.mean(np.array((pred_e - y_true) ** 2))
    return brier_scores


def weighted_cause_specific_c_index_all(t_train, y_train, predictions, t_test, y_test, time, num_causes):
    """
    Computes weighted cause-specific C-index for each cause in competing risks.
    """
    weighted_c_index_scores = {}
    G = censoring_prob(y_train, t_train)  # censoring probabilities from training data
    N = len(t_test)

    for cause in range(1, num_causes + 1):
        prediction = np.sum(predictions[:, cause - 1, :time], axis =1) # predictions for this cause at the time horizon
        A, Q, N_t = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))

        for i in range(N):
            tmp_idx = np.where(G[0, :] >= t_test[i])[0]
            W = (1.0 / G[1, -1]) ** 2 if len(tmp_idx) == 0 else (1.0 / G[1, tmp_idx[0]]) ** 2

            # Assign weights element-wise using multiplication for broadcasting
            A[i] = (t_test[i] < t_test) * W  # Boolean mask * W to assign correct weights
            Q[i] = (prediction[i] > prediction).astype(float)  # Boolean mask for ranking

            # Check if event occurred for this cause within the time horizon
            if t_test[i] <= time and y_test[i] == cause:
                N_t[i, :] = 1.0

        Num = np.sum(A * N_t * Q)
        Den = np.sum(A * N_t)
        weighted_c_index_scores[f"Cause_{cause} at horizon_{time}"] = float(Num / Den) if Den != 0 else -1

    return weighted_c_index_scores



### WEIGHTED BRIER SCORE FOR ALL CAUSES
def weighted_cause_specific_brier_score_all(t_train, y_train, predictions, t_test, y_test, time, num_causes):
    """
    Computes weighted cause-specific Brier score for each cause in competing risks.

    Parameters:
        t_train, y_train : Training set times and censoring indicators.
        predictions      : Risk scores for test set for each cause, shape (num_samples, num_causes).
        t_test, y_test   : Test set times and censoring indicators.
        time             : Evaluation time horizon for Brier score.
        num_causes       : Number of competing events.

    Returns:
        dict: Weighted cause-specific Brier score for each event.
    """
    weighted_brier_scores = {}
    G = censoring_prob(y_train, t_train)
    N = len(t_test)

    for cause in range(1, num_causes + 1):
        pred_e = np.sum(predictions[:, cause - 1,:time], axis=1)
        W = np.zeros(N)
        Y_tilde = (t_test > time).astype(float)

        for i in range(N):
            tmp_idx1 = np.where(G[0, :] >= t_test[i])[0]
            tmp_idx2 = np.where(G[0, :] >= time)[0]

            G1 = G[1, -1] if len(tmp_idx1) == 0 else G[1, tmp_idx1[0]]
            G2 = G[1, -1] if len(tmp_idx2) == 0 else G[1, tmp_idx2[0]]
            W[i] = (1.0 - Y_tilde[i]) * float(y_test[i] == cause) / G1 + Y_tilde[i] / G2

        y_true = ((t_test <= time) & (y_test == cause)).astype(float)
        weighted_brier_scores[f"Cause_{cause} at horizon_{time}"] = np.mean(np.array(W * (Y_tilde - (1.0 - pred_e)) ** 2))
    return weighted_brier_scores

def f_get_Normalization(X, norm_mode):
    num_Patient, num_Feature = np.shape(X)

    if norm_mode == 'standard': #zero mean unit variance
        for j in range(num_Feature):
            if np.std(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))/np.std(X[:,j])
            else:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))
    elif norm_mode == 'normal': #min-max normalization
        for j in range(num_Feature):
            X[:,j] = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j]) - np.min(X[:,j]))
    else:
        print("INPUT MODE ERROR!")

    return X

### MASK FUNCTIONS
'''
    fc_mask2      : To calculate LOSS_1 (log-likelihood loss)
    fc_mask3      : To calculate LOSS_2 (ranking loss)
'''
def f_get_fc_mask2(time, label, num_Event, num_Category):
    '''
        mask4 is required to get the log-likelihood loss
        mask4 size is [N, num_Event, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i,0] != 0:  #not censored
            time_idx = min(int(time[i,0]-1), num_Category - 1)
            mask[i,int(label[i,0]-1),time_idx] = 1
        else: #label[i,2]==0: censored
            time_idx = min(int(time[i,0]), num_Category)
            mask[i,:,time_idx:] =  1 #fill 1 until from the censoring time (to get 1 - \sum F)
    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category].
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    if np.shape(meas_time):  #lonogitudinal measurements
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0]) # last measurement time
            t2 = int(time[i, 0]) # censoring/event time
            mask[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
    else:                    #single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0]) # censoring/event time
            mask[i,:t] = 1  #this excludes the last measurement time and includes the event time
    return mask

def import_dataset_SDC_sim_outcome(norm_mode='standard'):
    df_with_miss = pd.read_pickle('/scratch/ling2/FSL-Mate/PaddleFSL/examples/molecular_property_prediction/SCD/df_baseline_outcomes_io_na_nomiss30.pkl')
    df_baseline_with_miss = df_with_miss.drop(columns=['alanine aminotransferase','alkaline phosphatase','bilirubin, direct','creatinine','urea nitrogen','IVS_with_t','TPV_with_t','RAA_with_t','HR_with_t','BMI_with_t','ProBNP_with_t','eGFR_AA_with_t','eGFR_NAA_with_t'])
    df_long_for_pred = df_with_miss[['alanine aminotransferase','alkaline phosphatase','bilirubin, direct','creatinine','urea nitrogen','IVS_with_t','TPV_with_t','RAA_with_t','HR_with_t','BMI_with_t','ProBNP_with_t','eGFR_AA_with_t','eGFR_NAA_with_t']]
    df_baseline_imputed = pd.read_pickle('/scratch/ling2/FSL-Mate/PaddleFSL/examples/molecular_property_prediction/SCD/df_imputed_baseline_v2.pkl')
    df_numerical = df_baseline_imputed.drop(columns=['MRN','Echo.date.and.time','Event_time', 'Event_status','Male','Genotype','Ethnic_group'])
    df_numerical_scaled = f_get_Normalization(np.array(df_numerical), norm_mode)
    df_categorical = df_baseline_imputed[['Male','Genotype','Ethnic_group']]
    data = np.concatenate((df_numerical_scaled, df_categorical), axis=1)

    label = np.asarray(df_baseline_imputed[['Event_status']])
    time = np.asarray(df_baseline_imputed[['Event_time']])/365.25
    time = np.ceil(time)
    # time >=15, set to 16
    time[time >= 15] = 16

    num_Category = int(np.max(time))
    num_Event       = int(len(np.unique(label)) - 1) 

    mask1           = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask2           = f_get_fc_mask3(time, -1, num_Category)
    
    df_numerical_miss = df_baseline_with_miss.drop(columns=['MRN','Echo.date.and.time','Event_time', 'Event_status','Male','Genotype','Ethnic_group'])

    df_categorical_miss = df_baseline_with_miss[['Male','Genotype','Ethnic_group']]
    data_miss = np.concatenate((df_numerical_miss, df_categorical_miss), axis=1)

    mask_miss = ~np.isnan(data_miss)
    outcome_matrix_pred = np.array(df_long_for_pred.to_numpy().tolist()).astype(np.float64) 
    mask_miss_pred = ~np.isnan(outcome_matrix_pred)
    

    outcome_matrix_scaled = outcome_matrix_pred.copy()

    for i in range(outcome_matrix_pred.shape[1]):  
        x = outcome_matrix_pred[:, i, :, 0] 
        mean = np.nanmean(x)  
        std = np.nanstd(x)

        outcome_matrix_scaled[:, i, :, 0] = (x - mean) / (std + 1e-8) 

    from scipy.interpolate import BSpline

    outcome_matrix_scaled = np.nan_to_num(outcome_matrix_scaled, nan=0.0)
    x = outcome_matrix_scaled[:,:,:,1]
    batch_size, covariate_dim, t_dim = x.shape
    k = 2  
    num_internal_knots = 2


    internal_knots = np.linspace(0, 3, num_internal_knots + 2)[1:-1] 
    t_knots = np.concatenate((
        np.repeat(0, k+1),
        internal_knots,
        np.repeat(3, k+1)
    ))

    num_basis = len(t_knots) - (k + 1)  # number of basis functions
    c = np.eye(num_basis)  # identity matrix for basis

    b_spline_basis = [BSpline(t_knots, c[i], k) for i in range(num_basis)]

    batch_basis = []

    for b in range(batch_size):
        x_temp = x[b]  # (Covariate_dim, x_dim)

        basis_values_covariates = []
        for j in range(covariate_dim):
            t_input = x_temp[j]  # (x_dim,)

            # apply each basis
            basis_eval = np.stack([basis(t_input) for basis in b_spline_basis], axis=0)  # (num_basis, x_dim)
            basis_values_covariates.append(basis_eval)

        basis_values_covariates = np.stack(basis_values_covariates, axis=0)  # (Covariate_dim, num_basis, x_dim)
        batch_basis.append(basis_values_covariates)


    batch_basis = np.stack(batch_basis, axis=0)  # (batch_size, Covariate_dim, num_basis, x_dim)    
    return data.astype('float32'), label.astype('float32'), time.astype('float32'), mask1.astype('float32'), mask2.astype('float32'), num_Category, num_Event, mask_miss.astype('float32'),outcome_matrix_scaled.astype('float32'), mask_miss_pred.astype('float32'), batch_basis.astype('float32')

from visualdl import LogWriter

class ModelDeepHit_Multitask(nn.Layer):
    def __init__(self, input_dims, network_settings, outcome_configs, autoencoder, log_writer):
        super(ModelDeepHit_Multitask, self).__init__()
        
        # Define input dimensions and network settings
        self.x_dim = input_dims['x_dim']
        self.num_Event = input_dims['num_Event']
        self.num_Category = input_dims['num_Category']
        self.h_dim_shared = network_settings['h_dim_shared']
        self.h_dim_CS = network_settings['h_dim_CS']
        self.num_layers_shared = network_settings['num_layers_shared']
        self.num_layers_CS = network_settings['num_layers_CS']
        self.active_fn = network_settings['active_fn']
        self.keep_prob = network_settings['keep_prob']
        

        self.initial_W = paddle.nn.initializer.XavierUniform()
        self.autoencoder = autoencoder
        self.add_sublayer('autoencoder', self.autoencoder)
        self.ae_out_dim = network_settings['ae_out_dim']
        self.log_writer = log_writer
        #self.ae_hidden_dim = network_settings['ae_hidden_dim']

        # Autoencoder
        #self.autoencoder = self._build_autoencoder()
        # pooling layer
        #self.pooling_layer = nn.AdaptiveAvgPool1D(1)
        #self.add_sublayer('pooling_layer', self.pooling_layer)
        # Shared Network
        self.linear_layer = nn.Linear(self.ae_out_dim , 1, weight_attr=self.initial_W)
        self.add_sublayer('linear_layer', self.linear_layer)
        self.shared_net = self._build_shared_network()
        self.add_sublayer('shared_net', self.shared_net) 

        # Cause-Specific Networks
        self.cs_nets = nn.LayerList([self._build_cs_network() for _ in range(self.num_Event)])
        self.add_sublayer('cs_nets', self.cs_nets)
        
        
        self.outcome_pred_nets = nn.LayerList([
            create_outcome_specific_net(
                input_dim=self.h_dim_shared+ self.x_dim,
                num_layers=self.num_layers_CS,           # Number of layers
                hidden_dim=self.h_dim_CS,           # Number of hidden units
                activation_fn=self.active_fn, # Activation function for hidden layers
                output_dim=config["output_dim"], # Dimension of the output layer
                output_activation=config["output_activation"],  # Activation function for the output layer
                keep_prob=self.keep_prob,
                use_resnet=True
            ) for config in outcome_configs
        ])
        self.add_sublayer('outcome_pred_nets', self.outcome_pred_nets)

        # Output layer
        self.output_layer = nn.Linear(self.num_Event * self.h_dim_CS, self.num_Event * self.num_Category)
        self.add_sublayer('output_layer', self.output_layer)
        #self.output_bn = nn.BatchNorm1D(self.num_Event * self.num_Category)
        self.softmax = nn.Softmax(axis=-1)

    def _build_shared_network(self):
        # Create shared network using create_fc_net
        return create_fc_net(self.x_dim, self.num_layers_shared, self.h_dim_shared, self.active_fn, 
                             self.h_dim_shared, self.active_fn, self.initial_W, keep_prob=self.keep_prob, use_resnet=True)

    def _build_cs_network(self):
        # Create cause-specific network using create_fc_net
        return create_fc_net(self.h_dim_shared + self.x_dim, self.num_layers_CS, self.h_dim_CS, self.active_fn, 
                             self.h_dim_CS, self.active_fn, self.initial_W, keep_prob=self.keep_prob, use_resnet=True)
    
    #def _build_autoencoder(self):
        # Create autoencoder network using create_fc_net
        #return create_fc_net(self.x_dim, 2, self.ae_hidden_dim, self.active_fn, 
                             #self.ae_out_dim, self.active_fn, self.initial_W, keep_prob=1)
    
    #@paddle.jit.to_static(input_spec=[paddle.static.InputSpec(shape=[None, input_dims['x_dim']], dtype='float32')])
    def forward(self, x):
        # Autoencoder

        ae_out, _= self.autoencoder(x, mask = self.mask)
        
        ae_out = self.linear_layer(ae_out)  # (batch_size, input_dim, 1)
        ae_out = ae_out.squeeze(-1)  # (batch_size, input_dim)

        #ae_out = self.pooling_layer(ae_out.transpose([0, 2, 1])).squeeze(-1)

        # Shared Network
        shared_out = self.shared_net(ae_out)
        #print(f"shared_out shape: {shared_out.shape}")  # Debug print
        
        h = paddle.concat([ae_out, shared_out], axis=1)
        #print(f"h shape after concat: {h.shape}")  # Debug print

        # Cause-Specific Networks
        out_list = [cs_net(h) for cs_net in self.cs_nets]
        out = paddle.concat(out_list, axis=1)
        #print(f"out shape after CS networks: {out.shape}")  # Debug print

        out = F.dropout(out, p=self.keep_prob, training=self.training)
        #print(f"out shape after dropout: {out.shape}")  # Debug print

        # Output Layer
        out = self.output_layer(out)
        #print(f"out shape before reshape: {out.shape}")  # Debug print
        #out = self.output_bn(out)

        out = self.softmax(paddle.reshape(out, [-1, self.num_Event, self.num_Category]))
        #print(f"out shape after reshape: {out.shape}")  # Debug print

        self.out = out


        self.outcome_preds = [net(h) for net in self.outcome_pred_nets]
        
       

        return self.out, self.outcome_preds

    def compute_loss(self, k, t, fc_mask1, fc_mask2, alpha, beta, gamma, delta, eta, 
                    outcomes_true, outcome_preds, outcome_configs, missing_mask, basis):
        """
        Compute the total loss, including survival analysis losses and multitask losses,
        incorporating missing masks for multitask outcomes.

        Parameters:
        - k, t, fc_mask1, fc_mask2: Inputs for survival-related loss functions.
        - alpha, beta, gamma: Weights for survival-related losses.
        - delta: Weight for multitask losses.
        - outcomes_true: Ground truth values for multitask outcomes, shape [batch_size, num_outcomes].
        - outcome_preds: Predictions for multitask outcomes as a list.
        - outcome_configs: Configuration for each multitask outcome.
        - missing_mask: Binary mask indicating missing outcomes, shape [batch_size, num_outcomes].

        Returns:
        - Total loss (combined survival and multitask losses).
        """
        # Survival analysis losses
        loss1 = self.loss_log_likelihood(k, fc_mask1)
        loss2 = self.loss_ranking(k, t, fc_mask2)
        loss3 = self.loss_calibration(k, fc_mask2)
        survival_loss = alpha * loss1 + beta * loss2 + gamma * loss3

        # Multitask losses
        outcome_losses = []
        loss_weights = []
        
        for i, config in enumerate(outcome_configs):
            task_type = config["task_type"]



            # Calculate loss based on the task type
            if task_type == "regression":
                # Regression loss (MSE)
                mask_indices = paddle.nonzero(missing_mask[:, 0, i]).squeeze()
                true_values = outcomes_true[:,0, i].index_select(mask_indices)
                pred_values = outcome_preds[i].index_select(mask_indices)
                loss = paddle.mean((true_values - pred_values.squeeze())**2)

            elif task_type == "binary_classification":
                # Binary classification loss (BCE)
                mask_indices = paddle.nonzero(missing_mask[:, 0, i]).squeeze()
                true_values = outcomes_true[:,0, i].index_select(mask_indices)
                pred_values = outcome_preds[i].index_select(mask_indices)
                loss = F.binary_cross_entropy(pred_values.squeeze(), true_values)

            elif task_type == "multiclass_classification":
                # Multiclass classification loss (Cross Entropy)
                mask_indices = paddle.nonzero(missing_mask[:, 0, i]).squeeze()
                true_values = outcomes_true[:,0, i].index_select(mask_indices)
                pred_values = outcome_preds[i].index_select(mask_indices)
                loss = F.cross_entropy(pred_values, true_values.astype('int64'))
            
            elif task_type == "longitudinal_regression":
                # Longitudinal regression loss (MSE)
                #outcomes_true = paddle.where(paddle.isnan(outcomes_true), paddle.zeros_like(outcomes_true), outcomes_true)
                loss = self.mse_longitudinal(outcomes_true[:,i,:,0], outcome_preds[i], missing_mask[:,i,:,0], basis[:,i,:,:])
                #mask_indices = paddle.nonzero(missing_mask[:, :, i]).squeeze()
                #true_values = outcomes_true[:,:, i].index_select(mask_indices)
                #pred_values = outcome_preds[i].index_select(mask_indices)
                #loss = paddle.mean((outcomes_true[:,:,i] - outcome_preds[i])**2 * missing_mask[:,:,i])

            else:
                # Raise an error for unsupported task types
                raise ValueError(f"Unsupported task type: {task_type}")

            outcome_losses.append(loss)

            # Compute weight for the current task (inverse of the loss or constant weight)
            loss_weights.append(1.0)

        # Convert weights list to a Paddle tensor
        loss_weights = paddle.to_tensor(loss_weights, dtype='float32')

        # Compute the combined multitask loss as the weighted sum of individual losses
        multitask_loss = paddle.sum(
            paddle.stack([w * l for w, l in zip(loss_weights, outcome_losses)])
        )

        # Total loss
        regularization_loss = self.get_regularization_loss()
        #print(regularization_loss)
        #print(f"survival_loss: {survival_loss}, multitask_loss: {multitask_loss}, regularization_loss: {regularization_loss}")

        loss_total = survival_loss + delta * multitask_loss + eta* regularization_loss 

        return loss_total
    
    #def get_regularization_loss(self):
        #regularization_loss = 0.0
        #for param in self.parameters():
            # L2 regularization
            #regularization_loss += paddle.norm(param, p=2) ** 2
        #return regularization_loss
    
    def get_regularization_loss(self):
        regularization_loss = 0.0
        for param in self.parameters():
            # L2 regularization
            regularization_loss += paddle.norm(param, p=2) ** 2
        return regularization_loss

    def loss_log_likelihood(self, k, fc_mask1):
        I_1 = paddle.sign(k)

        # For uncensored: log P(T=t,K=k|x)
        tmp1 = paddle.sum(paddle.sum(fc_mask1 * self.out, axis=2), axis=1, keepdim=True)
        tmp1 = I_1 * log(tmp1)

        # For censored: log \sum P(T>t|x)
        tmp2 = paddle.sum(paddle.sum(fc_mask1 * self.out, axis=2), axis=1, keepdim=True)
        tmp2 = (1. - I_1) * log(tmp2)

        loss_1 = -paddle.mean(tmp1 + 1.0 * tmp2)
        weight = 1.0 / (loss_1.detach().item() + 1e-6)
        return loss_1 

    def loss_ranking(self, k, t, fc_mask2):
        sigma1 = 0.1
        eta = []
        for e in range(self.num_Event):
            one_vector = paddle.ones_like(t)
            I_2 = (k == (e + 1)).astype('float32')  # Indicator for event
            I_2 = paddle.diag(I_2.squeeze())
            tmp_e = self.out[:, e, :]  # Shape should be [batch_size, num_Category]

            # Matrix multiplication
            R = paddle.matmul(tmp_e, fc_mask2.T)  # Should result in [32, 32]

            # Extract diagonal and reshape for matrix operations
            diag_R = paddle.diagonal(R)
            diag_R = paddle.reshape(diag_R, [-1, 1])  # Reshape to column vector [32, 1]

            # Use broadcasting instead of paddle.matmul
            R = diag_R - R
            R = R.T

            # Compute T
            T = F.relu(paddle.sign(paddle.matmul(one_vector, t.T) - paddle.matmul(t, one_vector.T)))
            T = paddle.matmul(I_2, T)

            # Calculate eta
            tmp_eta = paddle.mean(T * paddle.exp(-R / sigma1), axis=1, keepdim=True)
            eta.append(tmp_eta)

        eta = paddle.stack(eta, axis=1)
        eta = paddle.mean(paddle.reshape(eta, [-1, self.num_Event]), axis=1, keepdim=True)

        loss_2 = paddle.sum(eta)
        weight = 1.0 / (loss_2.detach().item() + 1e-6)
        return loss_2


    def loss_calibration(self, k, fc_mask2):
        eta = []
        for e in range(self.num_Event):
            I_2 = (k == (e + 1)).astype('float32')
            tmp_e = self.out[:, e, :]
            

            r = paddle.sum(tmp_e * fc_mask2, axis=1)
            
            tmp_eta = paddle.mean((r - I_2)**2, axis=0, keepdim=True)

            eta.append(tmp_eta)
       
        eta = paddle.stack(eta, axis=1)
        
        eta = paddle.mean(paddle.reshape(eta, [-1, self.num_Event]), axis=1, keepdim=True)
        
        loss_3 = paddle.sum(eta)
        weight = 1.0 / (loss_3.detach().item() + 1e-6)
        return loss_3    

    def mse_longitudinal(self, outcome_true, outcome_pred, missing_mask_fp, basis):

        outcome_true = paddle.where(paddle.isnan(outcome_true), paddle.zeros_like(outcome_true), outcome_true)


        outcome_pred = outcome_pred.unsqueeze(1)  # (batch_size, 1, x_dim)
        curve_prediction = paddle.matmul(outcome_pred, basis) # (batch_size,1, x_dim)
        curve_prediction = paddle.squeeze(curve_prediction, axis=1)  # (batch_size, x_dim)
        loss_elementwise = (outcome_true-curve_prediction)**2
        masked_loss = (loss_elementwise * missing_mask_fp).sum() / missing_mask_fp.sum()
        
        return masked_loss




    def train_model(self, train_loader, val_loader,optimizer, alpha, beta, gamma, delta, eta,outcome_configs, epochs=1000, patience=100, min_delta=1e-4, weights_on_metric=None):
        """
        Train the model with multitask support, and log all required metrics.
        """

        self.best_weighted_metric = float('-inf')
        self.best_survival_metric = float('inf')
        self.best_multitask_metrics = float('-inf')
        patience_counter = 0
        self.best_model_state = None
        best_model_path = "./saved_model/best_model_training_multi_long_v2.11_SCD_2.pdparams"

        for epoch in range(epochs):
            # Training phase
            self.train()
            total_loss = 0
            total_survival_loss = 0
            total_multitask_loss = 0
            total_nll_loss = 0
            total_ranking_loss = 0
            total_calibration_loss = 0
            total_regularization_loss = 0
            
            for batch_idx, (x_mb, k_mb, t_mb, m1_mb, m2_mb, outcomes_true, missing_mask, missing_mask_fp, basis_mb) in enumerate(train_loader):
                optimizer.clear_grad()
                self.mask = missing_mask
                x_mb = paddle.where(paddle.isnan(x_mb), paddle.zeros_like(x_mb), x_mb)
                # Forward pass
                cause_out, outcome_preds = self.forward(x_mb)

                # Compute the total loss (survival + multitask)
                loss = self.compute_loss(k_mb, t_mb, m1_mb, m2_mb, alpha, beta, gamma, delta, eta,outcomes_true, outcome_preds, outcome_configs, missing_mask_fp, basis_mb)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Compute and accumulate individual losses
                survival_loss = alpha * self.loss_log_likelihood(k_mb, m1_mb)  + \
                                beta * self.loss_ranking(k_mb, t_mb, m2_mb) + \
                                gamma * self.loss_calibration(k_mb, m2_mb)
                
                nll = self.loss_log_likelihood(k_mb, m1_mb)
                ranking = self.loss_ranking(k_mb, t_mb, m2_mb)
                calibration = self.loss_calibration(k_mb, m2_mb)
                regularization = self.get_regularization_loss()
                
                outcomes_true = paddle.where(paddle.isnan(outcomes_true), paddle.zeros_like(outcomes_true), outcomes_true)
                multitask_loss = paddle.sum(paddle.stack([
                    # Binary classification loss with missing mask
                    F.binary_cross_entropy(
                        outcome_preds[i].squeeze().index_select(paddle.nonzero(missing_mask_fp[:,0, i]).squeeze()),
                        outcomes_true[:,0, i].index_select(paddle.nonzero(missing_mask_fp[:,0, i]).squeeze())
                    )
                    if config["task_type"] == "binary_classification" else
                    
                    # Multiclass classification loss with missing mask
                    F.cross_entropy(
                        outcome_preds[i].index_select(paddle.nonzero(missing_mask_fp[:, 0,i]).squeeze()),
                        outcomes_true[:, 0,i].index_select(paddle.nonzero(missing_mask_fp[:, 0,i]).squeeze()).astype('int64')
                    )
                    if config["task_type"] == "multiclass_classification" else
                    
                    # Regression loss with missing mask
                    paddle.mean(
                        (outcomes_true[:, 0, i].index_select(paddle.nonzero(missing_mask_fp[:,0, i]).squeeze()) -
                        outcome_preds[i].squeeze().index_select(paddle.nonzero(missing_mask_fp[:,0, i]).squeeze()))**2
                    ) if config["task_type"] == "regression" else

                    # Longitudinal regression loss with missing mask
                    self.mse_longitudinal(outcomes_true[:,i,:,0], outcome_preds[i], missing_mask_fp[:,i,:,0], basis_mb[:,i,:,:])
                    for i, config in enumerate(outcome_configs)
                ]))
                
                total_survival_loss += survival_loss.item()
                total_nll_loss += nll.item()
                total_ranking_loss += ranking.item()
                total_calibration_loss += calibration.item()
                total_multitask_loss += multitask_loss.item()
                total_regularization_loss += regularization.item()
                
            # Calculate averages for the epoch
            avg_loss = total_loss / len(train_loader)
            avg_survival_loss = total_survival_loss / len(train_loader)
            avg_multitask_loss = total_multitask_loss / len(train_loader)
            avg_nll_loss = total_nll_loss / len(train_loader)
            avg_ranking_loss = total_ranking_loss / len(train_loader)
            avg_calibration_loss = total_calibration_loss / len(train_loader)
            avg_regularization_loss = total_regularization_loss / len(train_loader)
            

            # Log training metrics for the epoch
            self.log_writer.add_scalar(tag="Train/Total_Loss", step=epoch, value=avg_loss)
            self.log_writer.add_scalar(tag="Train/Survival_Loss", step=epoch, value=avg_survival_loss)
            self.log_writer.add_scalar(tag="Train/Multitask_Loss", step=epoch, value=avg_multitask_loss)
            self.log_writer.add_scalar(tag="Train/NLL_Loss", step=epoch, value=avg_nll_loss)
            self.log_writer.add_scalar(tag="Train/Ranking_Loss", step=epoch, value=avg_ranking_loss)
            self.log_writer.add_scalar(tag="Train/Calibration_Loss", step=epoch, value=avg_calibration_loss)
            self.log_writer.add_scalar(tag="Train/Regularization_Loss", step=epoch, value=avg_regularization_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Regularization:{avg_regularization_loss:.4f} ,Survival Loss: {avg_survival_loss:.4f}, NLL: {avg_nll_loss:.4f}, Ranking: {avg_ranking_loss:.4f}, Calibration: {avg_calibration_loss:.4f}, Multitask: {avg_multitask_loss:.4f}")

            # Validation phase every 100 epochs
            if (epoch + 1) % 5 == 0:
                self.evaluation(val_loader, outcome_configs)
                surv_metrics = self.surv_metrics
                multitask_metrics = self.multitask_metrics

                # Log validation metrics
                self.log_writer.add_scalar(tag="Validation/C-Index", step=epoch, value=surv_metrics[0])
                self.log_writer.add_scalar(tag="Validation/I_Brier_Score", step=epoch, value=surv_metrics[1])
                
                for i, metric in enumerate(multitask_metrics):
                    self.log_writer.add_scalar(tag=f"Validation/Multitask_{i}_Metric", step=epoch, value=metric)

                print(f"Validation C-index Scores: {surv_metrics[0]}, I Brier Scores: {surv_metrics[1]}, Multitask Metrics: {multitask_metrics}")

                if weights_on_metric:
                    weighted_metric = 0.0
                    for task_metrics, weight in zip(surv_metrics[0:2], weights_on_metric[0:2]):
                        weighted_metric += weight * task_metrics
                    for task_metrics, weight in zip(multitask_metrics, weights_on_metric[2:]):
                        weighted_metric += weight * task_metrics
                else:
                    weighted_metric = surv_metrics[0]  # Default to C-index
                self.log_writer.add_scalar(tag="Validation/Weighted_Metric", step=epoch, value=weighted_metric)

                if weighted_metric > self.best_weighted_metric + min_delta:
                    self.best_weighted_metric = weighted_metric
                    self.best_survival_metric = surv_metrics
                    self.best_multitask_metrics = multitask_metrics
                    paddle.save(self.state_dict(), best_model_path)
                    print(f"Best model updated at epoch {epoch + 1}.")
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}, no improvement for {patience} epochs.")
                    state = paddle.load(best_model_path)
                    self.set_state_dict(state)
                    print(f"Best survival metrics: {self.best_survival_metric}, Best Multi-task metrics: {self.best_multitask_metrics},Best weighted metric: {self.best_weighted_metric}")
                    break
    
    def evaluation(self, val_loader, outcome_configs):

        self.eval()
        with paddle.no_grad():
            x_val = val_loader.dataset[:][0]
            k_val = val_loader.dataset[:][1]
            t_val = val_loader.dataset[:][2]
            outcomes_val = val_loader.dataset[:][5]
            missing_mask_val = val_loader.dataset[:][6]
            missing_mask_fp_val = val_loader.dataset[:][7]
            basis_val = val_loader.dataset[:][8]
            self.mask = missing_mask_val
            # Compute predictions
            cause_out_val, outcome_preds_val= self.forward(x_val)

            # Compute validation metrics
            c_index_scores = overall_cause_specific_c_index(cause_out_val, k_val.flatten(),t_val.flatten(), num_causes_idx=0)
            brier_index_scores, intergrated_bs = cause_specific_intergrated_brier_score(cause_out_val, t_val.flatten(), k_val.flatten(), num_causes_idx=0)

            #auc_scores, iauc = cause_specific_auc(cause_out_val, k_val.flatten(), t_val.flatten(), cause_idx=0,time_grid=self.auc_grid)
            self.surv_metrics = [c_index_scores, intergrated_bs]
                    
            outcomes_val = paddle.where(paddle.isnan(outcomes_val), paddle.zeros_like(outcomes_val), outcomes_val)
            self.multitask_metrics = [
                # Regression metric: Mean Squared Error
                paddle.mean(
                    (outcomes_val[:,0, i].index_select(paddle.nonzero(missing_mask_fp_val[:,0, i]).squeeze()) -
                    outcome_preds_val[i].squeeze().index_select(paddle.nonzero(missing_mask_fp_val[:,0, i]).squeeze()))**2
                ).item()
                if config["task_type"] == "regression" else

                # Binary classification metric: Accuracy
                (outcome_preds_val[i].squeeze().index_select(paddle.nonzero(missing_mask_fp_val[:,0, i]).squeeze()).round() ==
                outcomes_val[:,0, i].index_select(paddle.nonzero(missing_mask_fp_val[:,0, i]).squeeze())).astype('float32').mean().item()
                if config["task_type"] == "binary_classification" else

                # Multiclass classification metric: Accuracy
                paddle.metric.accuracy(
                    outcome_preds_val[i].index_select(paddle.nonzero(missing_mask_fp_val[:,0, i]).squeeze()),
                    outcomes_val[:,0, i].index_select(paddle.nonzero(missing_mask_fp_val[:,0, i]).squeeze()).astype('int64')
                ).item()
                if config["task_type"] == "multiclass_classification" else

                # Longitudinal regression metric: Mean Squared Error
                self.mse_longitudinal(outcomes_val[:,i,:,0], outcome_preds_val[i], missing_mask_fp_val[:,i,:,0], basis_val[:,i,:,:]).item()
                for i, config in enumerate(outcome_configs)
            ]

            



    def predict(self, x, mask):
        self.eval()
        self.mask = mask
        with paddle.no_grad():
            cause_out, outcome_preds = self.forward(x)
        return cause_out, outcome_preds

def create_fc_net(input_dim, num_layers, h_dim, h_fn, o_dim, o_fn, w_init=None, keep_prob=1.0, w_reg=None, use_resnet=False):
    layers = []
    
    # Special case: single-layer network
    if num_layers == 1:
        layers.append(nn.Linear(input_dim, o_dim, weight_attr=w_init, bias_attr=True))
        layers.append(nn.LayerNorm(o_dim))
        if o_fn:
            layers.append(get_activation_fn(o_fn))
        return nn.Sequential(*layers)
    else:
        # Multi-layer network
        # First layer
        layers.append(nn.Linear(input_dim, h_dim, weight_attr=w_init, bias_attr=True))
        layers.append(nn.LayerNorm(h_dim))
        if h_fn:
            layers.append(get_activation_fn(h_fn))
        if keep_prob < 1.0:
            layers.append(nn.Dropout(p=1 - keep_prob))
        
        # Intermediate layers (supports ResNet)
        for layer in range(1, num_layers - 1):
            if use_resnet:
                layers.append(ResidualBlock(h_dim, h_fn, w_init, keep_prob))
            else:
                layers.append(nn.Linear(h_dim, h_dim, weight_attr=w_init, bias_attr=True))
                layers.append(nn.LayerNorm(h_dim))
                if h_fn:
                    layers.append(get_activation_fn(h_fn))
                if keep_prob < 1.0:
                    layers.append(nn.Dropout(p=1 - keep_prob))
        
        # Output layer
        layers.append(nn.Linear(h_dim, o_dim, weight_attr=w_init, bias_attr=True))
        layers.append(nn.LayerNorm(o_dim))

        #layers.append(nn.BatchNorm1D(o_dim))

        if o_fn:
            layers.append(get_activation_fn(o_fn))
        
        # Return the complete Sequential model
        return nn.Sequential(*layers)

class ResidualBlock(nn.Layer):
    def __init__(self, h_dim, h_fn=None, w_init=None, keep_prob=1.0):
        """
        Residual block with optional activation and dropout.

        Args:
            h_dim (int): Dimension of the hidden layer.
            h_fn (str or None): Activation function name (e.g., 'relu', 'tanh'). Default is None.
            w_init (paddle.ParamAttr or None): Weight initializer. Default is None.
            keep_prob (float): Dropout keep probability. Default is 1.0 (no dropout).
        """
        super(ResidualBlock, self).__init__()

     
        self.fc1 = self.add_sublayer(
            "fc1", nn.Linear(h_dim, h_dim, weight_attr=w_init, bias_attr=True)
        )
        
        self.bn1 = self.add_sublayer("bn1", nn.LayerNorm(h_dim)) # BatchNorm1D layer
        
        self.fc2 = self.add_sublayer(
            "fc2", nn.Linear(h_dim, h_dim, weight_attr=w_init, bias_attr=True)
        )
        
        self.bn2 = self.add_sublayer("bn2", nn.LayerNorm(h_dim)) 
       
        self.activation = self.add_sublayer(
            "activation", get_activation_fn(h_fn) if h_fn else None
        )

       
        self.dropout = self.add_sublayer(
            "dropout", nn.Dropout(p=1 - keep_prob) if keep_prob < 1.0 else None
        )

    def forward(self, x):
        """
        Forward pass for the ResidualBlock.
        
        Args:
            x (Tensor): Input tensor with shape [batch_size, h_dim].
        
        Returns:
            Tensor: Output tensor after applying the residual block.
        """
        residual = x  # Store input for residual connection

     
        out = self.fc1(x)
        out = self.bn1(out)
        if self.activation:
            out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)

     
        out = self.fc2(out)
        out = self.bn2(out)

        if self.activation:
            out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)

        out += residual
        return out




def create_outcome_specific_net(input_dim, num_layers, hidden_dim, activation_fn, output_dim, output_activation=None, keep_prob=1.0,use_resnet=False,w_init=None):
    layers = []
    
    if num_layers == 1:
        # Only output layer (no hidden layers)
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim)) 

        if output_activation == 'softmax':
            layers.append(get_activation_fn("softmax"))  # Multi-class classification

        elif output_activation == 'sigmoid':
            layers.append(get_activation_fn("sigmoid"))  # Binary classification


    else:

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))

        if activation_fn:
            layers.append(get_activation_fn(activation_fn))
        if keep_prob < 1.0:
            layers.append(nn.Dropout(p=1 - keep_prob))
        
        # Hidden layers
        for _ in range(1, num_layers - 1):  
            if use_resnet:
                layers.append(ResidualBlock(hidden_dim, activation_fn, w_init, keep_prob))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))

                if activation_fn:
                    layers.append(get_activation_fn(activation_fn))
                if keep_prob < 1.0:
                    layers.append(nn.Dropout(p=1 - keep_prob))

        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim)) 

        #layers.append(nn.BatchNorm1D(output_dim))

        if output_activation == 'softmax':
            layers.append(get_activation_fn("softmax"))  # Multi-class classification

        elif output_activation == 'sigmoid':
            layers.append(get_activation_fn("sigmoid"))         # Binary classification


    
    return nn.Sequential(*layers)


class MultiHeadSelfAttention(nn.Layer):

    def __init__(self, feature_dim, num_heads, w_init=None):
        super(MultiHeadSelfAttention, self).__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be num_heads times head_dim"
        
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads  


        self.query_proj = nn.Linear(feature_dim, feature_dim,weight_attr=w_init)  # (m, m)
        self.key_proj = nn.Linear(feature_dim, feature_dim,weight_attr=w_init)
        self.value_proj = nn.Linear(feature_dim, feature_dim,weight_attr=w_init)

        self.output_proj = nn.Linear(feature_dim, feature_dim,weight_attr=w_init)  # (m, m)

        self.norm = nn.LayerNorm(feature_dim)

        

    def forward(self, x, mask=None):
        """ x: (batch_size, T, feature_dim) """
        batch_size, input_dim, feature_dim = x.shape
        self.mask = mask # (batch_size, input_dim)

        
        Q = self.query_proj(x)  # (batch_size, input_dim, feature_dim)
        K = self.key_proj(x)  # (batch_size, input_dim, feature_dim)
        V = self.value_proj(x)  # (batch_size, input_dim, feature_dim)

        Q = Q.reshape([batch_size, input_dim, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])  # (batch_size, num_heads, input_dim, head_dim)
        K = K.reshape([batch_size, input_dim, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        V = V.reshape([batch_size, input_dim, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        #fake_V = paddle.ones([batch_size, self.num_heads, input_dim, self.head_dim], dtype='float32')
        attention_scores = paddle.matmul(Q, K, transpose_y=True) / (self.head_dim ** 0.5)  # (batch_size, num_heads, input_dim, input_dim)
        if self.mask is not None:
            #missing_mask_attention = paddle.mean(self.mask, axis=-1, keepdim=True)  # (batch_size, T, 1)
            missing_mask_attention = self.mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, input_dim)

            missing_mask_attention = missing_mask_attention.tile([1, self.num_heads, self.mask.shape[1], 1])  # (batch, num_heads, input_dim, input_dim)
            #penalty_strength = 1 - F.sigmoid((missing_mask_attention - 0.5) * 10)
            #penalty_factor = -1e4
            attention_scores = paddle.where(missing_mask_attention == 1, attention_scores, paddle.full_like(attention_scores, -1e9))  # (batch_size, num_heads, input_dim, input_dim)
            #attention_scores = paddle.where(missing_mask_attention < 0.5, paddle.full_like(attention_scores, -1e9), attention_scores)

        

        attention_weights = F.softmax(attention_scores, axis=-1)

        attention_output = paddle.matmul(attention_weights, V)  # (batch_size, num_heads, input_dim, head_dim)

        attention_output = attention_output.transpose([0, 2, 1, 3]).reshape([batch_size, input_dim, feature_dim])

        multi_head_output = self.output_proj(attention_output)  # (batch_size, input_dim, feature_dim)

        return self.norm(multi_head_output)  # (batch_size, input_dim, feature_dim)

class TransformerEncoderLayer(nn.Layer):

    def __init__(self, hidden_dim1,hidden_dim2, num_heads,w_init=None):
        super(TransformerEncoderLayer, self).__init__()
        self.input_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim1,weight_attr=w_init),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2,weight_attr=w_init)
        )

        self.attention = MultiHeadSelfAttention(hidden_dim2, num_heads,w_init=w_init)
        self.attention_norm = nn.LayerNorm(hidden_dim2)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1,weight_attr=w_init),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2,weight_attr=w_init)
        )

        self.ffn_norm = nn.LayerNorm(hidden_dim2)
    def forward(self, x, mask=None):
        """ x: (batch_size, input_dim) """
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        x = self.input_embedding(x)  # (batch_size, input_dim, hidden_dim2)
        attn_out= self.attention(x, mask=mask)  # (batch_size, input_dim, hidden_dim2)
        x = self.attention_norm(x + attn_out)
        ffn_out = self.feed_forward(attn_out)  # (batch_size, input_dim, hidden_dim2)
        x = self.ffn_norm(x + ffn_out)
        return x

class TransformerDecoderLayer(nn.Layer):

    def __init__(self, hidden_dim1,hidden_dim2, w_init=None):
        super(TransformerDecoderLayer, self).__init__()

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1,weight_attr=w_init),
            nn.ReLU(),
            nn.Linear(hidden_dim1, 1,weight_attr=w_init)
        )

    def forward(self, x):

        output = self.output_layer(x)  
        output = output.squeeze(-1)  # (batch_size, input_dim)
        return output

class DTA_AE(nn.Layer):

    def __init__(self, hidden_dim1,hidden_dim2, num_heads,  num_layers=1):
        super(DTA_AE, self).__init__()
        self.num_layers = num_layers
        self.w_init = paddle.nn.initializer.XavierUniform()
        self.encoder_layers = nn.LayerList([
            TransformerEncoderLayer(hidden_dim1,hidden_dim2, num_heads,w_init=self.w_init)
            for _ in range(num_layers)
        ])

        self.decoder_layers = nn.LayerList([
            TransformerDecoderLayer(hidden_dim1, hidden_dim2,w_init=self.w_init)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        """ x: (batch_size, input_dim) """
        for layer in self.encoder_layers:
            x = layer(x, mask)

        encoded_output = x  # (batch_size, input_dim, hidden_dim2)

        for layer in self.decoder_layers:
            x = layer(x)

        reconstructed_output = x  # (batch_size,  input_dim)
        return encoded_output, reconstructed_output

    def fit(self, train_loader, epochs, optimizer, patience=5):
        patience_count = 0
        best_loss = float("inf")

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                X = batch[0]  # X: (batch_size, T, input_dim)
                mask = batch[1]
                random_missing_rate = 0.2
                random_mask = (paddle.rand(mask.shape) > random_missing_rate).astype('float32')
                X_fixed = paddle.where(paddle.isnan(X), paddle.zeros_like(X), X)  


                encoded_output, reconstructed_output = self(X_fixed, random_mask)

                loss_elementwise = (reconstructed_output - X_fixed)**2  
                masked_loss = (loss_elementwise * mask).sum() / mask.sum()
                #loss = F.mse_loss(reconstructed_output*mask, X_fixed*mask)
                loss = masked_loss
                self.ae_loss = loss
                optimizer.clear_grad()
                loss.backward(retain_graph=True)

                optimizer.step()

                total_loss += float(loss)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")            
            if total_loss < best_loss:
                best_loss = total_loss
                patience_count = 0
                paddle.save(self.state_dict(), './saved_model/autoencoder_v2.11_temp_2.pdparams')
                print(f"Model saved at epoch {epoch + 1}")
            else:
                patience_count += 1
                if patience_count == patience:
                    state = paddle.load('./saved_model/autoencoder_v2.11_temp_2.pdparams')
                    self.set_state_dict(state)
                    print(f"Early stopping at epoch {epoch + 1}, the best loss is {best_loss:.6f}, best model has been loaded.")
                    break

# Load the trained model
log_writer = LogWriter(logdir="./logs_app")

input_dims = {
    'x_dim': 68,         # x_dim
    'num_Event': 1, # num_events
    'num_Category': 16 # num_categories (Time horizon for survival time)
}

weights_on_metric = [50,-100,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01,-0.01]

outcome_configs = [
    #{"output_dim": 1, "output_activation": None, "task_type": "regression"},  # Regression task
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"}, # Longitudinal Regression task
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"}, # Longitudinal Regression task
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"}, # Longitudinal Regression task
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"}, # Longitudinal Regression task
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"}, # Longitudinal Regression task
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"}, # Longitudinal Regression task
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"}, # Longitudinal Regression task
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"}, # Longitudinal Regression task
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"}, # Longitudinal Regression task
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"}, # Longitudinal Regression task
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"}, # Longitudinal Regression task
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"}, # Longitudinal Regression task
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"}, # Longitudinal Regression task
    #{"output_dim": 10, "output_activation": None, "task_type": "longitudinal_regression", "basis": basis_tensor, "t_index_fp": t_index_fp_tensor}, # Longitudinal Regression task
    #{"output_dim": 10, "output_activation": None, "task_type": "longitudinal_regression", "basis": basis_tensor, "t_index_fp": t_index_fp_tensor}, # Longitudinal Regression task
    #{"output_dim": 10, "output_activation": None, "task_type": "longitudinal_regression", "basis": basis_tensor, "t_index_fp": t_index_fp_tensor} 
    #{"output_dim": 3, "output_activation": "softmax", "task_type": "multiclass_classification"}  # Multi-class classification task
]
h_dim_shared= 61

h_dim_CS= 55

num_layers_shared= 2

num_layers_CS= 2

learning_rate= 0.006874616232622134

keep_prob= 0.5854197345129215

active_fn= 'relu'

network_settings = {
    'h_dim_shared': h_dim_shared,
    'h_dim_CS': h_dim_CS,
    'num_layers_shared': num_layers_shared,
    'num_layers_CS': num_layers_CS,
    'active_fn': active_fn,
    'keep_prob': keep_prob,
    'initial_W': paddle.nn.initializer.XavierUniform(),
    'ae_out_dim': 28
}
autoencoder = DTA_AE(hidden_dim1=35, hidden_dim2=28, num_heads=1, num_layers=1)

model = ModelDeepHit_Multitask(input_dims, network_settings, outcome_configs  , autoencoder, log_writer)
model.set_state_dict(paddle.load("/scratch/ling2/FSL-Mate/PaddleFSL/examples/molecular_property_prediction/saved_model/model_multitask_deephit_0.9732.pdparams"))
model.eval()

x_mean = np.load("./x_mean.npy")
x_std = np.load("./x_std.npy")
feature_name = np.load("./feature_name.npy", allow_pickle=True)
X_example = pd.read_csv("./example_scd_data.csv")
batch_basis_eval = np.load("./batch_basis_eval.npy")
mean_list = np.load("./mean_long.npy")
std_list = np.load("./std_long.npy")
long_names = np.load("./long_names.npy", allow_pickle=True)

import matplotlib.pyplot as plt
def predict_flat(X_and_mask):
    X = X_and_mask[:, :68]
    mask = X_and_mask[:, 68:]
    X_tensor = paddle.to_tensor(X, dtype='float32')
    mask_tensor = paddle.to_tensor(mask, dtype='float32')

    survival_pred,_ = model.predict(X_tensor, mask_tensor)
    survival_pred = survival_pred[:, 0, :]
    survival_pred = survival_pred.numpy()

    return np.sum(survival_pred[:, 0:5], axis=1)


def create_trajectory_plot(person_id,coeffcients,updated_coeffcients=None):
    batch_basis_eval_tensor = paddle.to_tensor(batch_basis_eval, dtype='float32')
    pred_time = np.linspace(0, 3, 100)
    num_variables = 12

    fig = make_subplots(
        rows=math.ceil(num_variables / 4), cols=4,
        subplot_titles=[long_names[i] for i in range(num_variables)],
        horizontal_spacing=0.05, vertical_spacing=0.15
    )

    for var_idx in range(num_variables):
        row = var_idx // 4 + 1
        col = var_idx % 4 + 1

        basis_tensor_var = batch_basis_eval_tensor[:, var_idx, :, :]
        coeffs_var = coeffcients[var_idx]

        basis_tensor_person = basis_tensor_var[person_id] 
        coeffs_person =  coeffs_var[person_id].unsqueeze(0) 

        curve = paddle.matmul(coeffs_person, basis_tensor_person).squeeze(0).numpy()
        curve = curve * std_list[var_idx] + mean_list[var_idx]

        showlegend_indicator = (var_idx == 0) if updated_coeffcients is not None else False
        fig.add_trace(
            go.Scatter(x=pred_time, y=curve, mode='lines', name='Original', showlegend=showlegend_indicator,line=dict(color='blue')),
            row=row, col=col
        )

        if updated_coeffcients is not None:
            updated_coeffs_var = updated_coeffcients[var_idx][0].unsqueeze(0)
            updated_curve = paddle.matmul(updated_coeffs_var, basis_tensor_person).squeeze(0).numpy()
            updated_curve = updated_curve * std_list[var_idx] + mean_list[var_idx]

            fig.add_trace(
                go.Scatter(x=pred_time, y=updated_curve, mode='lines', name='Updated', showlegend=showlegend_indicator,line=dict(color='red')),
                row=row, col=col
            )




    fig.update_layout(
        height=200 * math.ceil(num_variables / 4),
        title_text=f"Predicted 3-Year Trajectories of Risk Factors for Patient {person_id+1}",
        template='plotly_white'
    )

    return fig


np.random.seed(42)

import dill

with open("./explainer.dill","rb") as f:
    explainer = dill.load(f)

def get_waterfall_base64(X_and_mask_eval,df_combined_with_mask_eval,index, order=None):
    np.random.seed(42)
    shap_values_eval = explainer(X_and_mask_eval[index])
    shap_values_exp_eval = shap_v2.Explanation(
        values=shap_values_eval*100,
        base_values=explainer.expected_value*100,
        data=df_combined_with_mask_eval.iloc[index],
        feature_names=feature_name
    )

    shap_values_exp_eval.data = df_combined_with_mask_eval.loc[index]
    expl = shap_values_exp_eval
    current_order = np.argsort(-np.abs(shap_values_eval.values))

    fig = shap_v2.plots.waterfall_v2(expl, max_display=10, show=False,xlim=(-50,150), order=order)
    buf = io.BytesIO()
    plt.gcf().set_size_inches(10,3)
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{encoded}", current_order

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Sickle Cell Disease Mortality Prediction"

# Layout: UI improvements only
app.layout = dbc.Container(fluid=True, children=[
    # Hidden stores
    dcc.Store(id='memory-predictions'),
    dcc.Store(id='current-patient-index'),
    dcc.Store(id='edited-row'),
    dcc.Store(id='current-order'),
    dcc.Store(id='Current-coefficients'),
    dcc.Store(id='Current-mortality'),

    # Header
    dbc.Row(dbc.Col(html.H2("Sickle Cell Disease Mortality Prediction", className="text-center text-primary my-4"))),

    # Main content
    dbc.Row([
        # Sidebar: upload and instructions
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Upload Data", className="mb-0")),
                dbc.CardBody([
                    dbc.Button(
                        "Download Example Rows",
                        id="btn-download-example",
                        color="secondary",
                        className="w-100 mb-3"
                    ),
                    dcc.Download(id="download-example-csv"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div(['Click to upload patients CSV']),
                        style={
                            'width': '100%', 'height': '80px', 'lineHeight': '80px',
                            'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                            'textAlign': 'center', 'backgroundColor': '#f8f9fa'
                        },
                        multiple=False
                    ),
                    html.Div("Accepted CSV must have exactly 68 numeric columns.", className="text-muted mt-2")
                ])
            ], className="shadow-sm mb-4"),

            dbc.Card([
                dbc.CardHeader(html.H5("Instructions", className="mb-0")),
                dbc.CardBody([
                    html.P("1. Click 'Download Example Rows' to get a sample CSV format."),
                    html.P("2. Upload a CSV file with 68 numeric columns representing patient data."),
                    html.P("3. View predictions and SHAP analysis after uploading data."),
                    html.P("4. Click on a row in the predictions table to view individual predicted mortality plot over 15 years, predicted trajectories of 12 risk factors over 3 years and important baseline variables for predicted 5-year mortality."),
                    html.P("5. Edit the patient features and click 'Update Analysis' to see the updated results."),
                ])
            ], className="shadow-sm mb-4")
        ], width=3),

        # Main panel: tables and plots
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H5("Results", className="mb-0")),
                dbc.CardBody([
                    # Predictions table and plot
                    dcc.Loading(
                        id='loading-table',
                        type='circle',
                        children=html.Div(id='output', className="mt-3")
                    ),
                    html.Hr(),
                    dcc.Loading(
                        id='loading-mortality',
                        type='circle',
                        children=dcc.Graph(id='mortality-plot', config={'displayModeBar': False})
                    ),
                    html.Hr(),
                    dcc.Loading(
                        id='loading-trajectory',
                        type='circle',
                        children=dcc.Graph(id='trajectory-plot', config={'displayModeBar': False})
                    ),
                    html.Hr(),
                    # SHAP analysis
                    dcc.Loading(
                        id='loading-shap',
                        type='circle',
                        children=html.Div(id='shap-plot')
                    ),
                    html.Br(),
                    html.Div(id='feature-editor'),
                    dbc.Button("Update Analysis", id='update-shap-button', color="primary", className="mt-2"),
                    html.Hr(),
                    dcc.Loading(
                        id='loading-update_mortality',
                        type='circle',
                        children=dcc.Graph(id='mortality-plot-updated', config={'displayModeBar': False})
                    ),    
                    html.Hr(),
                    dcc.Loading(
                        id='loading-update_trajectory',
                        type='circle',
                        children=dcc.Graph(id='trajectory-plot-updated', config={'displayModeBar': False})
                    ),      
                    html.Hr(),           
                    dcc.Loading(
                        id='loading-update',
                        type='circle',
                        children=html.Div(id='shap-plot-updated', className="mt-3")
                    )
                ])
            ], className="shadow-sm"), width=9
        )
    ]),

    # Footer
    dbc.Row(dbc.Col(html.Footer(            [
                "Model powered by Multi-Task Deephit v2.11",
                html.Br(),
                "App designed by Gefei Lin",
                html.Br(),
                "Version 1.1.0"
            ], className="text-center text-muted mt-4")))
            
])

@app.callback(
    Output("download-example-csv", "data"),
    Input("btn-download-example", "n_clicks"),
    prevent_initial_call=True
)
def download_example(n_clicks):

    return dcc.send_data_frame(
        X_example.to_csv,
        "example_scd_data.csv",
        index=False
    )

@app.callback(
    Output('output', 'children'),
    Output('memory-predictions', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def predict(contents, filename):
    if contents is None:
        return html.Div("Please upload a CSV file."), dash.no_update

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    # round all values to 5 decimal places
    

    if df.shape[1] != 68:
        return html.Div("Please make sure the CSV contains exactly 68 column of numeric values.")

    mask = ~np.isnan(df)
    df_scaled = (df - x_mean) / x_std
    df_scaled = df_scaled.fillna(0)

    mask_tensor = paddle.to_tensor(mask.to_numpy().astype('float32'))
    input_tensor = paddle.to_tensor(df_scaled.values.astype('float32'))
    predictions, coefficients = model.predict(input_tensor, mask_tensor)
    predictions = predictions[:, 0, :].numpy()
    mortality = np.cumsum(predictions, axis=1)
    mortality[:, -1] = 1

    pred_df = pd.DataFrame(mortality, columns=[f"{i+1}-year Mortality" for i in range(mortality.shape[1])])
    
    
    cols = pred_df.columns.tolist()
    cols[-1] = "Over 15-year Mortality"
    pred_df.columns = cols

    pred_df.insert(0, 'Patient ID', range(1, len(pred_df) + 1))
    df_features = df.copy()
    df_features.insert(0, 'Patient ID', range(1, len(df) + 1))
    coefficients_np = [c.numpy().tolist() for c in coefficients]
    return html.Div([
        html.H5(f"Predictions from: {filename}"),
        html.Div([
            html.Div([
                dash_table.DataTable(
                    id='x-table',
                    data=df_features.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in df_features.columns],
                    style_table={'height': 'auto', 'overflowX': 'auto'},
                    style_cell={'minWidth': '80px', 'whiteSpace': 'normal'},
                )
            ], style={
                'height': '400px', 'overflowY': 'scroll', 'overflowX': 'auto',
                'width': '60%', 'display': 'inline-block'
            }),
            html.Div([
                dash_table.DataTable(
                    data=pred_df.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in pred_df.columns],
                    style_table={'height': 'auto', 'overflowX': 'auto'},
                    style_cell={'minWidth': '100px', 'whiteSpace': 'normal'}
                )
            ], style={
                'height': '400px', 'overflowY': 'scroll', 'overflowX': 'auto',
                'width': '40%', 'display': 'inline-block'
            })
        ]),html.Br(),html.Div("Click on a row to view cumulative mortality plot and important baseline variables at individual level for that patient."),
    ]), {'df_features': df.to_dict('records'), 'pred_df': pred_df.to_dict('records'), 'scaled_df': df_scaled.to_dict('records'), 'mask': mask.values.tolist(), 'coefficients': coefficients_np}

@app.callback(
    Output('mortality-plot', 'figure'),
    Output('Current-mortality', 'data'),
    Input('x-table', 'active_cell'),
    State('memory-predictions', 'data'),
    prevent_initial_call=True
)
def plot_mortality(active_cell, memory):
    if not memory or not active_cell:
        raise dash.exceptions.PreventUpdate

    i = active_cell['row']
    mortality = np.array([
        [v for k, v in row.items() if k != 'Patient ID']
        for row in memory['pred_df']
    ])

    y = mortality[i,:15]
    x = list(range(1, len(y)+1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', line_shape='hv',name=f'Patient {i+1}'))
    fig.update_layout(title=f'Mortality Risk for Patient {i+1}',
                      xaxis_title='Year',
                      yaxis_title='Cumulative Mortality',
                      template='plotly_white',
                      xaxis=dict(tickmode='linear', dtick=1),
                      yaxis=dict(range=[-0.01, 1.01]))
    return fig, y.tolist()

@app.callback(
    Output('trajectory-plot', 'figure'),
    Input('x-table', 'active_cell'),
    State('memory-predictions', 'data'),
    prevent_initial_call=True
)
def plot_trajectory(active_cell, memory):
    if not active_cell or not memory:
        raise dash.exceptions.PreventUpdate

    i = active_cell['row']
    coefficients_data = memory['coefficients']
    coeffcients = [paddle.to_tensor(np.array(c), dtype='float32') for c in coefficients_data]

    fig = create_trajectory_plot(i, coeffcients)
    return fig

@app.callback(
    Output('shap-plot', 'children'),
    Output('feature-editor', 'children'),
    Output('current-patient-index', 'data'),
    Output('edited-row', 'data'),
    Output('current-order', 'data'),
    Input('x-table', 'active_cell'),
    State('memory-predictions', 'data'),
    prevent_initial_call=True
)
def show_shap(active_cell, memory):
    if not active_cell or not memory:
        raise dash.exceptions.PreventUpdate

    i = active_cell['row']
    df_scaled = pd.DataFrame(memory['scaled_df'])
    mask = pd.DataFrame(memory['mask'],dtype='float32')
    X_and_mask_eval = np.concatenate((df_scaled.values, mask.values), axis=1)

    df_display = pd.DataFrame(memory['df_features'])

    df_combined = pd.concat([df_display, mask], axis=1)
    df_combined.columns = feature_name

    img, current_order = get_waterfall_base64(X_and_mask_eval, df_combined, i)


    row_data = df_combined.iloc[i].to_dict()
    table = dash_table.DataTable(
        id='editable-table',
        columns=[{'name': k, 'id': k, 'editable': True} for k in row_data],
        data=[row_data],
        style_table={'overflowX': 'auto'},
        style_cell={'minWidth': '80px', 'whiteSpace': 'normal'}
    )

    return (
        html.Div([
            html.H5(f"SHAP Waterfall Plot for Patient {i+1}"),
            html.Img(src=img, style={'maxWidth': '100%', 'height': 'auto', 'border': '1px solid lightgray'}),
            html.Br(),
            html.Br(),
            html.Div("You can edit the baseline variable values below and click 'Update SHAP' to see the updated SHAP plot.")
        ]),
        table,
        i,
        row_data,
        current_order
    )

@app.callback(
    Output('mortality-plot-updated', 'figure'),
    Output('Current-coefficients', 'data'),
    Input('update-shap-button', 'n_clicks'),
    State('editable-table', 'data'),
    State('current-patient-index', 'data'),
    State('Current-mortality', 'data'),
    prevent_initial_call=True
)
def update_mortality(n_clicks, edited_data, index, current_mortality):
    if not edited_data or index is None:
        raise dash.exceptions.PreventUpdate

    df_raw = pd.DataFrame(edited_data)
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    df_raw_feature = df_raw.iloc[:, :68]  
    mask = ~np.isnan(df_raw_feature)
    df_raw_feature_scaled = (df_raw_feature - x_mean) / x_std
    df_raw_feature_scaled = df_raw_feature_scaled.fillna(0)

    mask_tensor = paddle.to_tensor(mask.to_numpy().astype('float32'))
    input_tensor = paddle.to_tensor(df_raw_feature_scaled.values.astype('float32'))
    predictions, coefficients = model.predict(input_tensor, mask_tensor)
    
    predictions = predictions[:, 0, :].numpy()
    mortality = np.cumsum(predictions, axis=1)
    mortality[:, -1] = 1
    
    coefficients_np = [c.numpy().tolist() for c in coefficients]


    y = mortality[0,:15]
    y_current = np.array(current_mortality)
    
    x = list(range(1, len(y)+1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_current, mode='lines+markers', line_shape='hv',name='Original',line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', line_shape='hv',name='Updated',line=dict(color='red')))
    fig.update_layout(title=f'Updated Mortality Risk for Modified Patient {index+1}',
                      xaxis_title='Year',
                      yaxis_title='Cumulative Mortality',
                      template='plotly_white',
                      xaxis=dict(tickmode='linear', dtick=1),
                      yaxis=dict(range=[-0.01, 1.01]))
    return fig, {'coefficients': coefficients_np} 

@app.callback(
    Output('trajectory-plot-updated', 'figure'),
    Input('Current-coefficients', 'data'),  
    State('memory-predictions', 'data'),
    State('current-patient-index', 'data'),
    prevent_initial_call=True
)

def update_plot_trajectory(memory_coefficients, memory, index):
    if memory_coefficients is None or index is None:
        raise dash.exceptions.PreventUpdate
    coefficients_data = memory['coefficients']
    coeffcients = [paddle.to_tensor(np.array(c), dtype='float32') for c in coefficients_data]

    updated_coeffcients_data = memory_coefficients['coefficients']
    updated_coeffcients = [paddle.to_tensor(np.array(c), dtype='float32') for c in updated_coeffcients_data]

    fig = create_trajectory_plot(index, coeffcients, updated_coeffcients=updated_coeffcients)
    return fig

@app.callback(
    Output('shap-plot-updated', 'children'),
    Input('update-shap-button', 'n_clicks'),
    State('editable-table', 'data'),
    State('memory-predictions', 'data'),
    State('current-patient-index', 'data'),
    State('current-order', 'data'),
    prevent_initial_call=True
)
def update_shap(n_clicks, edited_data, memory, index, current_order):
    if not edited_data or not memory:
        raise dash.exceptions.PreventUpdate

    df_raw = pd.DataFrame(edited_data)
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    df_raw_feature = df_raw.iloc[:, :68]  
    #df_raw_mask = df_raw.iloc[:, 68:]
    df_raw_feature_scaled = (df_raw_feature - x_mean) / x_std
    df_raw_feature_scaled = df_raw_feature_scaled.fillna(0)
    mask = ~np.isnan(df_raw_feature)
    X_and_mask_eval = np.concatenate((df_raw_feature_scaled.values, mask), axis=1)
    df_raw_mask = pd.DataFrame(mask, dtype='float32')
    combined = pd.concat([df_raw_feature, df_raw_mask], axis=1)
    combined.columns = feature_name

    img,_= get_waterfall_base64(X_and_mask_eval, combined, 0, order=current_order)

    return html.Div([
        html.H5(f"Updated SHAP Waterfall Plot for Modified Patient {index+1}"),
        html.Img(src=img, style={'maxWidth': '100%', 'height': 'auto', 'border': '1px solid lightgray'})
    ])




if __name__=='__main__':
    port = int(os.environ.get("PORT", 10000))  
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,        
        use_reloader=False 
    )
