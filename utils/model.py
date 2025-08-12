import pandas as pd
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

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
            time_idx = min(int(time[i,0]-1), num_Category)
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

class ModelDeepHit_Multitask(nn.Layer):
    def __init__(self, input_dims, network_settings, outcome_configs, autoencoder, log_writer=None):
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
    def forward(self, x, mask):
        # Autoencoder

        ae_out, _= self.autoencoder(x, mask = mask)
        
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
                loss = self.mse_longitudinal(outcomes_true[:,i,:,:], outcome_preds[i], missing_mask[:,i,:,0],basis[:,i,:,:])
                #mask_indices = paddle.nonzero(missing_mask[:, :, i]).squeeze()
                #true_values = outcomes_true[:,:, i].index_select(mask_indices)
                #pred_values = outcome_preds[i].index_select(mask_indices)
                #loss = paddle.mean((outcomes_true[:,:,i] - outcome_preds[i])**2 * missing_mask[:,:,i])

            else:
                # Raise an error for unsupported task types
                raise ValueError(f"Unsupported task type: {task_type}")

            outcome_losses.append(loss)



        # Compute the combined multitask loss as the weighted sum of individual losses
        multitask_loss = paddle.sum(
            paddle.stack([w * l for w, l in zip(delta, outcome_losses)])
        )

        # Total loss
        regularization_loss = self.get_regularization_loss()
        #print(regularization_loss)
        #print(f"survival_loss: {survival_loss}, multitask_loss: {multitask_loss}, regularization_loss: {regularization_loss}")

        loss_total = survival_loss + multitask_loss + eta* regularization_loss 

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
        outcome_true_time = outcome_true[:, :, 1]  # Extract time points shape (batch_size, t_dim)
        outcome_true_values = outcome_true[:, :, 0]  # Extract outcome values shape (batch_size, t_dim)
        outcome_true_values = paddle.where(paddle.isnan(outcome_true_values), paddle.zeros_like(outcome_true_values), outcome_true_values)
        outcome_true_time = paddle.where(paddle.isnan(outcome_true_time), paddle.zeros_like(outcome_true_time), outcome_true_time)

        outcome_pred = outcome_pred.unsqueeze(1)  # (batch_size, 1, b_dim)
        time_aware_weight = 0.5**outcome_true_time  # (batch_size, t_dim)

        curve_prediction = paddle.matmul(outcome_pred, basis) # (batch_size,1, t_dim)
        #print(f"curve_prediction shape after matmul: {curve_prediction.shape}")  # Debug print
        curve_prediction = paddle.squeeze(curve_prediction, axis=1)  # (batch_size, t_dim)
        loss_elementwise = time_aware_weight*(outcome_true_values-curve_prediction)**2
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
        best_model_path = "./saved_model/best_model_training_multi_long_v2.14_SCD_2.pdparams"

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
                #self.mask = missing_mask
                x_mb = paddle.where(paddle.isnan(x_mb), paddle.zeros_like(x_mb), x_mb)

                # Forward pass
                cause_out, outcome_preds = self.forward(x_mb,missing_mask)

                # Compute the total loss (survival + multitask)
                loss = self.compute_loss(k_mb, t_mb, m1_mb, m2_mb, alpha, beta, gamma, delta, eta,outcomes_true, outcome_preds, outcome_configs, missing_mask_fp,basis_mb)

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
                    self.mse_longitudinal( outcomes_true[:,i,:,:], outcome_preds[i], missing_mask_fp[:,i,:,0],basis_mb[:,i,:,:])

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
            if self.log_writer is not None:
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
                if self.log_writer is not None:
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
                
                if self.log_writer is not None:
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
            #self.mask = missing_mask_val
            #x_for_long_val = x_val[:, self.indices_long_in_x]  # Extract features for longitudinal outcomes
            #missing_mask_x_val = missing_mask_val[:, self.indices_long_in_x]  # Extract missing mask for longitudinal outcomes
        

            # Compute predictions
            cause_out_val, outcome_preds_val= self.forward(x_val,missing_mask_val)

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
                self.mse_longitudinal( outcomes_val[:,i,:,:], outcome_preds_val[i], missing_mask_fp_val[:,i,:,0],basis_val[:,i,:,:]).item()
                for i, config in enumerate(outcome_configs)
            ]

            



    def predict(self, x, mask):
        self.eval()
        #self.mask = mask
        with paddle.no_grad():
            cause_out, outcome_preds = self.forward(x,mask)
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
        #self.mask = mask # (batch_size, input_dim)

        
        Q = self.query_proj(x)  # (batch_size, input_dim, feature_dim)
        K = self.key_proj(x)  # (batch_size, input_dim, feature_dim)
        V = self.value_proj(x)  # (batch_size, input_dim, feature_dim)

        Q = Q.reshape([batch_size, input_dim, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])  # (batch_size, num_heads, input_dim, head_dim)
        K = K.reshape([batch_size, input_dim, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        V = V.reshape([batch_size, input_dim, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        #fake_V = paddle.ones([batch_size, self.num_heads, input_dim, self.head_dim], dtype='float32')
        attention_scores = paddle.matmul(Q, K, transpose_y=True) / (self.head_dim ** 0.5)  # (batch_size, num_heads, input_dim, input_dim)
        if mask is not None:
            #missing_mask_attention = paddle.mean(self.mask, axis=-1, keepdim=True)  # (batch_size, T, 1)
            missing_mask_attention = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, input_dim)

            missing_mask_attention = missing_mask_attention.tile([1, self.num_heads, mask.shape[1], 1])  # (batch, num_heads, input_dim, input_dim)
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
                paddle.save(self.state_dict(), './saved_model/autoencoder_v2.13_temp_2.pdparams')
                print(f"Model saved at epoch {epoch + 1}")
            else:
                patience_count += 1
                if patience_count == patience:
                    state = paddle.load('./saved_model/autoencoder_v2.13_temp_2.pdparams')
                    self.set_state_dict(state)
                    print(f"Early stopping at epoch {epoch + 1}, the best loss is {best_loss:.6f}, best model has been loaded.")
                    break