import dash
from dash import html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dash_table.Format import Format, Scheme
import math
import os
import io
import base64
import time
import dill
import matplotlib.pyplot as plt
from utils.model import *
import shap_v2

########################## Load model and required files ###########################

input_dims = {
    'x_dim': 68,         # x_dim
    'num_Event': 1, # num_events
    'num_Category': 16 # num_categories (Time horizon for survival time)
}

outcome_configs = [

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
    {"output_dim": 5, "output_activation": None, "task_type": "longitudinal_regression"} # Longitudinal Regression task

]

hidden_dim1 = 62
num_heads_ae = 4
hidden_dim2 = 24
h_dim_shared= 63
h_dim_CS= 59
num_layers_shared= 4
num_layers_CS= 3
keep_prob= 0.7291386841658769
active_fn= 'relu'

network_settings = {
    'h_dim_shared': h_dim_shared,
    'h_dim_CS': h_dim_CS,
    'num_layers_shared': num_layers_shared,
    'num_layers_CS': num_layers_CS,
    'active_fn': active_fn,
    'keep_prob': keep_prob,
    'initial_W': paddle.nn.initializer.XavierUniform(),
    'ae_out_dim': hidden_dim2
}

autoencoder = DTA_AE(hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, num_heads=num_heads_ae, num_layers=1)

model = ModelDeepHit_Multitask(input_dims, network_settings, outcome_configs  , autoencoder)
model.set_state_dict(paddle.load("./utils/saved_models/model_multitask_deephit_v2.14_0.9829.pdparams"))
model.eval()

model_copy = ModelDeepHit_Multitask(input_dims, network_settings, outcome_configs  , autoencoder)
model_copy.set_state_dict(paddle.load("./utils/saved_models/model_multitask_deephit_v2.14_0.9829.pdparams"))
model_copy.eval()

x_mean = np.load("./utils/data_info/x_mean.npy")
x_std = np.load("./utils/data_info/x_std.npy")
feature_name = np.load("./utils/data_info/feature_name.npy", allow_pickle=True)
X_example = pd.read_csv("./utils/data_examples/example_scd_data.csv")
calibration_example = pd.read_csv("./utils/data_examples/example_scd_for_calibration.csv")
batch_basis_eval = np.load("./utils/data_info/batch_basis_eval.npy")
mean_list = np.load("./utils/data_info/mean_long.npy")
std_list = np.load("./utils/data_info/std_long.npy")
long_names = np.load("./utils/data_info/long_names.npy", allow_pickle=True)
with open("./utils/saved_explainers/explainer_v2.14.2.dill","rb") as f:
    explainer = dill.load(f)
explainer.model.f.__globals__['model'] = model

# Global simultaneous uncertainty band progress tracking
mc_progress = {'current': 0, 'total': 0, 'running': False, 'eta': '', 'label': '', 'trigger_ts': 0.0}

# Long-horizon risk-support table for mortality simultaneous uncertainty band
# t is in years.
DEFAULT_N_RISK_TABLE = {
    0: 598,
    1: 598,
    2: 543,
    3: 484,
    4: 409,
    5: 365,
    6: 318,
    7: 278,
    8: 239,
    9: 214,
    10: 189,
    11: 167,
    12: 139,
    13: 112,
    14: 90,
    15: 64,
}

DEFAULT_LONG_HORIZON_GAMMA = 0
DEFAULT_LONG_HORIZON_START_YEAR = 1
DEFAULT_MORTALITY_DISPLAY_HORIZON = 10

ALPHA_MARKS_DESKTOP = {
    0.01: '99%',
    0.05: '95%',
    0.1: '90%',
    0.2: '80%',
    0.3: '70%',
    0.5: '50%'
}

ALPHA_MARKS_MOBILE = {
    0.01: '99%',
    0.1: '90%',
    0.2: '80%',
    0.3: '70%',
    0.5: '50%'
}


def get_mobile_43_height(window_width, min_height=300, max_height=520):
    """Compute mobile chart height using ~4:3 aspect ratio based on viewport width."""
    if window_width is None:
        return 400
    plot_width = max(float(window_width) - 36.0, 280.0)
    return int(np.clip(plot_width * 3.0 / 4.0, min_height, max_height))

#######################refine utility functions for the app ###########################

def create_kpi_cards(mortality_values, lower_bounds=None, upper_bounds=None, accent_color="#0067B1"):
    values = np.asarray(mortality_values, dtype=np.float64).reshape(-1)
    lower = None if lower_bounds is None else np.asarray(lower_bounds, dtype=np.float64).reshape(-1)
    upper = None if upper_bounds is None else np.asarray(upper_bounds, dtype=np.float64).reshape(-1)

    def risk_with_bounds_at(year_idx):
        if year_idx < len(values):
            risk_text = f"{values[year_idx] * 100:.1f}%"
            if lower is not None and upper is not None and year_idx < len(lower) and year_idx < len(upper):
                return f"{risk_text} [{lower[year_idx] * 100:.1f}%, {upper[year_idx] * 100:.1f}%]"
            return risk_text
        return "—"

    cards = [
        ("3-Year Risk", risk_with_bounds_at(2)),
        ("5-Year Risk", risk_with_bounds_at(4)),
        ("8-Year Risk", risk_with_bounds_at(7)),
        ("10-Year Risk", risk_with_bounds_at(9)),
    ]

    return html.Div([
        html.Div([
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value", style={"color": accent_color}),
        ], className="kpi-card")
        for label, value in cards
    ], className="kpi-row")


def build_long_horizon_weights(
    horizon_len,
    gamma=DEFAULT_LONG_HORIZON_GAMMA,
    n_risk_table=None,
    start_year=DEFAULT_LONG_HORIZON_START_YEAR,
):
    """Build long-horizon penalty weights.

        Year indexing follows the app horizon:
            - Year 1 uses n_risk(t=0)
            - Year y uses n_risk(t=y-1)

        For year >= start_year,
        w(year) = (n_risk(0) / max(n_risk(t=year-1), 1))^gamma.
    """
    if horizon_len <= 0:
        return np.ones((0,), dtype=np.float32)

    table = DEFAULT_N_RISK_TABLE if n_risk_table is None else n_risk_table
    table = {int(k): max(int(v), 1) for k, v in table.items()}

    if 0 not in table:
        min_key = min(table.keys())
        table[0] = table[min_key]

    n0 = max(int(table.get(0, 1)), 1)
    gamma = float(max(0.0, gamma))
    start_year = int(max(1, start_year))

    sorted_keys = sorted(table.keys())
    last_n = n0
    weights = []
    for year in range(1, horizon_len + 1):
        if year < start_year:
            weights.append(1.0)
            continue

        t_idx = year - 1
        if t_idx in table:
            last_n = table[t_idx]
        else:
            for k in sorted_keys:
                if k <= t_idx:
                    last_n = table[k]
                else:
                    break
        wt = (n0 / max(last_n, 1)) ** gamma
        weights.append(wt)

    return np.asarray(weights, dtype=np.float32)

def create_trajectory_plot(
    person_id,
    coeffcients,
    updated_coeffcients=None,
    window_width=None,
    original_band=None,
    updated_band=None,
    alpha=0.05,
):
    batch_basis_eval_tensor = paddle.to_tensor(batch_basis_eval, dtype='float32')
    pred_time = np.linspace(0, 3, 100)
    num_variables = 12
    num_cols = 3
    num_rows = math.ceil(num_variables / num_cols)

    # Responsive height: compact on mobile
    is_mobile = window_width is not None and window_width < 768
    height_per_row = 130 if is_mobile else 190
    fig_height = height_per_row * num_rows
    legend_font_size = 8 if is_mobile else 10
    margin_left = 20 if is_mobile else 50
    margin_right = 8 if is_mobile else 20

    # Nice palette
    COLOR_ORIGINAL = "#3B82F6"  # vivid blue
    COLOR_UPDATED  = "#F97316"  # warm orange

    fig = make_subplots(
        rows=num_rows, cols=num_cols,
        subplot_titles=[long_names[i] for i in range(num_variables)],
        horizontal_spacing=0.07, vertical_spacing=0.08
    )

    for var_idx in range(num_variables):
        row = var_idx // num_cols + 1
        col = var_idx % num_cols + 1

        basis_tensor_var = batch_basis_eval_tensor[:, var_idx, :, :]
        coeffs_var = coeffcients[var_idx]

        basis_tensor_person = basis_tensor_var[person_id]
        coeffs_person = coeffs_var[person_id].unsqueeze(0)

        curve = paddle.matmul(coeffs_person, basis_tensor_person).squeeze(0).numpy()
        curve = curve * std_list[var_idx] + mean_list[var_idx]
        curve = np.clip(curve, 0, None)  # values cannot be negative
        lo = None
        up = None

        showlegend_indicator = (var_idx == 0) if updated_coeffcients is not None else False

        if original_band is not None:
            lo = np.asarray(original_band['lower'][var_idx], dtype=np.float64)
            up = np.asarray(original_band['upper'][var_idx], dtype=np.float64)
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([pred_time, pred_time[::-1]]),
                    y=np.concatenate([up, lo[::-1]]),
                    fill='toself',
                    fillcolor='rgba(59,130,246,0.16)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                        name=f"Orig band {int((1-alpha)*100)}%",
                    showlegend=(var_idx == 0),
                ),
                row=row, col=col
            )

        original_customdata = np.column_stack([lo, up]) if lo is not None and up is not None else None
        original_hover = "%{y:.2f} [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra>%{fullData.name}</extra>" if original_customdata is not None else "%{y:.2f}<extra>%{fullData.name}</extra>"
        fig.add_trace(
            go.Scatter(
                x=pred_time, y=curve, mode='lines', name='Original',
                showlegend=showlegend_indicator,
                line=dict(color=COLOR_ORIGINAL, width=2),
                customdata=original_customdata,
                hovertemplate=original_hover,
            ),
            row=row, col=col
        )

        if updated_coeffcients is not None:
            updated_coeffs_var = updated_coeffcients[var_idx][0].unsqueeze(0)
            updated_curve = paddle.matmul(updated_coeffs_var, basis_tensor_person).squeeze(0).numpy()
            updated_curve = updated_curve * std_list[var_idx] + mean_list[var_idx]
            updated_curve = np.clip(updated_curve, 0, None)  # values cannot be negative
            lo_u = None
            up_u = None

            if updated_band is not None:
                lo_u = np.asarray(updated_band['lower'][var_idx], dtype=np.float64)
                up_u = np.asarray(updated_band['upper'][var_idx], dtype=np.float64)
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([pred_time, pred_time[::-1]]),
                        y=np.concatenate([up_u, lo_u[::-1]]),
                        fill='toself',
                        fillcolor='rgba(249,115,22,0.16)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo='skip',
                        name=f"Upd band {int((1-alpha)*100)}%",
                        showlegend=(var_idx == 0),
                    ),
                    row=row, col=col
                )

            updated_customdata = np.column_stack([lo_u, up_u]) if lo_u is not None and up_u is not None else None
            updated_hover = "%{y:.2f} [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra>%{fullData.name}</extra>" if updated_customdata is not None else "%{y:.2f}<extra>%{fullData.name}</extra>"
            fig.add_trace(
                go.Scatter(
                    x=pred_time, y=updated_curve, mode='lines', name='Updated',
                    showlegend=showlegend_indicator,
                    line=dict(color=COLOR_UPDATED, width=2),
                    customdata=updated_customdata,
                    hovertemplate=updated_hover,
                ),
                row=row, col=col
            )

        # x-axis "Years" only on bottom row; y-axis "Values" only on first column
        if row == num_rows:
            fig.update_xaxes(title_text="Years", title_font=dict(size=10), row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text="Values", title_font=dict(size=10), title_standoff=2, row=row, col=col)

    trajectory_title = (
        f"Predicted 3-Year Trajectories of Risk Factors<br>Patient {person_id + 1}"
        if is_mobile
        else f"Predicted 3-Year Trajectories of Risk Factors — Patient {person_id + 1}"
    )

    fig.update_layout(
        height=fig_height,
        title_text=trajectory_title,
        title_font=dict(size=15, color="#003087", family="Segoe UI, Arial"),
        template='plotly_white',
        plot_bgcolor="rgba(248,251,255,0.9)",
        paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=11, color="#444"),
        margin=dict(t=105, b=50, l=margin_left, r=margin_right),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.60)", bordercolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(size=legend_font_size), itemsizing="constant"
        ),
    )
    fig.update_annotations(font=dict(size=10, color="#555"))

    return fig


class PaddleWrapper:
    def __init__(self, paddle_model,horizon=5):
        self.model_original = paddle_model
        self.horizon = horizon
        

    def predict(self, X):
        X_feature = X[:, :68]
        mask = X[:, 68:]
        X_tensor = paddle.to_tensor(X_feature, dtype='float32')
        mask_tensor = paddle.to_tensor(mask, dtype='float32')

        survival_pred,_ = self.model_original.predict(X_tensor, mask_tensor)
        survival_pred = survival_pred[:, 0, :]
        survival_pred = survival_pred.numpy()

        return np.sum(survival_pred[:, 0:self.horizon], axis=1)

def paddle_manual_conformal_interval(paddle_model, X_calib, y_calib, X_test, alpha=0.1,horizon=5):
    model_wraped = PaddleWrapper(paddle_model,horizon=horizon)

    alpha = 0.05 if alpha is None else float(alpha)
    alpha = float(np.clip(alpha, 1e-6, 0.999))
    X_calib = np.asarray(X_calib, dtype=np.float32)
    y_calib = np.asarray(y_calib, dtype=np.float32).reshape(-1)
    X_test = np.asarray(X_test, dtype=np.float32)

    y_pred_calib = model_wraped.predict(X_calib)

    residuals = np.abs(y_calib - y_pred_calib.reshape(-1)) 
    n = len(y_calib)
    if n == 0:
        raise ValueError("Calibration set is empty after filtering.")
    # Practical quantile to avoid saturation to max residual when n is small.
    k = int(np.ceil((1 - alpha) * (n + 1)))
    k = min(max(k, 1), n)
    tau = (k - 0.5) / n
    tau = float(np.clip(tau, 0.0, 1.0))
    q = np.quantile(residuals, tau, method='linear')
    #q = np.quantile(residuals, 1 - alpha)
    
    y_pred_test = model_wraped.predict(X_test)

    lower_bound = np.clip(y_pred_test - q, 0, 1)
    upper_bound = np.clip(y_pred_test + q, 0, 1)
    #lower_bound = y_pred_test - q
    #upper_bound = y_pred_test + q
    intervals = np.vstack([lower_bound, upper_bound]).T
    return y_pred_test, intervals,residuals


def _pava_non_decreasing(values):
    """Project 1D sequence to a non-decreasing sequence via PAVA."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size <= 1:
        return x

    blocks = []  # [start, end, mean]
    for i, v in enumerate(x):
        blocks.append([i, i, float(v)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            n1 = b1[1] - b1[0] + 1
            n2 = b2[1] - b2[0] + 1
            m = (b1[2] * n1 + b2[2] * n2) / (n1 + n2)
            blocks.append([b1[0], b2[1], m])

    y = np.empty_like(x)
    for s, e, m in blocks:
        y[s:e + 1] = m
    return y

def conformal_mortality_prediction(
    model_original,
    X_and_mask,
    E_train,
    T_train,
    X_and_mask_test,
    max_horizon=15,
    alpha=0.05
):
    X_and_mask = np.asarray(X_and_mask, dtype=np.float32)
    X_and_mask_test = np.asarray(X_and_mask_test, dtype=np.float32)
    if X_and_mask_test.ndim == 1:
        X_and_mask_test = X_and_mask_test.reshape(1, -1)

    E_train = np.asarray(E_train).reshape(-1).astype(np.int64)
    T_train = np.asarray(T_train).reshape(-1).astype(np.int64)

    years = []
    predicted_mortality = []
    lower_bounds = []
    upper_bounds = []

    for i in range(max_horizon):
        censored_mask = (E_train == 0)
        uncensored_mask = (E_train == 1)
        censored_alive_mask = censored_mask & (T_train >= i + 1)
        available_mask = (uncensored_mask | censored_alive_mask)

        X_and_mask_available = X_and_mask[available_mask]
        

        y_label = np.zeros(len(E_train), dtype=np.float32)
        for j in range(len(E_train)):
            if E_train[j] == 1 and T_train[j] <= i + 1:
                y_label[j] = 1
            elif E_train[j] == 1 and T_train[j] > i + 1:
                y_label[j] = 0
            elif E_train[j] == 0 and T_train[j] >= i + 1:
                y_label[j] = 0

        y_label_available = y_label[available_mask]

        y_pred_test, intervals, residuals = paddle_manual_conformal_interval(
            model_original, X_and_mask_available, y_label_available,
            X_and_mask_test, alpha=alpha, horizon=i + 1
        )

        years.append(i + 1)
        predicted_mortality.append(y_pred_test[0])
        lower_bounds.append(intervals[0, 0])
        upper_bounds.append(intervals[0, 1])

        print(f'At year {i+1}, predicted mortality: {y_pred_test[0]:.4f}, '
              f'interval: [{intervals[0, 0]:.4f}, {intervals[0, 1]:.4f}]')

    corrected_lower_bounds = _pava_non_decreasing(lower_bounds)
    corrected_upper_bounds = _pava_non_decreasing(upper_bounds)
    corrected_upper_bounds = np.maximum(corrected_upper_bounds, corrected_lower_bounds)

    corrected_lower_bounds = np.clip(corrected_lower_bounds, 0.0, 1.0).tolist()
    corrected_upper_bounds = np.clip(corrected_upper_bounds, 0.0, 1.0).tolist()

    return {
        'years': years,
        'predicted_mortality': predicted_mortality,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'corrected_lower_bounds': corrected_lower_bounds,
        'corrected_upper_bounds': corrected_upper_bounds
    }


def mc_dropout_predict(
    model_obj,
    input_tensor,
    mask_tensor,
    n_samples=1000,
    alpha=0.05,
    label='',
    mc_dropout_p=0.01,
    long_horizon_gamma=DEFAULT_LONG_HORIZON_GAMMA,
    long_horizon_start_year=DEFAULT_LONG_HORIZON_START_YEAR,
    n_risk_table=None,
):
    """Run simultaneous uncertainty band sampling and construct log-log bands on CIF scale.

    Steps:
        1) Generate MC CIF curves F^(m)(t)
        2) Transform with g(F) = log(-log(1-F))
           3) Compute se_Y(t) = sd(Y^(1)(t), ..., Y^(M)(t))
          4) Baseline standardized process Z_base^(m)(t) = (Y^(m)(t)-Y(t))/se_Y(t)
          5) S^(m) = max_t |Z_base^(m)(t)| and c_{1-alpha} = quantile(S, 1-alpha)
          6) Long-horizon penalty: se_tilde(t) = w(t) * se_Y(t),
              w(t) = (n_risk(0) / max(n_risk(t), 1))^gamma
          7) Band on transformed scale: Y +/- c*se_tilde
        7) Back-transform: F = 1 - exp(-exp(Y))

    Uses forward() directly instead of predict(), because predict() calls
    self.eval() internally which disables dropout.
    """
    # Save random state, set fixed seed for reproducibility, then restore
    np_state = np.random.get_state()
    np.random.seed(42)
    paddle.seed(42)
    global mc_progress

    # -- Temporarily lower every Dropout layer's p to mc_dropout_p ----------
    dropout_layers = [m for m in model_obj.sublayers()
                      if isinstance(m, paddle.nn.Dropout)]
    original_p = [m.p for m in dropout_layers]
    for m in dropout_layers:
        m.p = mc_dropout_p

    # Also patch the functional F.dropout call in forward() which uses self.keep_prob
    original_keep_prob = model_obj.keep_prob
    model_obj.keep_prob = mc_dropout_p
    # -------------------------------------------------------------------------

    model_obj.train()
    predictions_list = []
    mc_progress.update({'current': 0, 'total': n_samples, 'running': True, 'eta': '', 'label': label})
    t_start = time.time()
    for i in range(n_samples):
        with paddle.no_grad():
            preds, _ = model_obj.forward(input_tensor, mask_tensor)
        preds = preds[:, 0, :].numpy()
        mort = np.cumsum(preds, axis=1)
        mort[:, -1] = 1
        predictions_list.append(mort)
        if (i + 1) % 10 == 0 or i == n_samples - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_samples - i - 1) / rate if rate > 0 else 0
            mc_progress.update({'current': i + 1, 'eta': f'{remaining:.1f}s'})
    model_obj.eval()
    # -- Restore original dropout rates --------------------------------------
    for m, p in zip(dropout_layers, original_p):
        m.p = p
    model_obj.keep_prob = original_keep_prob
    # -------------------------------------------------------------------------
    np.random.set_state(np_state)
    mc_progress.update({'current': n_samples, 'running': False, 'eta': '', 'label': ''})

    all_preds = np.stack(predictions_list, axis=0)  # (M, B, T)
    mean = np.clip(np.mean(all_preds, axis=0), 0, 1)

    # Log-log transform g(F)=log(-log(1-F)) and inverse g^{-1}(Y)=1-exp(-exp(Y))
    eps = 1e-6
    alpha = float(np.clip(alpha, 1e-6, 0.999))
    F_mc = np.clip(all_preds, eps, 1.0 - eps)
    Y_mc = np.log(-np.log(1.0 - F_mc))

    F_hat = np.clip(mean, eps, 1.0 - eps)
    Y_hat = np.log(-np.log(1.0 - F_hat))

    se_Y_base = np.std(Y_mc, axis=0, ddof=1)
    se_Y_base = np.maximum(se_Y_base, eps)

    # Compute simultaneous-band critical value on baseline scale first.
    # This keeps short-horizon widths unchanged when w(t)=1 (e.g., year 1).
    Z_base = (Y_mc - Y_hat[None, :, :]) / se_Y_base[None, :, :]
    S = np.max(np.abs(Z_base), axis=2)  # (M, B)
    c = np.quantile(S, 1.0 - alpha, axis=0, method='linear')  # (B,)

    # Long-horizon penalty based on training-set risk support:
    # se_tilde_i(t) = w(t) * se_i(t), where t starts at year 1.
    horizon_len = se_Y_base.shape[1]
    w = build_long_horizon_weights(
        horizon_len=horizon_len,
        gamma=long_horizon_gamma,
        n_risk_table=n_risk_table,
        start_year=long_horizon_start_year,
    )
    se_Y = np.maximum(se_Y_base * w[None, :], eps)

    L_Y = Y_hat - c[:, None] * se_Y
    U_Y = Y_hat + c[:, None] * se_Y

    lower = 1.0 - np.exp(-np.exp(L_Y))
    upper = 1.0 - np.exp(-np.exp(U_Y))
    lower = np.clip(lower, 0.0, 1.0).astype(np.float32)
    upper = np.clip(upper, 0.0, 1.0).astype(np.float32)

    return lower, upper, mean


def mc_dropout_trajectory_band(model_obj, input_tensor, mask_tensor, person_id, n_samples=1000, alpha=0.05, mc_dropout_p=0.01, label='Trajectory'):
    """MC simultaneous band for trajectories using standardized sup-process.

    Constructs, for each variable independently:
      Z^(m)(t) = (Y^(m)(t) - Y(t)) / se_Y(t),
      S^(m)    = max_t |Z^(m)(t)|,
      c_{1-a}  = quantile(S, 1-a),
      band     = Y(t) ± c_{1-a} se_Y(t).
    """
    global mc_progress
    np_state = np.random.get_state()
    np.random.seed(42)
    paddle.seed(42)

    dropout_layers = [m for m in model_obj.sublayers() if isinstance(m, paddle.nn.Dropout)]
    original_p = [m.p for m in dropout_layers]
    for m in dropout_layers:
        m.p = mc_dropout_p
    original_keep_prob = model_obj.keep_prob
    model_obj.keep_prob = mc_dropout_p

    batch_basis_eval_tensor = paddle.to_tensor(batch_basis_eval, dtype='float32')
    num_variables = 12
    eps = 1e-6
    alpha = float(np.clip(alpha, 1e-6, 0.999))

    curves_mc = []
    mc_progress.update({'current': 0, 'total': n_samples, 'running': True, 'eta': '', 'label': label})
    t_start = time.time()
    model_obj.train()
    for i in range(n_samples):
        with paddle.no_grad():
            _, coeffs_sample = model_obj.forward(input_tensor, mask_tensor)

        one_sample_curves = np.zeros((num_variables, 100), dtype=np.float32)
        for var_idx in range(num_variables):
            basis_tensor_var = batch_basis_eval_tensor[:, var_idx, :, :]
            basis_tensor_person = basis_tensor_var[person_id]
            coeffs_person = coeffs_sample[var_idx][0].unsqueeze(0)
            curve = paddle.matmul(coeffs_person, basis_tensor_person).squeeze(0).numpy()
            curve = curve * std_list[var_idx] + mean_list[var_idx]
            curve = np.clip(curve, 0, None)
            one_sample_curves[var_idx] = curve.astype(np.float32)
        curves_mc.append(one_sample_curves)
        if (i + 1) % 5 == 0 or i == n_samples - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_samples - i - 1) / rate if rate > 0 else 0
            mc_progress.update({'current': i + 1, 'eta': f'{remaining:.1f}s'})

    model_obj.eval()
    for m, p in zip(dropout_layers, original_p):
        m.p = p
    model_obj.keep_prob = original_keep_prob
    np.random.set_state(np_state)
    mc_progress.update({'current': n_samples, 'running': False, 'eta': '', 'label': ''})

    curves_mc = np.stack(curves_mc, axis=0)  # (M, V, T)
    center = np.mean(curves_mc, axis=0)
    se = np.std(curves_mc, axis=0, ddof=1)
    se = np.maximum(se, eps)

    Z = (curves_mc - center[None, :, :]) / se[None, :, :]
    S = np.max(np.abs(Z), axis=2)  # (M, V)
    c = np.quantile(S, 1.0 - alpha, axis=0, method='linear')  # (V,)

    lower = center - c[:, None] * se
    upper = center + c[:, None] * se
    lower = np.clip(lower, 0.0, None).astype(np.float32)
    upper = np.clip(upper, 0.0, None).astype(np.float32)

    return lower, upper, center


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

#################################### Dash app #################################

# Initialize Dash app
BRAND_COLOR   = "#003087"   # NHLBI deep navy
ACCENT_COLOR  = "#0067B1"   # NHLBI medium blue
DANGER_COLOR  = "#C8102E"   # NHLBI red
SUCCESS_COLOR = "#2dc653"

custom_css = {
    "body": {"background": "#f0f4f8"},
}

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css",
    ],
    suppress_callback_exceptions=True,
)
app.title = "Sickle Cell Disease Mortality Prediction"

# ── global inline CSS injected into <head> ──────────────────────────────────
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
            body {
                background:
                    radial-gradient(circle at top left, rgba(70,120,255,.10), transparent 24%),
                    radial-gradient(circle at top right, rgba(0,160,220,.08), transparent 18%),
                    linear-gradient(180deg, #f6f9fd 0%, #eef3f8 100%) !important;
                font-family: "Segoe UI", Arial, sans-serif;
            }

      /* ── upload drop-zone ── */
      .upload-zone {
        width: 100%; height: 80px; line-height: 80px;
                border: 1.5px dashed rgba(0,103,177,.45); border-radius: 12px;
                text-align: center; background: linear-gradient(180deg, #fbfdff 0%, #eef6ff 100%);
                color: #0067B1; cursor: pointer;
                transition: background .2s, transform .18s ease, box-shadow .18s ease, border-color .18s ease;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.8), 0 4px 14px rgba(0,61,128,.06);
      }
            .upload-zone:hover {
                background: linear-gradient(180deg, #f2f8ff 0%, #dcecff 100%);
                border-color: rgba(0,103,177,.72);
                transform: translateY(-1px);
                box-shadow: inset 0 1px 0 rgba(255,255,255,.85), 0 8px 20px rgba(0,61,128,.10);
            }

      /* ── card tweaks ── */
            .card {
                border: 1px solid rgba(213,223,236,.95) !important;
                border-radius: 16px !important;
                box-shadow: 0 10px 24px rgba(18,52,86,.08), 0 2px 8px rgba(18,52,86,.05) !important;
                background: rgba(255,255,255,.96) !important;
                overflow: hidden;
            }
            .card-header {
                border-radius: 16px 16px 0 0 !important;
                letter-spacing: .15px;
            }
            .app-card .card-body {
                background: linear-gradient(180deg, rgba(255,255,255,.98) 0%, rgba(248,251,255,.98) 100%);
            }
            .app-results-card .card-body {
                background: linear-gradient(180deg, #fcfdff 0%, #f4f8fc 100%);
            }

            .accordion-item {
                border: 1px solid rgba(220,228,238,.95) !important;
                border-radius: 14px !important;
                overflow: hidden;
                margin-bottom: 12px;
                box-shadow: 0 4px 12px rgba(16,38,66,.05);
            }
            .accordion-button {
                background: linear-gradient(180deg, #ffffff 0%, #f6f9fd 100%) !important;
                box-shadow: none !important;
                font-weight: 600;
            }
            .accordion-button:not(.collapsed) {
                background: linear-gradient(180deg, #f7fbff 0%, #eef6ff 100%) !important;
                color: #003087 !important;
            }
            .accordion-button:focus {
                box-shadow: 0 0 0 .18rem rgba(0,103,177,.12) !important;
                border-color: rgba(0,103,177,.15) !important;
            }
            .accordion-body {
                background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
            }

            .app-subtle-panel {
                background: linear-gradient(180deg, #f9fbff 0%, #eef5ff 100%);
                border: 1px solid rgba(213,223,236,.95);
                border-radius: 12px;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.75);
            }

            .btn {
                border-radius: 10px !important;
            }

            .kpi-row {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                margin-top: 10px;
                margin-bottom: 14px;
            }
            .kpi-card {
                background: linear-gradient(180deg, #ffffff 0%, #f6faff 100%);
                border: 1px solid rgba(213,223,236,.95);
                border-radius: 14px;
                padding: 12px 14px;
                box-shadow: 0 6px 16px rgba(20,66,114,.06);
            }
            .kpi-label {
                font-size: 11px;
                font-weight: 600;
                letter-spacing: .2px;
                text-transform: uppercase;
                color: #6b7b8f;
                margin-bottom: 6px;
            }
            .kpi-value {
                font-size: 1.15rem;
                font-weight: 700;
                line-height: 1.1;
            }

      /* ── instruction steps ── */
      .step-badge {
        display: inline-block; width: 22px; height: 22px; line-height: 22px;
        border-radius: 50%; background: #0067B1; color: white;
        font-size: 11px; font-weight: bold; text-align: center; margin-right: 6px;
      }

      /* ── DataTable header ── */
      .dash-header { background-color: #003087 !important; color: white !important; }

            /* ── slider tooltip layering ── */
            #alpha-container, #alpha-container-updated,
            #alpha-container .rc-slider, #alpha-container-updated .rc-slider {
                position: relative;
                overflow: visible !important;
            }
            .rc-slider-tooltip,
            .rc-slider-tooltip-placement-bottom {
                z-index: 3000 !important;
            }

      /* ── footer bar ── */
      .app-footer {
        background: linear-gradient(135deg, #003087 0%, #0067B1 100%);
        color: rgba(255,255,255,.85); padding: 18px 0; margin-top: 40px;
        font-size: 13px; text-align: center; border-radius: 12px;
      }

      /* ── side-by-side tables ── */
      .tbl-left  { width: 60%; display: inline-block; vertical-align: top;
                   padding-right: 10px; box-sizing: border-box; }
      .tbl-right { width: 40%; display: inline-block; vertical-align: top;
                   box-sizing: border-box; }

      /* ══ MOBILE (≤ 767 px) ══════════════════════════════════════════════ */
      @media (max-width: 767px) {

        /* header: shrink text */
        .app-header-title { font-size: 1.2rem !important; }
        .app-header-sub   { font-size: 0.72rem !important; }
                .app-header-meta  { font-size: 0.66rem !important; }
        .app-header-icon  { font-size: 1.4rem !important; }
        .app-header-wrap  { padding: 18px 16px !important; }
                .app-header-main  { flex-direction: column !important; align-items: stretch !important; width: 100% !important; }
        .app-header-text  { width: 100% !important; flex: 0 0 100% !important; max-width: 100% !important; }
        .app-header-text > div { width: 100% !important; }
                .app-header-logo  { margin-left: 0 !important; margin-top: 10px !important; align-self: flex-end !important; }
                .app-header-logo img { height: 34px !important; }

        /* card body padding */
        .card-body { padding: 12px !important; }

        /* tables: stack vertically */
        .tbl-left, .tbl-right {
          width: 100% !important;
          display: block !important;
          padding-right: 0 !important;
          margin-bottom: 14px;
        }
        /* table containers: auto height on mobile */
        .tbl-left, .tbl-right { height: auto !important; max-height: 320px; overflow-y: auto; }

        /* footer: allow wrapping */
        .app-footer { font-size: 11px; padding: 12px 8px; line-height: 2; }

        /* conformal slider boxes: reduce padding */
        #alpha-container, #alpha-container-updated { padding: 10px !important; }

        /* upload zone: shorter */
        .upload-zone { height: 64px; line-height: 64px; font-size: 0.82rem; }

        .kpi-row { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
        .kpi-card { padding: 10px 12px; }
        .kpi-label { font-size: 8px; }
        .kpi-value { font-size: 0.8rem; }

        /* main panel body padding */
        #main-results-body { padding: 12px !important; }
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""

# Layout: UI improvements only
app.layout = dbc.Container(fluid=True, style={"padding": "0 20px 20px"}, children=[
    # ── Hidden stores ───────────────────────────────────────────────────────
    dcc.Store(id='memory-predictions'),
    dcc.Store(id='memory-calibration'),
    dcc.Store(id='current-patient-index'),
    dcc.Store(id='edited-row'),
    dcc.Store(id='current-order'),
    dcc.Store(id='Current-coefficients'),
    dcc.Store(id='Current-mortality'),
    dcc.Store(id='Current-bounds'),
    dcc.Store(id='update-ran', data=False),
    dcc.Store(id='interval-enabled', data=False),
    dcc.Store(id='interval-enabled-updated', data=False),
    dcc.Store(id='window-width-store', data=1200),

    # ── Header banner ───────────────────────────────────────────────────────
    dbc.Row(dbc.Col(
        html.Div(
            style={
                "background": "linear-gradient(135deg, #003087 0%, #0067B1 100%)",
                "borderRadius": "0 0 16px 16px",
                "padding": "28px 36px",
                "marginBottom": "28px",
                "boxShadow": "0 4px 18px rgba(30,58,92,.25)",
            },
            className="app-header-wrap",
            children=[
                html.Div([
                    # Left: title block
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-heartbeat me-3 app-header-icon",
                                   style={"fontSize": "2rem", "color": "#C8102E"}),
                            html.Span("Sickle Cell Disease",
                                      className="app-header-title",
                                      style={"fontSize": "1.9rem", "fontWeight": "700",
                                             "color": "white", "letterSpacing": ".5px"}),
                            html.Span("Mortality Prediction",
                                      className="app-header-title",
                                      style={"fontSize": "1.9rem", "fontWeight": "300",
                                             "color": "rgba(255,255,255,.85)",
                                             "marginLeft": "0.4rem"}),
                        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
                        html.P(
                            "App v1.3  ·  Model: Multi-Task DeepHit v2.14  ·  Flexible Prediction Intervals  ·  SHAP Explainability",
                            className="app-header-sub",
                            style={"color": "rgba(255,255,255,.65)", "marginTop": "6px",
                                   "fontSize": "0.85rem", "marginBottom": "0"}
                        ),
                        html.P(
                            "Demo · Research only",
                            className="app-header-meta",
                            style={"color": "rgba(255,255,255,.55)", "marginTop": "6px",
                                   "fontSize": "0.72rem", "marginBottom": "0",
                                   "letterSpacing": ".3px", "textTransform": "none"}
                        ),
                    ], className="app-header-text", style={"flex": "1"}),
                    # Right: NHLBI logo
                    html.Div(
                        html.Img(
                            src="https://www.nhlbi.nih.gov/themes/custom/nhlbi/logo.svg",
                            alt="NHLBI Logo",
                            style={
                                "height": "56px",
                                "filter": "brightness(0) invert(1)",  # white version on dark bg
                                "opacity": "0.92",
                            }
                        ),
                        className="app-header-logo",
                        style={"marginLeft": "24px", "flexShrink": "0"}
                    ),
                ], className="app-header-main", style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"}),
            ]
        )
    )),

    # Main content
    dbc.Row([
        # ── Sidebar ─────────────────────────────────────────────────────────
        dbc.Col([

            # Upload patient data
            dbc.Card([
                dbc.CardHeader(
                    html.Div([
                        html.I(className="fas fa-upload me-2"),
                        html.Span("Upload Patient Data", style={"fontWeight": "600"})
                    ]),
                    style={"background": "linear-gradient(90deg,#003087,#0067B1)",
                           "color": "white", "padding": "12px 16px"}
                ),
                dbc.CardBody([
                    dbc.Button([
                        html.I(className="fas fa-file-csv me-2"),
                        "Download Example CSV"
                    ], id="btn-download-example", color="outline-primary",
                       className="w-100 mb-3", size="sm"),
                    dbc.Button([
                        html.I(className="fas fa-bolt me-2"),
                        "Load Example Data"
                    ], id="btn-load-example-data", color="primary",
                       className="w-100 mb-3", size="sm"),
                    dcc.Download(id="download-example-csv"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt me-2",
                                   style={"fontSize": "1.1rem"}),
                            "Click or drag & drop patients CSV"
                        ]),
                        className="upload-zone",
                        multiple=False
                    ),
                    html.Div(id='data-status',
                             children=dbc.Badge("Data not uploaded", color="warning",
                                                className="mt-2 px-3 py-2 w-100 text-start"),
                             className="mt-2"),
                    html.Div(id='upload-data-status', className="mt-1 small text-muted")
                ])
            ], className="app-card shadow-sm mb-3"),

            # Prediction interval methods
            dbc.Card([
                dbc.CardHeader(
                    html.Div([
                        html.I(className="fas fa-route me-2"),
                        html.Span("Prediction Interval Methods", style={"fontWeight": "600"})
                    ]),
                    style={"background": "linear-gradient(90deg,#155724,#28a745)",
                           "color": "white", "padding": "12px 16px"}
                ),
                dbc.CardBody([
                    dbc.Label([
                        html.I(className="fas fa-sliders-h me-2 text-primary"),
                        "Choose interval method"
                    ], style={"fontWeight": "600", "marginBottom": "6px", "fontSize": "0.88rem"}),
                    dbc.RadioItems(
                        id='interval-method-selector',
                        options=[
                            {'label': 'Conformal Prediction (Addiitonal calibration data required)', 'value': 'conformal'},
                            {'label': 'Simultaneous uncertainty band (No calibration data needed)', 'value': 'mc'},
                            {'label': 'No interval (curve only)', 'value': 'none'},
                        ],
                        value='mc',
                        inline=False,
                        style={"fontSize": "0.82rem"}
                    ),
                    html.Small(
                        id='interval-method-hint',
                        children="",
                        className="text-muted", style={"fontSize": "0.75rem"}
                    ),

                    html.Div(id='conformal-upload-panel', children=[
                        html.Hr(style={"borderColor": "#dee2e6", "margin": "10px 0 10px"}),
                        dbc.Alert([
                            html.I(className="fas fa-info-circle me-2"),
                            "Conformal mode requires calibration data. Upload a calibration CSV below."
                        ], color="light", className="py-2 px-3 mb-2",
                           style={"fontSize": "0.82rem", "borderRadius": "8px"}),
                        dbc.Button([
                            html.I(className="fas fa-file-csv me-2"),
                            "Download Calibration Example CSV"
                        ], id="btn-download-calibration-example", color="outline-success",
                           className="w-100 mb-3", size="sm"),
                        dcc.Download(id="download-calibration-example-csv"),
                        dcc.Upload(
                            id='upload-calibration-data',
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt me-2",
                                       style={"fontSize": "1.1rem"}),
                                "Click or drag & drop calibration CSV"
                            ]),
                            className="upload-zone",
                            style={"borderColor": "#28a745", "color": "#28a745",
                                   "background": "#f6fff8"},
                            multiple=False
                        ),
                        html.Div(id='calibration-status',
                                 children=dbc.Badge("Prediction interval not applied",
                                                    color="warning",
                                                    className="mt-2 px-3 py-2 w-100 text-start"),
                                 className="mt-2"),
                        html.Div(id='upload-calibration-status', className="mt-1 small text-muted"),
                    ], className="app-subtle-panel", style={"display": "none", "padding": "12px"}),

                    dcc.Interval(id='mc-progress-interval', interval=500, disabled=True),
                    dcc.Store(id='mc-trigger-store'),
                ])
            ], className="app-card shadow-sm mb-3"),

            # Instructions
            dbc.Card([
                dbc.CardHeader(
                    html.Div([
                        html.I(className="fas fa-info-circle me-2"),
                        html.Span("How to Use", style={"fontWeight": "600"})
                    ]),
                    style={"background": "linear-gradient(90deg,#4a1a6c,#7b2fbf)",
                           "color": "white", "padding": "12px 16px"}
                ),
                dbc.CardBody([
                    *[html.Div([
                        html.Span(str(n), className="step-badge"),
                        html.Span(txt, style={"fontSize": "0.82rem"})
                    ], className="mb-2") for n, txt in [
                        (1, "Download Example CSV for the required format."),
                        (2, "Upload a CSV with 68 numeric columns."),
                        (3, "Choose a prediction interval method (Conformal / Simultaneous uncertainty band / None)."),
                        (4, "If Conformal is selected, upload calibration CSV in the same panel."),
                        (5, "Click a patient row to view results in all sections:"),
                    ]],
                    html.Ul([
                        html.Li("Mortality Risk — 10-year cumulative mortality curve", style={"fontSize": "0.8rem"}),
                        html.Li("Risk Factor Trajectories — 3-year forecast of 12 variables", style={"fontSize": "0.8rem"}),
                        html.Li("SHAP Analysis — feature importance waterfall plot", style={"fontSize": "0.8rem"}),
                    ], style={"paddingLeft": "28px", "marginBottom": "6px"}),
                    html.Div([
                        html.Span("6", className="step-badge"),
                        html.Span("Use 'Edit Features' below SHAP to modify values, then click 'Update Analysis'.", style={"fontSize": "0.82rem"})
                    ]),
                ])
            ], className="app-card shadow-sm mb-3"),

        ], xs=12, md=4, lg=3),

        # ── Main Results Panel ───────────────────────────────────────────────
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(
                    html.Div([
                        html.I(className="fas fa-poll-h me-2"),
                        html.Span("Analysis Results", style={"fontWeight": "600", "fontSize": "1.05rem"})
                    ]),
                    style={"background": "linear-gradient(90deg,#003087,#0067B1)",
                           "color": "white", "padding": "14px 20px"}
                ),
                dbc.CardBody([
                    dbc.Accordion([

                        # ── Section 1: Prediction Table ──────────────────────
                        dbc.AccordionItem([
                            dcc.Loading(id='loading-table', type='dot',
                                        custom_spinner=html.Div([
                                            dbc.Spinner(size="sm", color="primary"),
                                            html.Span("Loading predictions...", style={"color": "#003087", "fontWeight": "600", "fontSize": "0.9rem", "marginLeft": "8px"})
                                        ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "padding": "20px"}),

                                        children=html.Div(id='output')),
                        ],
                            title=html.Span([
                                html.I(className="fas fa-table me-2"),
                                "Prediction Table"
                            ], style={"fontWeight": "600", "color": "#003087"}),
                            item_id="acc-predictions",
                        ),

                        # ── Section 2: Mortality Risk ────────────────────────
                        dbc.AccordionItem([
                            html.Div(
                                id='alpha-container',
                                children=[
                                    dbc.Label([
                                        html.I(className="fas fa-sliders-h me-2 text-primary"),
                                        "Uncertainty Level (1 − α)"
                                    ], style={"fontWeight": "600", "marginBottom": "6px"}),
                                    dcc.Slider(
                                        id='alpha-slider', min=0.01, max=0.5, step=0.01, value=0.05,
                                        marks=ALPHA_MARKS_DESKTOP,
                                        tooltip={"placement": "bottom", "always_visible": False}
                                    )
                                ],
                                    style={"display": "none",
                                        "background": "linear-gradient(180deg, #f7fbff 0%, #edf5ff 100%)", "borderRadius": "12px",
                                        "padding": "10px 12px", "marginBottom": "12px",
                                         "overflow": "visible", "position": "relative", "zIndex": 20,
                                         "border": "1px solid rgba(198,218,242,.95)",
                                         "boxShadow": "0 6px 18px rgba(20,66,114,.06)"}
                            ),
                            html.Div(id='mc-progress-container', children=[
                                html.Div(id='mc-progress-label', style={
                                    "fontSize": "0.8rem", "fontWeight": "600", "color": "#003087", "marginBottom": "4px"
                                }),
                                dbc.Progress(id='mc-progress-bar', value=0, striped=True, animated=True,
                                             style={"height": "16px", "borderRadius": "8px"}, className="mb-1"),
                                html.Div(id='mc-progress-eta', style={"fontSize": "0.75rem", "color": "#888"}),
                            ], style={"marginBottom": "10px", "display": "none"}),
                            html.Div(id='wrap-mortality',
                                     children=dcc.Loading(id='loading-mortality', type='dot',
                                                          custom_spinner=html.Div([
                                                              dbc.Spinner(size="sm", color="primary"),
                                                              html.Span("Computing mortality curve... ETA ~15s", style={"color": "#0067B1", "fontWeight": "600", "fontSize": "0.9rem", "marginLeft": "8px"})
                                                          ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "padding": "20px"}),
                                                          children=html.Div(id='mortality-plot'))),
                            html.Div(
                                id='alpha-container-updated',
                                children=[
                                    dbc.Label([
                                        html.I(className="fas fa-sliders-h me-2 text-primary"),
                                        "Uncertainty Level (1 − α) — Updated Patient"
                                    ], style={"fontWeight": "600", "marginBottom": "6px"}),
                                    dcc.Slider(
                                        id='alpha-slider-updated', min=0.01, max=0.5, step=0.01, value=0.05,
                                        marks=ALPHA_MARKS_DESKTOP,
                                        tooltip={"placement": "bottom", "always_visible": False}
                                    )
                                ],
                                    style={"display": "none",
                                        "background": "linear-gradient(180deg, #fff8f8 0%, #fff0f0 100%)", "borderRadius": "12px",
                                        "padding": "10px 12px", "marginTop": "8px", "marginBottom": "12px",
                                         "overflow": "visible", "position": "relative", "zIndex": 20,
                                         "border": "1px solid rgba(244,210,210,.95)",
                                         "boxShadow": "0 6px 18px rgba(122,40,40,.05)"}
                            ),
                            html.Div(
                                id='wrap-loading-update-mortality',
                                style={"display": "none"},
                                children=dcc.Loading(id='loading-update_mortality', type='dot',
                                                     custom_spinner=html.Div([
                                                         dbc.Spinner(size="sm", color="danger"),
                                                         html.Span("Updating mortality analysis... ETA ~15s", style={"color": "#C8102E", "fontWeight": "600", "fontSize": "0.9rem", "marginLeft": "8px"})
                                                     ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "padding": "20px"}),
                                                     children=html.Div(id='mortality-plot-updated'))
                            ),
                        ],
                            title=html.Span([
                                html.I(className="fas fa-heartbeat me-2"),
                                "Mortality Risk"
                            ], style={"fontWeight": "600", "color": "#0067B1"}),
                            item_id="acc-mortality",
                        ),

                        # ── Section 3: Risk Factor Trajectories ─────────────
                        dbc.AccordionItem([
                            html.Div(id='traj-progress-container', children=[
                                html.Div(id='traj-progress-label', style={
                                    "fontSize": "0.8rem", "fontWeight": "600", "color": "#155724", "marginBottom": "4px"
                                }),
                                dbc.Progress(id='traj-progress-bar', value=0, striped=True, animated=True,
                                             style={"height": "16px", "borderRadius": "8px"}, className="mb-1"),
                                html.Div(id='traj-progress-eta', style={"fontSize": "0.75rem", "color": "#888"}),
                            ], style={"marginBottom": "10px", "display": "none"}),
                            html.Div(id='wrap-trajectory',
                                     children=dcc.Loading(id='loading-trajectory', type='dot',
                                                          custom_spinner=html.Div([
                                                              dbc.Spinner(size="sm", color="success"),
                                                              html.Span("Generating risk factor trajectories... ETA ~15s", style={"color": "#155724", "fontWeight": "600", "fontSize": "0.9rem", "marginLeft": "8px"})
                                                          ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "padding": "20px"}),
                                                          children=html.Div(id='trajectory-plot'))),
                            html.Div(
                                id='wrap-loading-update-trajectory',
                                style={"display": "none"},
                                children=dcc.Loading(id='loading-update_trajectory', type='dot',
                                                     custom_spinner=html.Div([
                                                         dbc.Spinner(size="sm", color="success"),
                                                         html.Span("Updating risk factor trajectories... ETA ~15s", style={"color": "#155724", "fontWeight": "600", "fontSize": "0.9rem", "marginLeft": "8px"})
                                                     ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "padding": "20px"}),
                                                     children=html.Div(id='trajectory-plot-updated'))
                            ),
                        ],
                            title=html.Span([
                                html.I(className="fas fa-chart-line me-2"),
                                "Risk Factor Trajectories"
                            ], style={"fontWeight": "600", "color": "#155724"}),
                            item_id="acc-trajectories",
                        ),

                        # ── Section 4: SHAP Analysis ─────────────────────────
                        dbc.AccordionItem([
                            # 1. Original SHAP
                            dcc.Loading(id='loading-shap', type='dot',
                                        custom_spinner=html.Div([
                                            dbc.Spinner(size="sm", color="warning"),
                                            html.Span("Computing SHAP feature importance... ETA ~15s", style={"color": "#856404", "fontWeight": "600", "fontSize": "0.9rem", "marginLeft": "8px"})
                                        ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "padding": "20px"}),
                                        children=html.Div(id='shap-plot')),

                            # 2. Updated SHAP (appears after Update Analysis)
                            html.Div(
                                id='wrap-loading-update-shap',
                                style={"display": "none"},
                                children=dcc.Loading(id='loading-update', type='dot',
                                                     custom_spinner=html.Div([
                                                         dbc.Spinner(size="sm", color="warning"),
                                                         html.Span("Updating SHAP analysis... ETA ~15s", style={"color": "#856404", "fontWeight": "600", "fontSize": "0.9rem", "marginLeft": "8px"})
                                                     ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "padding": "20px"}),
                                                     children=html.Div(id='shap-plot-updated', className="mt-3"))
                            ),

                            # hint — visible above the button
                            html.Div([
                                html.I(className="fas fa-edit me-1 text-primary"),
                                html.Span(
                                    "Click 'Edit Features' below to modify this patient's values and see how the predictions change.",
                                    style={"fontSize": "0.82rem", "color": "#6c757d"})
                            ], id='editor-hint',
                               style={"display": "none", "marginTop": "12px", "marginBottom": "4px"}),

                            # 3. "Edit Features" toggle button
                            dbc.Button([
                                html.I(className="fas fa-sliders-h me-2"),
                                "Edit Features"
                            ], id='btn-open-editor', color="primary",
                               className="px-4",
                               style={"display": "none", "borderRadius": "8px",
                                      "fontWeight": "600",
                                      "background": "linear-gradient(90deg,#003087,#0067B1)",
                                      "border": "none"}),

                            # 4. Inline collapsible editor panel
                            dbc.Collapse(
                                html.Div([
                                    dbc.Alert([
                                        html.I(className="fas fa-info-circle me-2 text-primary"),
                                        "Search and edit feature values below, then click 'Update Analysis' to see the revised predictions."
                                    ], color="light", className="py-2 px-3 border mb-2",
                                       style={"fontSize": "0.85rem", "borderRadius": "8px",
                                              "borderColor": "#bee2ff !important"}),
                                    dbc.InputGroup([
                                        dbc.InputGroupText(
                                            html.I(className="fas fa-search"),
                                            style={"background": "#eef6ff",
                                                   "borderColor": "#0067B1"}),
                                        dbc.Input(
                                            id='feature-search',
                                            placeholder="Search feature name...",
                                            debounce=False,
                                            style={"borderColor": "#0067B1",
                                                   "fontSize": "0.9rem"}),
                                        dbc.Button(
                                            html.I(className="fas fa-times"),
                                            id='feature-search-clear',
                                            color="outline-secondary", size="sm",
                                            style={"borderColor": "#dee2e6"}),
                                    ], className="mb-2"),
                                    html.Div(id='feature-editor',
                                             style={"overflowX": "auto", "width": "100%"}),
                                    html.Hr(style={"borderColor": "#dee2e6"}),
                                    dbc.Button([
                                        html.I(className="fas fa-sync-alt me-2"),
                                        "Update Analysis"
                                    ], id='update-shap-button', color="primary",
                                       className="px-4",
                                       style={"borderRadius": "8px", "fontWeight": "600",
                                              "background": "linear-gradient(90deg,#003087,#0067B1)",
                                              "border": "none"}),
                                ], style={"background": "#f8faff", "borderRadius": "10px",
                                          "padding": "16px", "marginTop": "12px",
                                          "border": "1px solid #dee2e6"}),
                                id="feature-offcanvas",
                                is_open=False,
                            ),
                        ],
                            title=html.Span([
                                html.I(className="fas fa-water me-2"),
                                "SHAP Analysis"
                            ], style={"fontWeight": "600", "color": "#4a1a6c"}),
                            item_id="acc-shap",
                        ),

                    ],
                        id="result-accordion",
                        active_item=["acc-predictions", "acc-mortality",
                                     "acc-trajectories", "acc-shap"],
                        always_open=True,
                        flush=False,
                        style={"borderRadius": "10px", "overflow": "hidden"},
                    ),
                ], id="main-results-body", style={"padding": "18px"})
            ], className="app-card app-results-card shadow-sm"), xs=12, md=8, lg=9
        )
    ]),

    # ── Update Toast notification ────────────────────────────────────────────
    dbc.Toast(
        [html.I(className="fas fa-check-circle me-2 text-success"),
         "Analysis updated — see results below."],
        id="update-toast",
        header="Results Ready",
        is_open=False,
        dismissable=True,
        duration=4000,
        style={"position": "fixed", "bottom": 24, "right": 24,
               "zIndex": 9999, "minWidth": "260px",
               "boxShadow": "0 4px 16px rgba(0,0,0,.18)",
               "borderLeft": "4px solid #0067B1"},
    ),


    # ── Scroll anchor (hidden, used by clientside callback) ──────────────────
    html.Div(id='scroll-dummy', style={"display": "none"}),

    # ── Footer ──────────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(
        html.Div(className="app-footer", children=[
            html.Span([
                html.I(className="fas fa-microchip me-2"),
                "Model: Multi-Task DeepHit v2.14"
            ]),
            html.Span("  ·  ", style={"opacity": ".5"}),
            html.Span([
                html.I(className="fas fa-code me-2"),
                "App by Gefei Lin"
            ]),
            html.Span("  ·  ", style={"opacity": ".5"}),
            html.Span([
                html.I(className="fas fa-tag me-2"),
                "v1.3"
            ]),
        ])
    ))
            
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
    Output("download-calibration-example-csv", "data"),
    Input("btn-download-calibration-example", "n_clicks"),
    prevent_initial_call=True
)
def download_calibration_example(n_clicks):

    return dcc.send_data_frame(
        calibration_example.to_csv,
        "example_scd_calibration_data.csv",
        index=False
    )

@app.callback(
    Output('memory-calibration', 'data'),
    Output('upload-calibration-status', 'children'),
    Output('calibration-status', 'children'),
    Input('upload-calibration-data', 'contents'),
    State('upload-calibration-data', 'filename')
)

def preprocess_calibration(contents, filename):
    if contents is None:
        return dash.no_update
    

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df_calibration = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    _calib_err_status = dbc.Badge(
        [html.I(className="fas fa-times-circle me-1"), "Prediction interval not applied"],
        color="danger", className="mt-2 px-3 py-2 w-100 text-start"
    )
    def _calib_err(msg):
        return (dash.no_update,
                dbc.Alert(msg, color="danger", className="py-2 px-3 mt-1",
                          style={"fontSize": "0.83rem", "borderRadius": "6px"}),
                _calib_err_status)

    if 'Event_status' not in df_calibration.columns or 'Event_time' not in df_calibration.columns:
        return _calib_err("Error: Calibration data must contain 'Event_status' and 'Event_time' columns.")

    max_event_time = df_calibration['Event_time'].max()
    if max_event_time < 16:
        return _calib_err("Error: 'Event_time' must be provided in units of days.")

    df_calibration_no_outcome = df_calibration.drop(columns=['Event_status','Event_time'], errors='ignore')
    
    calibration_example_no_outcome = calibration_example.drop(columns=['Event_status','Event_time'], errors='ignore')
    calibration_example_columns = calibration_example_no_outcome.columns.tolist()
    if not all(col in calibration_example_columns for col in df_calibration_no_outcome.columns):
        return _calib_err("Error: Calibration data columns do not match the example data.")
    if not all(df_calibration_no_outcome.columns[i] == calibration_example_no_outcome.columns[i] for i in range(len(df_calibration_no_outcome.columns))):
        return _calib_err("Error: Calibration data columns are not in the correct order.")

    mask = ~np.isnan(df_calibration_no_outcome)
    #mask = np.load("./mask_miss.npy", allow_pickle=True)
    df_scaled = (df_calibration_no_outcome - x_mean) / x_std
    df_scaled = df_scaled.fillna(0)

    data_calibration_scaled = np.asarray(df_scaled, dtype=np.float32)
    data_calibration_scaled_with_mask = np.concatenate((data_calibration_scaled, mask), axis=1)
    Event_time = np.asarray(df_calibration['Event_time'], dtype=np.float64) / 365.25
    Event_time = np.ceil(Event_time)
    Event_time[Event_time >= 15] = 16
    Event_time = np.clip(Event_time, 1, 16).astype(np.int64)

    Event_status = np.asarray(df_calibration['Event_status'], dtype=np.int64)
    Event_status = np.where(Event_status > 0, 1, 0).astype(np.int64)

    print("Calibration data processed successfully.")
    

    return (
        {'data_calibration_scaled_with_mask': data_calibration_scaled_with_mask.tolist(),
         'Event_time_calibration': Event_time.tolist(),
         'Event_status_calibration': Event_status.tolist()},
        dbc.Badge([html.I(className="fas fa-check-circle me-1"), f"{filename} uploaded"],
                  color="success", className="px-3 py-2"),
        dbc.Badge([html.I(className="fas fa-shield-alt me-1"), "Prediction interval enabled"],
                  color="success", className="mt-2 px-3 py-2 w-100 text-start"),
    )

@app.callback(
    Output('interval-method-selector', 'options'),
    Output('interval-method-selector', 'value'),
    Output('interval-method-hint', 'children'),
    Input('memory-calibration', 'data'),
    State('interval-method-selector', 'value'),
)
def update_interval_method_selector(calib_data, current_method):
    has_calib = calib_data is not None
    options = [
        {'label': 'Conformal Prediction (Addiitonal calibration data required)', 'value': 'conformal'},
        {'label': 'Simultaneous uncertainty band (No calibration data needed)', 'value': 'mc'},
        {'label': 'No interval (curve only)', 'value': 'none'},
    ]

    next_method = current_method if current_method in ['conformal', 'mc', 'none'] else 'mc'

    hint = ""
    return options, next_method, hint


@app.callback(
    Output('conformal-upload-panel', 'style'),
    Input('interval-method-selector', 'value'),
)
def toggle_conformal_upload_panel(interval_method):
    if interval_method == 'conformal':
        return {"display": "block", "marginTop": "4px"}
    return {"display": "none"}


# Clientside callback: signal that MC might start (patient click or update button while MC enabled)
app.clientside_callback(
    """
    function(active_cell, n_clicks, interval_method) {
        if (interval_method === 'mc') {
            return Date.now();  // unique trigger value
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('mc-trigger-store', 'data'),
    Input('x-table', 'active_cell'),
    Input('update-shap-button', 'n_clicks'),
    State('interval-method-selector', 'value'),
    prevent_initial_call=True
)


@app.callback(
    Output('mc-progress-container', 'style'),
    Output('mc-progress-bar', 'value'),
    Output('mc-progress-bar', 'label'),
    Output('mc-progress-label', 'children'),
    Output('mc-progress-eta', 'children'),
    Output('traj-progress-container', 'style'),
    Output('traj-progress-bar', 'value'),
    Output('traj-progress-bar', 'label'),
    Output('traj-progress-label', 'children'),
    Output('traj-progress-eta', 'children'),
    Output('mc-progress-interval', 'disabled'),
    Input('mc-progress-interval', 'n_intervals'),
    Input('mc-trigger-store', 'data'),
    Input('x-table', 'active_cell'),
    Input('update-shap-button', 'n_clicks'),
    State('interval-method-selector', 'value'),
    prevent_initial_call=True
)
def update_mc_progress(n_intervals, trigger, active_cell, n_update_clicks, interval_method):
    from dash import ctx
    triggered_id = ctx.triggered_id
    selected_method = interval_method or 'mc'

    # Robust start signal (mobile-friendly): directly react to row click/update click
    # when MC mode is active.
    if triggered_id in ['x-table', 'update-shap-button'] and selected_method == 'mc':
        mc_progress['trigger_ts'] = time.time() * 1000.0
        return (
            {"marginTop": "8px", "display": "none"},
            0, '', '', '',
            {"marginTop": "8px", "display": "none"},
            0, '', '', '',
            False
        )

    # If triggered by mc-trigger-store, enable interval and show progress
    if triggered_id == 'mc-trigger-store':
        return (
            {"marginTop": "8px", "display": "none"},
            0, '', '', '',
            {"marginTop": "8px", "display": "none"},
            0, '', '', '',
            False
        )
    # Polled by interval
    if not mc_progress['running']:
        # Avoid race condition on update: keep polling briefly after trigger
        # so progress bar does not disappear before long MC jobs actually start.
        try:
            now_ms = time.time() * 1000.0
            trigger_ts = float(trigger) if trigger is not None else 0.0
            server_ts = float(mc_progress.get('trigger_ts', 0.0))
            recent_client = trigger_ts > 0 and (now_ms - trigger_ts < 15000)
            recent_server = server_ts > 0 and (now_ms - server_ts < 15000)
            is_recent_trigger = bool(recent_client or recent_server)
        except Exception:
            is_recent_trigger = False
        if is_recent_trigger:
            return (
                {"marginTop": "8px", "display": "none"}, 0, '', '', '',
                {"marginTop": "8px", "display": "none"}, 0, '', '', '',
                False
            )
        return (
            {"marginTop": "8px", "display": "none"}, 0, '', '', '',
            {"marginTop": "8px", "display": "none"}, 0, '', '', '',
            True
        )  # disable interval
    pct = int(mc_progress['current'] / mc_progress['total'] * 100) if mc_progress['total'] > 0 else 0
    label_text = f"{mc_progress['current']}/{mc_progress['total']}"
    title = f"Simultaneous uncertainty band — {mc_progress['label']}" if mc_progress['label'] else "Simultaneous uncertainty band"
    eta = f"ETA: {mc_progress['eta']}" if mc_progress['eta'] else ''
    return (
        {"marginTop": "8px", "display": "block"},
        pct,
        label_text,
        title,
        eta,
        {"marginTop": "8px", "display": "block"},
        pct,
        label_text,
        title,
        eta,
        False  # keep interval running
    )


@app.callback(
    Output('output', 'children'),
    Output('memory-predictions', 'data'),
    Output('data-status', 'children'),
    Output('upload-data-status', 'children'),
    Input('upload-data', 'contents'),
    Input('btn-load-example-data', 'n_clicks'),
    State('upload-data', 'filename')
)
def predict(contents, n_load_example, filename):
    from dash import ctx
    trigger = ctx.triggered_id

    if trigger == 'btn-load-example-data':
        df = X_example.copy()
        filename = "example_scd_data.csv"
    elif contents is None:
        return html.Div("Please upload a CSV file."), dash.no_update,dash.no_update,dash.no_update
    else:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    # round all values to 5 decimal places

    _err_status = dbc.Badge([html.I(className="fas fa-times-circle me-1"), "Data not uploaded"],
                            color="danger", className="mt-2 px-3 py-2 w-100 text-start")
    if not all(col in X_example.columns for col in df.columns):
        return dash.no_update, dash.no_update, _err_status, \
               dbc.Alert("Error: columns do not match the example data.", color="danger",
                         className="py-2 px-3 mt-1", style={"fontSize": "0.83rem", "borderRadius": "6px"})

    if not all(df.columns[i] == X_example.columns[i] for i in range(len(df.columns))):
        return dash.no_update, dash.no_update, _err_status, \
               dbc.Alert("Error: columns are not in the correct order.", color="danger",
                         className="py-2 px-3 mt-1", style={"fontSize": "0.83rem", "borderRadius": "6px"})

    if df.shape[1] != 68:
        return dash.no_update, dash.no_update, _err_status, \
               dbc.Alert("Please ensure the CSV contains exactly 68 numeric columns.", color="danger",
                         className="py-2 px-3 mt-1", style={"fontSize": "0.83rem", "borderRadius": "6px"})

    mask = ~np.isnan(df)
    #mask = np.load("./mask_miss.npy", allow_pickle=True)
    #mask = pd.DataFrame(mask, columns=df.columns, dtype='float32')
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
    _tbl_header = {
        'backgroundColor': '#003087', 'color': 'white',
        'fontWeight': '600', 'textAlign': 'center', 'fontSize': '12px'
    }
    _tbl_cell = {
        'padding': '7px 10px', 'minWidth': '80px',
        'whiteSpace': 'nowrap', 'fontSize': '12px',
        'border': '1px solid #dee2e6'
    }
    _tbl_data_cond = [
        {'if': {'row_index': 'odd'}, 'backgroundColor': '#f2f6fc'},
        {'if': {'state': 'selected'},
         'backgroundColor': '#d0e8ff', 'border': '1px solid #0067B1'},
        {'if': {'state': 'active'},
         'backgroundColor': '#cce0ff', 'border': '1px solid #0067B1'},
    ]

    return html.Div([
        html.Div([
            html.I(className="fas fa-table me-2 text-primary"),
            html.Span(f"Predictions from: {filename}",
                      style={"fontWeight": "600", "fontSize": "1rem"})
        ], style={"marginBottom": "12px"}),

        html.Div([
            # Feature table
            html.Div([
                html.P([html.I(className="fas fa-user-injured me-1 text-secondary"),
                        " Patient Features"],
                       style={"fontSize": "0.85rem", "fontWeight": "600",
                              "color": "#6c757d", "marginBottom": "6px"}),
                dash_table.DataTable(
                    id='x-table',
                    data=df_features.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in df_features.columns],
                    page_action='none',
                    style_table={'height': 'auto', 'overflowX': 'auto',
                                 'borderRadius': '8px',
                                 'boxShadow': '0 1px 6px rgba(0,0,0,.08)'},
                    style_header=_tbl_header,
                    style_cell=_tbl_cell,
                    style_data_conditional=_tbl_data_cond,
                )
            ], className="tbl-left",
               style={'height': f'{min(44 + len(df) * 36, 600)}px',
                      'overflowY': 'auto', 'overflowX': 'auto'}),

            # Predictions table
            html.Div([
                html.P([html.I(className="fas fa-chart-bar me-1 text-primary"),
                        " Cumulative Mortality"],
                       style={"fontSize": "0.85rem", "fontWeight": "600",
                              "color": "#6c757d", "marginBottom": "6px"}),
                dash_table.DataTable(
                    data=pred_df.to_dict('records'),
                    columns=[
                        {'name': col, 'id': col} if col == 'Patient ID'
                        else {'name': col, 'id': col, 'type': 'numeric',
                              'format': Format(precision=5, scheme=Scheme.fixed)}
                        for col in pred_df.columns
                    ],
                    page_action='none',
                    style_table={'height': 'auto', 'overflowX': 'auto',
                                 'borderRadius': '8px',
                                 'boxShadow': '0 1px 6px rgba(0,0,0,.08)'},
                    style_header=_tbl_header,
                    style_cell={**_tbl_cell, 'minWidth': '100px'},
                    style_data_conditional=_tbl_data_cond,
                )
            ], className="tbl-right",
               style={'height': f'{min(44 + len(df) * 36, 600)}px',
                      'overflowY': 'auto', 'overflowX': 'auto'})
        ]),

        html.Br(),
        dbc.Button([
            html.I(className="fas fa-download me-2"),
            "Download Mortality Table"
        ], id='download-mortality-button', color="outline-primary",
           size="sm", className="mt-2", style={"display": "none",
                                               "borderRadius": "6px"}),
        dcc.Download(id="download-mortality-table-csv"),
        html.Br(), html.Br(),
        dbc.Alert([
            html.I(className="fas fa-mouse-pointer me-2"),
            "Click on a patient row to populate the Mortality Risk, Risk Factor Trajectories, and SHAP Analysis sections."
        ], color="info", className="py-2 px-3",
           style={"fontSize": "0.85rem", "borderRadius": "8px"}),
        html.Hr(style={"borderColor": "#dee2e6"})
    ]), {'df_features': df.to_dict('records'), 'pred_df': pred_df.to_dict('records'), 'scaled_df': df_scaled.to_dict('records'), 'mask': mask.values.tolist(), 'coefficients': coefficients_np}, \
        dbc.Badge([html.I(className="fas fa-check-circle me-1"), "Patients data uploaded"],
                  color="success", className="mt-2 px-3 py-2 w-100 text-start"), \
        dbc.Badge([html.I(className="fas fa-check-circle me-1"), f"{filename} uploaded"],
                  color="success", className="px-3 py-2")

@app.callback(
    Output('download-mortality-button', 'style'),
    Input('memory-predictions', 'data'),
    prevent_initial_call=True
)
def toggle_download_button(memory):
    if memory is not None:
        return {"display": "inline-block", "marginTop": "10px"}
    return {"display": "none"}

@app.callback(
    Output('update-ran', 'data'),
    Input('update-shap-button', 'n_clicks'),
    Input('x-table', 'active_cell'),
    prevent_initial_call=True
)
def track_update_ran(n_clicks, active_cell):
    from dash import ctx
    if ctx.triggered_id == 'update-shap-button' and n_clicks:
        return True
    return False   # reset when new patient row is clicked


@app.callback(
    Output('alpha-container', 'style'),
    Input('memory-calibration', 'data'),
    Input('interval-method-selector', 'value'),
    Input('x-table', 'active_cell'),
    Input('update-shap-button', 'n_clicks'),
    State('update-ran', 'data'),
    State('memory-predictions', 'data'),
)
def toggle_slider_visibility(calib_data, interval_method, active_cell, n_clicks, update_ran, memory_predictions):
    from dash import ctx
    # Hide if update was just clicked, or if update already ran and calib is being reloaded
    if ctx.triggered_id == 'update-shap-button' and n_clicks:
        return {"display": "none"}
    if update_ran:
        return {"display": "none"}
    if interval_method not in ['conformal', 'mc']:
        return {"display": "none"}
    # Show if calibration loaded and there's any prediction result
    has_result = active_cell is not None or memory_predictions is not None
    if interval_method == 'conformal' and calib_data is None:
        return {"display": "none"}
    if has_result:
        return {
            "display": "block",
            "marginTop": "10px",
            "marginBottom": "16px",
            "background": "#eef6ff",
            "borderRadius": "10px",
            "padding": "10px 12px",
            "border": "1px solid rgba(198,218,242,.95)",
            "boxShadow": "0 6px 18px rgba(20,66,114,.06)",
        }
    return {"display": "none"}

@app.callback(
    Output('download-mortality-table-csv', 'data'),
    Input('download-mortality-button', 'n_clicks'),
    State('memory-predictions', 'data'),
    prevent_initial_call=True
)
def download_mortality_table(n_clicks, memory):
    if memory is None:
        raise dash.exceptions.PreventUpdate

    pred_df = pd.DataFrame(memory['pred_df'])
    
    return dcc.send_data_frame(
        pred_df.to_csv,
        "predicted_mortality.csv",
        index=False
    )

@app.callback(
    Output('mortality-plot', 'children'),
    Output('Current-mortality', 'data'),
    Output('Current-bounds','data'),
    Input('x-table', 'active_cell'),
    Input('alpha-slider', 'value'),
    Input('memory-calibration', 'data'),
    State('interval-method-selector', 'value'),
    State('memory-predictions', 'data'),
    State('window-width-store', 'data'),
    prevent_initial_call=True
)
def plot_mortality(active_cell, alpha_value, memory_calibration, interval_method, memory, window_width):
    if not memory or not active_cell:
        raise dash.exceptions.PreventUpdate

    alpha_value = 0.05 if alpha_value is None else float(alpha_value)
    alpha_value = float(np.clip(alpha_value, 0.01, 0.5))

    selected_method = interval_method or ('conformal' if memory_calibration is not None else 'mc')
    plot_conformal = bool(selected_method == 'conformal' and memory_calibration is not None)
    plot_mc = bool(selected_method == 'mc')

    i = active_cell['row']
    mortality = np.array([
        [v for k, v in row.items() if k != 'Patient ID']
        for row in memory['pred_df']
    ])

    display_horizon = DEFAULT_MORTALITY_DISPLAY_HORIZON
    y = mortality[i, :display_horizon]
    x = list(range(1, len(y)+1))
    is_mobile = window_width is not None and window_width < 768
    legend_font_size = 8 if is_mobile else 10
    legend_entry_width = 120 if is_mobile else None
    yaxis_title_size = 10 if is_mobile else 12
    mortality_height = get_mobile_43_height(window_width) if is_mobile else 500
    margin_left = 38 if is_mobile else 60
    margin_right = 8 if is_mobile else 20

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines+markers', name=f'Patient {i + 1}',
        line=dict(color="#0067B1", width=2.5),
        marker=dict(size=7, color="#0067B1", line=dict(color="white", width=1.5))
    ))

    has_interval = False
    lower = None
    upper = None

    if plot_conformal:
        data_calibration_scaled_with_mask = np.asarray(memory_calibration['data_calibration_scaled_with_mask'], dtype=np.float32)

        Event_time_calibration = np.asarray(memory_calibration['Event_time_calibration']).reshape(-1).astype(np.int64)

        Event_status_calibration = np.asarray(memory_calibration['Event_status_calibration']).reshape(-1).astype(np.int64)
        df_scaled = pd.DataFrame(memory['scaled_df'])
        mask = pd.DataFrame(memory['mask'],dtype='float32')
        X_and_mask_eval = np.concatenate((df_scaled.values, mask.values), axis=1)

        X_and_mask_test = X_and_mask_eval[i].reshape(1, -1)

        result = conformal_mortality_prediction(
            model_original=model_copy,
            X_and_mask=data_calibration_scaled_with_mask,
            E_train=Event_status_calibration,
            T_train=Event_time_calibration,
            X_and_mask_test=X_and_mask_test,
            max_horizon=display_horizon,
            alpha=alpha_value
        )
        lower = result['corrected_lower_bounds']
        upper = result['corrected_upper_bounds']
        has_interval = True

        # Add shaded confidence interval
        fig.add_traces([
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill='toself',
                fillcolor='rgba(0, 123, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f"Conf int {int((1 - alpha_value) * 100)}%"
            ),
            go.Scatter(x=x, y=upper, line=dict(dash='dash', color='rgba(0, 123, 255, 0.2)'), mode='lines', showlegend=False, hoverinfo='skip', name='Upper Bound'),
            go.Scatter(x=x, y=lower, line=dict(dash='dash', color='rgba(0, 123, 255, 0.2)'), mode='lines', showlegend=False, hoverinfo='skip', name='Lower Bound')
        ])

    elif plot_mc:
        df_scaled = pd.DataFrame(memory['scaled_df'])
        mask_df = pd.DataFrame(memory['mask'], dtype='float32')
        # Only compute simultaneous uncertainty band for the selected patient
        single_input = paddle.to_tensor(df_scaled.values[i:i+1].astype('float32'))
        single_mask = paddle.to_tensor(mask_df.values[i:i+1].astype('float32'))
        mc_lo, mc_hi, mc_mean = mc_dropout_predict(
            model_copy,
            single_input,
            single_mask,
            alpha=alpha_value,
            label='Mortality Curve'
        )
        lower = np.clip(mc_lo[0, :display_horizon], 0, 1).tolist()
        upper = np.clip(mc_hi[0, :display_horizon], 0, 1).tolist()
        mc_mean_vals = mc_mean[0, :display_horizon].tolist()
        has_interval = True

        fig.add_traces([
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill='toself',
                fillcolor='rgba(40, 167, 69, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f'Band {int((1-alpha_value)*100)}%'
            ),
            go.Scatter(x=x, y=upper, line=dict(dash='dash', color='rgba(40, 167, 69, 0.4)'), mode='lines', showlegend=False, hoverinfo='skip', name='Band Upper'),
            go.Scatter(x=x, y=lower, line=dict(dash='dash', color='rgba(40, 167, 69, 0.4)'), mode='lines', showlegend=False, hoverinfo='skip', name='Band Lower'),
            go.Scatter(x=x, y=mc_mean_vals, mode='lines+markers', name='Mean',
                       line=dict(color='rgba(40, 167, 69, 0.8)', width=2, dash='dot'),
                       marker=dict(size=5, color='rgba(40, 167, 69, 0.8)'),
                       visible='legendonly')
        ])

    if lower is not None and upper is not None:
        fig.data[0].customdata = np.column_stack([
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64)
        ])
        fig.data[0].hovertemplate = "%{y:.2f} [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra>%{fullData.name}</extra>"
    else:
        fig.data[0].hovertemplate = "%{y:.2f}<extra>%{fullData.name}</extra>"

    fig.update_layout(
        title=dict(text=f'Cumulative Mortality Risk — Patient {i + 1}',
                   font=dict(size=15, color="#003087", family="Segoe UI, Arial")),
        xaxis_title='Year',
        template='plotly_white',
        xaxis=dict(tickmode='linear', dtick=1, showgrid=True,
                   gridcolor="#e8eef4", gridwidth=1),
        yaxis=dict(range=[-0.01, 1.01], showgrid=True,
                   gridcolor="#e8eef4", gridwidth=1,
                   title=dict(text='Cumulative Mortality', font=dict(size=yaxis_title_size), standoff=2 if is_mobile else 8),
                   tickformat=".0%"),
        plot_bgcolor="rgba(248,251,255,0.9)", paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#444"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.60)", bordercolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(size=legend_font_size), itemsizing="constant",
                    entrywidth=legend_entry_width, entrywidthmode='pixels'),
        margin=dict(t=105, b=70, l=margin_left, r=margin_right),
        hovermode="x unified",
        height=mortality_height,
    )
    kpi_cards = create_kpi_cards(y, lower, upper)
    return [kpi_cards, dcc.Graph(figure=fig, config={'displayModeBar': False, 'responsive': True}, style={"width": "100%"}), html.Hr()], y.tolist(), {'lower_bounds': lower, 'upper_bounds': upper} if has_interval else None


@app.callback(
    Output('interval-enabled', 'data'),
    Input('alpha-slider', 'value'),
    Input('memory-calibration', 'data'),
    Input('x-table', 'active_cell'),
    prevent_initial_call=True
)
def control_interval_visibility(alpha_value, memory_calibration, active_cell):
    from dash import ctx
    trigger = ctx.triggered_id
    if trigger == 'alpha-slider' and memory_calibration is not None and active_cell is not None:
        return True
    if trigger in ['memory-calibration', 'x-table']:
        return False
    raise dash.exceptions.PreventUpdate

@app.callback(
    Output('trajectory-plot', 'children'),
    Input('x-table', 'active_cell'),
    Input('alpha-slider', 'value'),
    State('memory-predictions', 'data'),
    State('window-width-store', 'data'),
    State('interval-method-selector', 'value'),
    prevent_initial_call=True
)
def plot_trajectory(active_cell, alpha_value, memory, window_width, interval_method):
    if not active_cell or not memory:
        raise dash.exceptions.PreventUpdate

    i = active_cell['row']
    coefficients_data = memory['coefficients']
    coeffcients = [paddle.to_tensor(np.array(c), dtype='float32') for c in coefficients_data]

    original_band = None
    alpha_value = 0.05 if alpha_value is None else float(alpha_value)
    alpha_value = float(np.clip(alpha_value, 0.01, 0.5))
    if (interval_method or 'mc') == 'mc':
        df_scaled = pd.DataFrame(memory['scaled_df'])
        mask_df = pd.DataFrame(memory['mask'], dtype='float32')
        single_input = paddle.to_tensor(df_scaled.values[i:i+1].astype('float32'))
        single_mask = paddle.to_tensor(mask_df.values[i:i+1].astype('float32'))
        lo_t, up_t, _ = mc_dropout_trajectory_band(
            model_copy,
            single_input,
            single_mask,
            person_id=i,
            alpha=alpha_value,
            n_samples=1000,
            label='Trajectory (original)',
        )
        original_band = {'lower': lo_t, 'upper': up_t}

    fig = create_trajectory_plot(
        i,
        coeffcients,
        window_width=window_width,
        original_band=original_band,
        alpha=alpha_value,
    )
    return [dcc.Graph(figure=fig, config={'displayModeBar': False, 'responsive': True}, style={"width": "100%"}), html.Hr()]

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
    _edit_tbl_hdr = {
        'backgroundColor': '#003087', 'color': 'white',
        'fontWeight': '600', 'textAlign': 'center', 'fontSize': '12px'
    }
    table = dash_table.DataTable(
        id='editable-table',
        columns=[{'name': k, 'id': k, 'editable': True} for k in row_data],
        data=[row_data],
        style_table={'overflowX': 'auto', 'borderRadius': '8px',
                     'boxShadow': '0 1px 6px rgba(0,0,0,.08)'},
        style_header=_edit_tbl_hdr,
        style_cell={'padding': '7px 10px', 'minWidth': '80px',
                    'whiteSpace': 'normal', 'fontSize': '12px',
                    'border': '1px solid #dee2e6'},
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f2f6fc'},
        ],
    )

    return (
        html.Div([
            html.Div([
                html.I(className="fas fa-water me-2 text-primary"),
                html.Span(f"SHAP Waterfall — Patient {i + 1}",
                          style={"fontWeight": "600", "fontSize": "1rem"})
            ], style={"marginBottom": "10px"}),
            html.Img(src=img, style={
                'maxWidth': '100%', 'height': 'auto',
                'border': '1px solid #dee2e6', 'borderRadius': '8px',
                'boxShadow': '0 2px 8px rgba(0,0,0,.08)'
            }),
            html.Hr(style={"borderColor": "#dee2e6"})
        ]),
        table,
        i,
        row_data,
        current_order
    )

@app.callback(
    Output('update-shap-button', 'style'),
    Input('current-patient-index', 'data'),
)
def toggle_update_button_visibility(index):
    # Button lives in offcanvas — always fully visible there
    return {"borderRadius": "8px", "fontWeight": "600",
            "background": "linear-gradient(90deg,#003087,#0067B1)", "border": "none"}


@app.callback(
    Output('btn-open-editor', 'style'),
    Output('editor-hint', 'style'),
    Input('current-patient-index', 'data'),
)
def show_edit_button(index):
    if index is not None:
        return ({"display": "inline-block", "borderRadius": "8px", "fontWeight": "600",
                 "background": "linear-gradient(90deg,#003087,#0067B1)", "border": "none",
                 "marginTop": "12px"},
                {"display": "block", "marginTop": "6px", "marginBottom": "2px"})
    return {"display": "none"}, {"display": "none"}


@app.callback(
    Output('feature-offcanvas', 'is_open'),
    Input('btn-open-editor', 'n_clicks'),
    Input('update-shap-button', 'n_clicks'),
    State('feature-offcanvas', 'is_open'),
    prevent_initial_call=True
)
def toggle_offcanvas(open_clicks, update_clicks, is_open):
    from dash import ctx
    if ctx.triggered_id == 'btn-open-editor':
        return True
    if ctx.triggered_id == 'update-shap-button':
        return False
    return is_open

@app.callback(
    Output('alpha-container-updated', 'style'),
    Input('update-shap-button', 'n_clicks'),
    Input('memory-calibration', 'data'),
    Input('interval-method-selector', 'value'),
    State('current-patient-index', 'data'),
    State('update-ran', 'data'),
)
def toggle_slider_visibility_updated(n_clicks, calib_data, interval_method, index, update_ran):
    if interval_method not in ['conformal', 'mc']:
        return {"display": "none"}
    if interval_method == 'conformal' and calib_data is None:
        return {"display": "none"}
    if index is not None and update_ran:
        return {
            "display": "block",
            "marginTop": "8px",
            "marginBottom": "16px",
            "background": "linear-gradient(180deg, #fff8f8 0%, #fff0f0 100%)",
            "borderRadius": "12px",
            "padding": "10px 12px",
            "border": "1px solid rgba(244,210,210,.95)",
            "boxShadow": "0 6px 18px rgba(122,40,40,.05)",
        }
    return {"display": "none"}


@app.callback(
    Output('mortality-plot-updated', 'children'),
    Output('Current-coefficients', 'data'),
    Input('update-shap-button', 'n_clicks'),
    Input('alpha-slider-updated', 'value'),
    Input('memory-calibration', 'data'),
    State('editable-table', 'data'),
    State('current-patient-index', 'data'),
    State('Current-mortality', 'data'),
    State('interval-method-selector', 'value'),
    State('memory-predictions', 'data'),
    State('window-width-store', 'data'),
    State('update-ran', 'data'),
    prevent_initial_call=True
)
def update_mortality(n_clicks, alpha_value, memory_calibration, edited_data, index, current_mortality, interval_method, memory_current, window_width, update_ran):
    from dash import ctx

    trigger = ctx.triggered_id
    # Avoid showing "Updating mortality..." when user just switches patients.
    # Only run after update is executed, or when adjusting updated alpha afterwards.
    if trigger == 'update-shap-button':
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
    elif trigger in ['alpha-slider-updated', 'memory-calibration']:
        if not update_ran:
            raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate

    alpha_value = 0.05 if alpha_value is None else float(alpha_value)
    alpha_value = float(np.clip(alpha_value, 0.01, 0.5))

    selected_method = interval_method or ('conformal' if memory_calibration is not None else 'mc')
    plot_interval = bool(selected_method == 'conformal' and memory_calibration is not None)
    plot_mc = bool(selected_method == 'mc')

    if not edited_data or index is None:
        raise dash.exceptions.PreventUpdate

    df_raw = pd.DataFrame(edited_data)
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    df_raw_feature = df_raw.iloc[:, :68]  
    mask = ~np.isnan(df_raw_feature)
    df_raw_feature_scaled = (df_raw_feature - x_mean) / x_std
    df_raw_feature_scaled = df_raw_feature_scaled.fillna(0)
    df_raw_feature_scaled_with_mask = np.concatenate((df_raw_feature_scaled.values, mask), axis=1)
    mask_tensor = paddle.to_tensor(mask.to_numpy().astype('float32'))
    input_tensor = paddle.to_tensor(df_raw_feature_scaled.values.astype('float32'))
    predictions, coefficients = model.predict(input_tensor, mask_tensor)
    
    predictions = predictions[:, 0, :].numpy()
    mortality = np.cumsum(predictions, axis=1)
    mortality[:, -1] = 1
    
    coefficients_np = [c.numpy().tolist() for c in coefficients]


    display_horizon = DEFAULT_MORTALITY_DISPLAY_HORIZON
    y = mortality[0, :display_horizon]
    y_current = np.array(current_mortality)[:display_horizon]
    is_mobile = window_width is not None and window_width < 768
    legend_font_size = 8 if is_mobile else 10
    legend_entry_width = 120 if is_mobile else None
    yaxis_title_size = 10 if is_mobile else 12
    margin_left = 38 if is_mobile else 60
    margin_right = 8 if is_mobile else 20
    
    x = list(range(1, len(y)+1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_current, mode='lines+markers', name='Original',
        line=dict(color="#0067B1", width=2.5),
        marker=dict(size=7, color="#0067B1", line=dict(color="white", width=1.5))
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines+markers', name='Updated',
        line=dict(color="#C8102E", width=2.5),
        marker=dict(size=7, color="#C8102E", line=dict(color="white", width=1.5))
    ))

    orig_lower = None
    orig_upper = None
    upd_lower = None
    upd_upper = None
    
    if plot_interval:
        data_calibration_scaled_with_mask = np.asarray(memory_calibration['data_calibration_scaled_with_mask'], dtype=np.float32)
        
        Event_time_calibration = np.asarray(memory_calibration['Event_time_calibration']).reshape(-1).astype(np.int64)
        
        Event_status_calibration = np.asarray(memory_calibration['Event_status_calibration']).reshape(-1).astype(np.int64)
        
        X_and_mask_test = df_raw_feature_scaled_with_mask 
        
        result = conformal_mortality_prediction(
            model_original=model_copy,
            X_and_mask=data_calibration_scaled_with_mask,
            E_train=Event_status_calibration,
            T_train=Event_time_calibration,
            X_and_mask_test=X_and_mask_test,
            max_horizon=display_horizon,
            alpha=alpha_value
        )
        lower_update = result['corrected_lower_bounds']
        upper_update = result['corrected_upper_bounds']

        df_scaled = pd.DataFrame(memory_current['scaled_df'])
        mask = pd.DataFrame(memory_current['mask'],dtype='float32')
        X_and_mask_eval = np.concatenate((df_scaled.values, mask.values), axis=1)
        
        X_and_mask_test_current = X_and_mask_eval[index].reshape(1, -1)
        result_current = conformal_mortality_prediction(
            model_original=model_copy,
            X_and_mask=data_calibration_scaled_with_mask,
            E_train=Event_status_calibration,
            T_train=Event_time_calibration,
            X_and_mask_test=X_and_mask_test_current,
            max_horizon=display_horizon,
            alpha=alpha_value
        )
        lower = result_current['corrected_lower_bounds']
        upper = result_current['corrected_upper_bounds']
        orig_lower = lower
        orig_upper = upper
        upd_lower = lower_update
        upd_upper = upper_update

        fig.add_traces([
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),  # forward + reverse
                y=np.concatenate([upper, lower[::-1]]),  # upper bound followed by lower bound reversed
                fill='toself',
                fillcolor='rgba(0, 123, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f"Orig int {int((1 - alpha_value) * 100)}%"
            ),
            go.Scatter(x=x, y=upper, line=dict(dash='dash', color='rgba(0, 123, 255, 0.2)'), mode='lines', showlegend=False, hoverinfo='skip', name='Upper Bound'),
            go.Scatter(x=x, y=lower, line=dict(dash='dash', color='rgba(0, 123, 255, 0.2)'), mode='lines', showlegend=False, hoverinfo='skip', name='Lower Bound')
        ])
        # Add shaded confidence interval
        fig.add_traces([
            go.Scatter(
                x=x + x[::-1],  # forward + reverse
                y=upper_update + lower_update[::-1],  # upper bound followed by lower bound reversed
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f"Upd int {int((1 - alpha_value) * 100)}%"
            ),
            go.Scatter(x=x, y=upper_update, line=dict(dash='dash', color='rgba(255, 0, 0, 0.2)'), mode='lines', showlegend=False, hoverinfo='skip', name='Upper Bound'),
            go.Scatter(x=x, y=lower_update, line=dict(dash='dash', color='rgba(255, 0, 0, 0.2)'), mode='lines', showlegend=False, hoverinfo='skip', name='Lower Bound')
        ])

    elif plot_mc:
        # Simultaneous uncertainty band for original patient
        df_scaled = pd.DataFrame(memory_current['scaled_df'])
        mask_orig = pd.DataFrame(memory_current['mask'], dtype='float32')
        i = index
        orig_input = paddle.to_tensor(df_scaled.values[i:i+1].astype('float32'))
        orig_mask = paddle.to_tensor(mask_orig.values[i:i+1].astype('float32'))
        lower_orig, upper_orig, mc_mean_orig = mc_dropout_predict(model_copy, orig_input, orig_mask, alpha=alpha_value, label='Original Patient')
        lower_orig = lower_orig[0, :display_horizon]
        upper_orig = upper_orig[0, :display_horizon]
        mc_mean_orig = mc_mean_orig[0, :display_horizon]

        fig.add_traces([
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([upper_orig, lower_orig[::-1]]),
                fill='toself',
                fillcolor='rgba(0, 123, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f"Orig band {int((1 - alpha_value) * 100)}%"
            ),
            go.Scatter(x=x, y=upper_orig.tolist(), line=dict(dash='dash', color='rgba(0, 123, 255, 0.3)'), mode='lines', showlegend=False, hoverinfo='skip', name='Upper Bound'),
            go.Scatter(x=x, y=lower_orig.tolist(), line=dict(dash='dash', color='rgba(0, 123, 255, 0.3)'), mode='lines', showlegend=False, hoverinfo='skip', name='Lower Bound'),
            go.Scatter(x=x, y=mc_mean_orig.tolist(), mode='lines+markers', name='Orig mean',
                       line=dict(color='rgba(0, 123, 255, 0.8)', width=2, dash='dot'),
                       marker=dict(size=5, color='rgba(0, 123, 255, 0.8)'),
                       visible='legendonly')
        ])

        # Simultaneous uncertainty band for updated patient
        lower_upd, upper_upd, mc_mean_upd = mc_dropout_predict(model_copy, input_tensor, mask_tensor, alpha=alpha_value, label='Updated Patient')
        lower_upd = lower_upd[0, :display_horizon]
        upper_upd = upper_upd[0, :display_horizon]
        mc_mean_upd = mc_mean_upd[0, :display_horizon]

        fig.add_traces([
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([upper_upd, lower_upd[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f"Upd band {int((1 - alpha_value) * 100)}%"
            ),
            go.Scatter(x=x, y=upper_upd.tolist(), line=dict(dash='dash', color='rgba(255, 0, 0, 0.3)'), mode='lines', showlegend=False, hoverinfo='skip', name='Upper Bound'),
            go.Scatter(x=x, y=lower_upd.tolist(), line=dict(dash='dash', color='rgba(255, 0, 0, 0.3)'), mode='lines', showlegend=False, hoverinfo='skip', name='Lower Bound'),
            go.Scatter(x=x, y=mc_mean_upd.tolist(), mode='lines+markers', name='Upd mean',
                       line=dict(color='rgba(255, 0, 0, 0.8)', width=2, dash='dot'),
                       marker=dict(size=5, color='rgba(255, 0, 0, 0.8)'),
                       visible='legendonly')
        ])

        orig_lower = lower_orig.tolist()
        orig_upper = upper_orig.tolist()
        upd_lower = lower_upd.tolist()
        upd_upper = upper_upd.tolist()

    if orig_lower is not None and orig_upper is not None:
        fig.data[0].customdata = np.column_stack([
            np.asarray(orig_lower, dtype=np.float64),
            np.asarray(orig_upper, dtype=np.float64)
        ])
        fig.data[0].hovertemplate = "%{y:.2f} [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra>%{fullData.name}</extra>"
    else:
        fig.data[0].hovertemplate = "%{y:.2f}<extra>%{fullData.name}</extra>"

    if upd_lower is not None and upd_upper is not None:
        fig.data[1].customdata = np.column_stack([
            np.asarray(upd_lower, dtype=np.float64),
            np.asarray(upd_upper, dtype=np.float64)
        ])
        fig.data[1].hovertemplate = "%{y:.2f} [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra>%{fullData.name}</extra>"
    else:
        fig.data[1].hovertemplate = "%{y:.2f}<extra>%{fullData.name}</extra>"

    
    mobile_height = get_mobile_43_height(window_width) if is_mobile else 500

    fig.update_layout(
        title=dict(text=f'Updated Cumulative Mortality — Modified Patient {index + 1}',
                   font=dict(size=15, color="#003087", family="Segoe UI, Arial")),
        xaxis_title='Year',
        template='plotly_white',
        xaxis=dict(tickmode='linear', dtick=1, showgrid=True,
                   gridcolor="#e8eef4", gridwidth=1),
        yaxis=dict(range=[-0.01, 1.01], showgrid=True,
                   gridcolor="#e8eef4", gridwidth=1,
                   title=dict(text='Cumulative Mortality', font=dict(size=yaxis_title_size), standoff=2 if is_mobile else 8),
                   tickformat=".0%"),
        plot_bgcolor="rgba(248,251,255,0.9)", paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#444"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.60)", bordercolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(size=legend_font_size), itemsizing="constant",
                    entrywidth=legend_entry_width, entrywidthmode='pixels'),
        margin=dict(t=105, b=70, l=margin_left, r=margin_right),
        hovermode="x unified",
        height=mobile_height
    )
    interval_lower = None
    interval_upper = None
    if plot_interval:
        interval_lower = lower_update
        interval_upper = upper_update
    elif plot_mc:
        interval_lower = lower_upd.tolist()
        interval_upper = upper_upd.tolist()
    kpi_cards = create_kpi_cards(y, interval_lower, interval_upper, accent_color="#C8102E")
    return [kpi_cards, dcc.Graph(figure=fig, config={'displayModeBar': False, 'responsive': True}, style={"width": "100%"}), html.Hr()], {'coefficients': coefficients_np} 


@app.callback(
    Output('interval-enabled-updated', 'data'),
    Input('alpha-slider-updated', 'value'),
    Input('memory-calibration', 'data'),
    Input('update-shap-button', 'n_clicks'),
    prevent_initial_call=True
)
def control_interval_visibility_updated(alpha_value, memory_calibration, n_clicks):
    from dash import ctx
    trigger = ctx.triggered_id
    if trigger == 'alpha-slider-updated' and memory_calibration is not None:
        return True
    if trigger in ['memory-calibration', 'update-shap-button']:
        return False
    raise dash.exceptions.PreventUpdate

@app.callback(
    Output('trajectory-plot-updated', 'children'),
    Input('Current-coefficients', 'data'),
    Input('alpha-slider-updated', 'value'),
    State('memory-predictions', 'data'),
    State('current-patient-index', 'data'),
    State('window-width-store', 'data'),
    State('interval-method-selector', 'value'),
    State('editable-table', 'data'),
    prevent_initial_call=True
)
def update_plot_trajectory(memory_coefficients, alpha_value, memory, index, window_width, interval_method, edited_data):
    if memory_coefficients is None or index is None:
        raise dash.exceptions.PreventUpdate
    coefficients_data = memory['coefficients']
    coeffcients = [paddle.to_tensor(np.array(c), dtype='float32') for c in coefficients_data]

    updated_coeffcients_data = memory_coefficients['coefficients']
    updated_coeffcients = [paddle.to_tensor(np.array(c), dtype='float32') for c in updated_coeffcients_data]

    original_band = None
    updated_band = None
    alpha_value = 0.05 if alpha_value is None else float(alpha_value)
    alpha_value = float(np.clip(alpha_value, 0.01, 0.5))

    if (interval_method or 'mc') == 'mc':
        df_scaled = pd.DataFrame(memory['scaled_df'])
        mask_df = pd.DataFrame(memory['mask'], dtype='float32')
        orig_input = paddle.to_tensor(df_scaled.values[index:index+1].astype('float32'))
        orig_mask = paddle.to_tensor(mask_df.values[index:index+1].astype('float32'))
        lo_o, up_o, _ = mc_dropout_trajectory_band(
            model_copy,
            orig_input,
            orig_mask,
            person_id=index,
            alpha=alpha_value,
            n_samples=1000,
            label='Trajectory (original)',
        )
        original_band = {'lower': lo_o, 'upper': up_o}

        if edited_data:
            df_raw = pd.DataFrame(edited_data)
            df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
            df_raw_feature = df_raw.iloc[:, :68]
            mask_upd = ~np.isnan(df_raw_feature)
            df_raw_feature_scaled = (df_raw_feature - x_mean) / x_std
            df_raw_feature_scaled = df_raw_feature_scaled.fillna(0)
            upd_input = paddle.to_tensor(df_raw_feature_scaled.values.astype('float32'))
            upd_mask = paddle.to_tensor(mask_upd.to_numpy().astype('float32'))
            lo_u, up_u, _ = mc_dropout_trajectory_band(
                model_copy,
                upd_input,
                upd_mask,
                person_id=index,
                alpha=alpha_value,
                n_samples=1000,
                label='Trajectory (updated)',
            )
            updated_band = {'lower': lo_u, 'upper': up_u}

    fig = create_trajectory_plot(
        index,
        coeffcients,
        updated_coeffcients=updated_coeffcients,
        window_width=window_width,
        original_band=original_band,
        updated_band=updated_band,
        alpha=alpha_value,
    )
    return [dcc.Graph(figure=fig, config={'displayModeBar': False, 'responsive': True}, style={"width": "100%"}), html.Hr()]

@app.callback(
    Output('shap-plot-updated', 'children'),
    Input('update-shap-button', 'n_clicks'),
    State('editable-table', 'data'),
    State('memory-predictions', 'data'),
    State('current-patient-index', 'data'),
    State('current-order', 'data'),
    State('edited-row', 'data'),
    prevent_initial_call=True
)
def update_shap(n_clicks, edited_data, memory, index, current_order, original_row):
    if not edited_data or not memory:
        raise dash.exceptions.PreventUpdate

    df_raw = pd.DataFrame(edited_data)
    df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
    df_raw_feature = df_raw.iloc[:, :68]
    df_raw_feature_scaled = (df_raw_feature - x_mean) / x_std
    df_raw_feature_scaled = df_raw_feature_scaled.fillna(0)
    mask = ~np.isnan(df_raw_feature)
    X_and_mask_eval = np.concatenate((df_raw_feature_scaled.values, mask), axis=1)
    df_raw_mask = pd.DataFrame(mask, dtype='float32')
    combined = pd.concat([df_raw_feature, df_raw_mask], axis=1)
    combined.columns = feature_name

    img, _ = get_waterfall_base64(X_and_mask_eval, combined, 0, order=current_order)

    # ── Build change summary ─────────────────────────────────────────────────
    change_items = []
    if original_row:
        orig = {k: v for k, v in original_row.items()}
        new_row = df_raw.iloc[0].to_dict()
        feat_cols = list(df_raw_feature.columns)
        for col in feat_cols:
            orig_val = orig.get(col)
            new_val  = new_row.get(col)
            try:
                orig_f = float(orig_val) if orig_val is not None else None
                new_f  = float(new_val)  if new_val  is not None else None
                if orig_f is not None and new_f is not None and abs(orig_f - new_f) > 1e-9:
                    change_items.append(
                        html.Tr([
                            html.Td(col,            style={"padding": "4px 10px",
                                                           "fontWeight": "500"}),
                            html.Td(f"{orig_f:.4g}", style={"padding": "4px 10px",
                                                             "color": "#C8102E",
                                                             "textDecoration": "line-through"}),
                            html.Td("→",             style={"padding": "4px 6px",
                                                             "color": "#6c757d"}),
                            html.Td(f"{new_f:.4g}",  style={"padding": "4px 10px",
                                                             "color": "#0067B1",
                                                             "fontWeight": "600"}),
                        ])
                    )
            except (TypeError, ValueError):
                pass

    change_summary = html.Div()
    if change_items:
        change_summary = html.Div([
            html.Div([
                html.I(className="fas fa-exchange-alt me-2"),
                html.Span("Feature Changes", style={"fontWeight": "600"})
            ], style={"marginBottom": "8px", "color": "#003087",
                      "fontSize": "0.95rem"}),
            html.Table(
                [html.Thead(html.Tr([
                    html.Th("Variable",      style={"padding": "4px 10px",
                                                    "borderBottom": "2px solid #dee2e6"}),
                    html.Th("Original",      style={"padding": "4px 10px",
                                                    "borderBottom": "2px solid #dee2e6",
                                                    "color": "#C8102E"}),
                    html.Th("",              style={"padding": "4px 6px",
                                                    "borderBottom": "2px solid #dee2e6"}),
                    html.Th("Updated",       style={"padding": "4px 10px",
                                                    "borderBottom": "2px solid #dee2e6",
                                                    "color": "#0067B1"}),
                ]))] + [html.Tbody(change_items)],
                style={"fontSize": "0.85rem", "borderCollapse": "collapse",
                       "width": "100%"}
            )
        ], style={"background": "#f8f9ff", "borderRadius": "8px",
                  "padding": "14px 16px", "marginBottom": "16px",
                  "border": "1px solid #dee2e6"})

    return html.Div([
        html.Div([
            html.I(className="fas fa-water me-2 text-danger"),
            html.Span(f"Updated SHAP Waterfall — Modified Patient {index + 1}",
                      style={"fontWeight": "600", "fontSize": "1rem", "color": "#C8102E"})
        ], style={"marginBottom": "10px"}),
        change_summary,
        html.Img(src=img, style={
            'maxWidth': '100%', 'height': 'auto',
            'border': '1px solid #f5c2c7', 'borderRadius': '8px',
            'boxShadow': '0 2px 8px rgba(230,57,70,.12)'
        }),
        html.Hr(style={"borderColor": "#dee2e6"})
    ])



# ── Toast: show when Update Analysis is clicked ─────────────────────────────
@app.callback(
    Output('update-toast', 'is_open'),
    Input('update-shap-button', 'n_clicks'),
    prevent_initial_call=True
)
def show_update_toast(n_clicks):
    return True


# ── Auto-scroll to updated results after update ──────────────────────────────
app.clientside_callback(
    """
    function(children) {
        if (children) {
            setTimeout(function() {
                var el = document.getElementById('mortality-plot-updated');
                if (el) {
                    el.scrollIntoView({behavior: 'smooth', block: 'start'});
                }
            }, 400);
        }
        return '';
    }
    """,
    Output('scroll-dummy', 'children'),
    Input('mortality-plot-updated', 'children'),
    prevent_initial_call=True
)


# ── Hide original plots once updated plots are available ─────────────────────
@app.callback(
    Output('wrap-mortality',  'style'),
    Output('wrap-trajectory', 'style'),
    Input('update-shap-button', 'n_clicks'),
    Input('x-table', 'active_cell'),
    prevent_initial_call=True
)
def toggle_original_plots(n_clicks, active_cell):
    from dash import ctx
    if ctx.triggered_id == 'update-shap-button' and n_clicks:
        return {"display": "none"}, {"display": "none"}
    return {"display": "block"}, {"display": "block"}


# ── Hide stale updated plots when selecting a different patient ─────────────
@app.callback(
    Output('mortality-plot-updated', 'style'),
    Output('trajectory-plot-updated', 'style'),
    Output('shap-plot-updated', 'style'),
    Input('update-shap-button', 'n_clicks'),
    Input('x-table', 'active_cell'),
    prevent_initial_call=True
)
def toggle_updated_plots_visibility(n_clicks, active_cell):
    from dash import ctx
    if ctx.triggered_id == 'update-shap-button' and n_clicks:
        return {"display": "block"}, {"display": "block"}, {"display": "block"}
    return {"display": "none"}, {"display": "none"}, {"display": "none"}


@app.callback(
    Output('wrap-loading-update-mortality', 'style'),
    Output('wrap-loading-update-trajectory', 'style'),
    Output('wrap-loading-update-shap', 'style'),
    Input('update-shap-button', 'n_clicks'),
    Input('x-table', 'active_cell'),
    prevent_initial_call=True
)
def toggle_updated_loading_blocks(n_clicks, active_cell):
    from dash import ctx
    if ctx.triggered_id == 'update-shap-button' and n_clicks:
        return {"display": "block"}, {"display": "block"}, {"display": "block"}
    return {"display": "none"}, {"display": "none"}, {"display": "none"}


# ── Feature search: filter editable-table columns by name ────────────────────
@app.callback(
    Output('editable-table', 'columns'),
    Output('feature-search', 'value'),
    Input('feature-search', 'value'),
    Input('feature-search-clear', 'n_clicks'),
    State('edited-row', 'data'),
    prevent_initial_call=True
)
def search_feature_columns(search, clear_clicks, row_data):
    from dash import ctx
    if not row_data:
        raise dash.exceptions.PreventUpdate
    all_cols = [{'name': k, 'id': k, 'editable': True} for k in row_data]
    if ctx.triggered_id == 'feature-search-clear':
        return all_cols, ""
    if not search or not search.strip():
        return all_cols, dash.no_update
    filtered = [c for c in all_cols if search.strip().lower() in c['name'].lower()]
    return (filtered if filtered else all_cols), dash.no_update


@app.callback(
    Output('alpha-slider', 'marks'),
    Output('alpha-slider-updated', 'marks'),
    Input('window-width-store', 'data'),
    prevent_initial_call=False
)
def update_alpha_slider_marks(window_width):
    is_mobile = window_width is not None and window_width < 768
    marks = ALPHA_MARKS_MOBILE if is_mobile else ALPHA_MARKS_DESKTOP
    return marks, marks


# Capture browser window width once on page load
app.clientside_callback(
    """
    function(_) {
        return window.innerWidth;
    }
    """,
    Output('window-width-store', 'data'),
    Input('window-width-store', 'id'),
)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)

