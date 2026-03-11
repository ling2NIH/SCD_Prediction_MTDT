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

#######################refine utility functions for the app ###########################

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
        height=220 * math.ceil(num_variables / 4),
        title_text=f"Predicted 3-Year Trajectories of Risk Factors — Patient {person_id + 1}",
        title_font=dict(size=15, color="#003087", family="Segoe UI, Arial"),
        template='plotly_white',
        plot_bgcolor="rgba(248,251,255,0.9)",
        paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=11, color="#444"),
        margin=dict(t=80, b=40, l=40, r=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#dee2e6", borderwidth=1
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

    y_pred_calib = model_wraped.predict(X_calib)

    residuals = np.abs(y_calib.reshape(-1) - y_pred_calib.reshape(-1)) 
    n = len(y_calib)
    tau = np.ceil((1 - alpha) * (n + 1)) / n
    q = np.quantile(residuals, tau)
    #q = np.quantile(residuals, 1 - alpha)
    
    y_pred_test = model_wraped.predict(X_test)

    lower_bound = np.clip(y_pred_test - q, 0, 1)
    upper_bound = np.clip(y_pred_test + q, 0, 1)
    #lower_bound = y_pred_test - q
    #upper_bound = y_pred_test + q
    intervals = np.vstack([lower_bound, upper_bound]).T
    return y_pred_test, intervals,residuals

def conformal_mortality_prediction(
    model_original,
    X_and_mask,
    E_train,
    T_train,
    X_and_mask_test,
    max_horizon=10,
    alpha=0.05
):

    corrected_upper_bound = 0
    corrected_lower_bound = 0

    years = []
    predicted_mortality = []
    lower_bounds = []
    upper_bounds = []
    corrected_lower_bounds = []
    corrected_upper_bounds = []

    for i in range(max_horizon):
        censored_mask = (E_train == 0).squeeze()
        uncensored_mask = (E_train == 1).squeeze()
        censored_alive_mask = censored_mask & (T_train.squeeze() >= i + 1)
        available_mask = (uncensored_mask | censored_alive_mask)

        X_and_mask_available = X_and_mask[available_mask]
        

        y_label = np.zeros_like(E_train)
        for j in range(len(E_train)):
            if E_train[j] == 1 and T_train[j] <= i + 1:
                y_label[j] = 1
            elif E_train[j] == 1 and T_train[j] > i + 1:
                y_label[j] = 0
            elif E_train[j] == 0 and T_train[j] >= i + 1:
                y_label[j] = 0

        y_label_available = y_label[available_mask].astype(float)

        y_pred_test, intervals, residuals = paddle_manual_conformal_interval(
            model_original, X_and_mask_available, y_label_available,
            X_and_mask_test, alpha=alpha, horizon=i + 1
        )

        # Conformal accumulation (monotonic correction)
        corrected_upper_bound = max(corrected_upper_bound, intervals[0, 1])
        corrected_lower_bound = max(corrected_lower_bound, intervals[0, 0])

        years.append(i + 1)
        predicted_mortality.append(y_pred_test[0])
        lower_bounds.append(intervals[0, 0])
        upper_bounds.append(intervals[0, 1])
        corrected_lower_bounds.append(corrected_lower_bound)
        corrected_upper_bounds.append(corrected_upper_bound)

        print(f'At year {i+1}, predicted mortality: {y_pred_test[0]:.4f}, '
              f'interval: [{intervals[0, 0]:.4f}, {intervals[0, 1]:.4f}], '
              f'corrected interval: [{corrected_lower_bound:.4f}, {corrected_upper_bound:.4f}]')

    return {
        'years': years,
        'predicted_mortality': predicted_mortality,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'corrected_lower_bounds': corrected_lower_bounds,
        'corrected_upper_bounds': corrected_upper_bounds
    }


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
      body { background: #eef2f7 !important; font-family: "Segoe UI", Arial, sans-serif; }

      /* ── upload drop-zone ── */
      .upload-zone {
        width: 100%; height: 80px; line-height: 80px;
        border: 2px dashed #0067B1; border-radius: 8px;
        text-align: center; background: #f8fbff; color: #0067B1;
        cursor: pointer; transition: background .2s;
      }
      .upload-zone:hover { background: #d6e8ff; }

      /* ── card tweaks ── */
      .card { border: none !important; border-radius: 12px !important; }
      .card-header { border-radius: 12px 12px 0 0 !important; }

      /* ── instruction steps ── */
      .step-badge {
        display: inline-block; width: 22px; height: 22px; line-height: 22px;
        border-radius: 50%; background: #0067B1; color: white;
        font-size: 11px; font-weight: bold; text-align: center; margin-right: 6px;
      }

      /* ── DataTable header ── */
      .dash-header { background-color: #003087 !important; color: white !important; }

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
        .app-header-icon  { font-size: 1.4rem !important; }
        .app-header-wrap  { padding: 18px 16px !important; }

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
                            "Multi-Task DeepHit v2.14  ·  Conformal Prediction Intervals  ·  SHAP Explainability",
                            className="app-header-sub",
                            style={"color": "rgba(255,255,255,.65)", "marginTop": "6px",
                                   "fontSize": "0.85rem", "marginBottom": "0"}
                        ),
                    ], style={"flex": "1"}),
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
                        style={"marginLeft": "24px", "flexShrink": "0"}
                    ),
                ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"}),
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
            ], className="shadow mb-3"),

            # Upload calibration data
            dbc.Card([
                dbc.CardHeader(
                    html.Div([
                        html.I(className="fas fa-chart-line me-2"),
                        html.Span("Upload Calibration Data", style={"fontWeight": "600"})
                    ]),
                    style={"background": "linear-gradient(90deg,#155724,#28a745)",
                           "color": "white", "padding": "12px 16px"}
                ),
                dbc.CardBody([
                    dbc.Button([
                        html.I(className="fas fa-file-csv me-2"),
                        "Download Example CSV"
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
                    html.Div(id='upload-calibration-status', className="mt-1 small text-muted")
                ])
            ], className="shadow mb-3"),

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
                        (3, "(Optional) Upload calibration data for prediction intervals."),
                        (4, "View mortality predictions in the results panel."),
                        (5, "Click a patient row to explore:"),
                    ]],
                    html.Ul([
                        html.Li("10-year cumulative mortality curve", style={"fontSize": "0.8rem"}),
                        html.Li("3-year trajectories of 12 risk factors", style={"fontSize": "0.8rem"}),
                        html.Li("SHAP waterfall for 5-year mortality", style={"fontSize": "0.8rem"}),
                    ], style={"paddingLeft": "28px", "marginBottom": "6px"}),
                    html.Div([
                        html.Span("6", className="step-badge"),
                        html.Span("Edit features & click 'Update Analysis'.", style={"fontSize": "0.82rem"})
                    ]),
                ])
            ], className="shadow mb-3"),

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

                    # ── Prediction table ────────────────────────────────────
                    dcc.Loading(id='loading-table', type='dot',
                                children=html.Div(id='output')),

                    # ── Conformal slider (original patient) ─────────────────
                    html.Div(
                        id='alpha-container',
                        children=[
                            dbc.Label([
                                html.I(className="fas fa-sliders-h me-2 text-primary"),
                                "Conformal Prediction Level (1 − α)"
                            ], style={"fontWeight": "600", "marginBottom": "6px"}),
                            dcc.Slider(
                                id='alpha-slider', min=0.01, max=0.5, step=0.01, value=0.05,
                                marks={0.01: '99%', 0.05: '95%', 0.1: '90%',
                                       0.2: '80%', 0.3: '70%', 0.5: '50%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ],
                        style={"display": "none",
                               "background": "#eef6ff", "borderRadius": "10px",
                               "padding": "16px", "marginTop": "10px"}
                    ),

                    # ── Mortality plot ──────────────────────────────────────
                    dcc.Loading(id='loading-mortality', type='dot',
                                children=html.Div(id='mortality-plot')),

                    # ── Trajectory plot ─────────────────────────────────────
                    dcc.Loading(id='loading-trajectory', type='dot',
                                children=html.Div(id='trajectory-plot')),

                    # ── SHAP ────────────────────────────────────────────────
                    dcc.Loading(id='loading-shap', type='dot',
                                children=html.Div(id='shap-plot')),

                    # ── Feature editor + Update button ──────────────────────
                    html.Div(id='feature-editor'),
                    dbc.Button([
                        html.I(className="fas fa-sync-alt me-2"),
                        "Update Analysis"
                    ], id='update-shap-button', color="primary",
                       className="mt-3 px-4", style={"display": "none",
                                                     "borderRadius": "8px",
                                                     "fontWeight": "600"}),

                    # ── Conformal slider (updated patient) ──────────────────
                    html.Div(
                        id='alpha-container-updated',
                        children=[
                            dbc.Label([
                                html.I(className="fas fa-sliders-h me-2 text-primary"),
                                "Conformal Prediction Level (1 − α) — Updated Patient"
                            ], style={"fontWeight": "600", "marginBottom": "6px"}),
                            dcc.Slider(
                                id='alpha-slider-updated', min=0.01, max=0.5, step=0.01, value=0.05,
                                marks={0.01: '99%', 0.05: '95%', 0.1: '90%',
                                       0.2: '80%', 0.3: '70%', 0.5: '50%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ],
                        style={"display": "none",
                               "background": "#fff3f3", "borderRadius": "10px",
                               "padding": "16px", "marginTop": "10px"}
                    ),

                    dcc.Loading(id='loading-update_mortality', type='dot',
                                children=html.Div(id='mortality-plot-updated')),
                    dcc.Loading(id='loading-update_trajectory', type='dot',
                                children=html.Div(id='trajectory-plot-updated')),
                    dcc.Loading(id='loading-update', type='dot',
                                children=html.Div(id='shap-plot-updated', className="mt-3")),
                ], id="main-results-body", style={"padding": "24px"})
            ], className="shadow"), xs=12, md=8, lg=9
        )
    ]),

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
                "v1.2.2"
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

    data_calibration_scaled = np.asarray(df_scaled)
    data_calibration_scaled_with_mask = np.concatenate((data_calibration_scaled, mask), axis=1)
    Event_time = np.asarray(df_calibration[['Event_time']])/365.25
    Event_time = np.ceil(Event_time)
    Event_time[Event_time >= 15] = 16

    Event_status = np.asarray(df_calibration[['Event_status']])

    print("Calibration data processed successfully.")
    

    return (
        {'data_calibration_scaled_with_mask': data_calibration_scaled_with_mask.tolist(),
         'Event_time_calibration': Event_time, 'Event_status_calibration': Event_status},
        dbc.Badge([html.I(className="fas fa-check-circle me-1"), f"{filename} uploaded"],
                  color="success", className="px-3 py-2"),
        dbc.Badge([html.I(className="fas fa-shield-alt me-1"), "Prediction interval enabled"],
                  color="success", className="mt-2 px-3 py-2 w-100 text-start"),
    )

@app.callback(
    Output('output', 'children'),
    Output('memory-predictions', 'data'),
    Output('data-status', 'children'),
    Output('upload-data-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def predict(contents, filename):
    if contents is None:
        return html.Div("Please upload a CSV file."), dash.no_update,dash.no_update,dash.no_update

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
        'whiteSpace': 'normal', 'fontSize': '12px',
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
                                 'borderRadius': '8px', 'overflow': 'hidden',
                                 'boxShadow': '0 1px 6px rgba(0,0,0,.08)'},
                    style_header=_tbl_header,
                    style_cell=_tbl_cell,
                    style_data_conditional=_tbl_data_cond,
                )
            ], className="tbl-left",
               style={'height': '380px', 'overflowY': 'scroll', 'overflowX': 'auto'}),

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
                                 'borderRadius': '8px', 'overflow': 'hidden',
                                 'boxShadow': '0 1px 6px rgba(0,0,0,.08)'},
                    style_header=_tbl_header,
                    style_cell={**_tbl_cell, 'minWidth': '100px'},
                    style_data_conditional=_tbl_data_cond,
                )
            ], className="tbl-right",
               style={'height': '380px', 'overflowY': 'scroll', 'overflowX': 'auto'})
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
            "Click on a patient row to view the mortality curve, risk factor trajectories, and SHAP analysis."
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
    Output('alpha-container', 'style'),
    Input('memory-calibration', 'data'),
    Input('x-table', 'active_cell')
)
def toggle_slider_visibility(calib_data, active_cell):
    if calib_data is not None and active_cell is not None:
        return {"display": "block", "marginTop": "20px"}
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
    State('memory-predictions', 'data'),
    State('memory-calibration', 'data'),
    prevent_initial_call=True
)
def plot_mortality(active_cell, alpha_value, memory, memory_calibration):
    if not memory or not active_cell:
        raise dash.exceptions.PreventUpdate

    if memory_calibration is not None:
        plot_interval = True
    else:
        plot_interval = False

    i = active_cell['row']
    mortality = np.array([
        [v for k, v in row.items() if k != 'Patient ID']
        for row in memory['pred_df']
    ])

    y = mortality[i,:10]
    x = list(range(1, len(y)+1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines+markers', name=f'Patient {i + 1}',
        line=dict(color="#0067B1", width=2.5),
        marker=dict(size=7, color="#0067B1", line=dict(color="white", width=1.5))
    ))

    if plot_interval:
        data_calibration_scaled_with_mask = np.array(memory_calibration['data_calibration_scaled_with_mask'])
        
        Event_time_calibration = np.array(memory_calibration['Event_time_calibration'])
        
        Event_status_calibration = np.array(memory_calibration['Event_status_calibration'])
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
            max_horizon=10,
            alpha=alpha_value
        )
        lower = result['corrected_lower_bounds']
        upper = result['corrected_upper_bounds']

        # Add shaded confidence interval
        fig.add_traces([
            go.Scatter(
                x=x + x[::-1],  # forward + reverse
                y=upper + lower[::-1],  # upper bound followed by lower bound reversed
                fill='toself',
                fillcolor='rgba(0, 123, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f"{int((1 - alpha_value) * 100)}% Interval"
            ),
            go.Scatter(x=x, y=upper, line=dict(dash='dash', color='rgba(0, 123, 255, 0.2)'), mode='lines', showlegend=False,name='Upper Bound'),
            go.Scatter(x=x, y=lower, line=dict(dash='dash', color='rgba(0, 123, 255, 0.2)'), mode='lines', showlegend=False,name='Lower Bound')
        ])
        

    fig.update_layout(
        title=dict(text=f'Cumulative Mortality Risk — Patient {i + 1}',
                   font=dict(size=15, color="#003087", family="Segoe UI, Arial")),
        xaxis_title='Year', yaxis_title='Cumulative Mortality',
        template='plotly_white',
        xaxis=dict(tickmode='linear', dtick=1, showgrid=True,
                   gridcolor="#e8eef4", gridwidth=1),
        yaxis=dict(range=[-0.01, 1.01], showgrid=True,
                   gridcolor="#e8eef4", gridwidth=1,
                   tickformat=".0%"),
        plot_bgcolor="rgba(248,251,255,0.9)", paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#444"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#dee2e6", borderwidth=1),
        margin=dict(t=60, b=50, l=60, r=20),
        hovermode="x unified",
    )
    return [dcc.Graph(figure=fig, config={'displayModeBar': False}), html.Hr()], y.tolist(), {'lower_bounds': lower, 'upper_bounds': upper} if plot_interval else None

@app.callback(
    Output('trajectory-plot', 'children'),
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
    return [dcc.Graph(figure=fig, config={'displayModeBar': False}), html.Hr()]

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
                     'overflow': 'hidden', 'boxShadow': '0 1px 6px rgba(0,0,0,.08)'},
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
            html.Br(), html.Br(),
            dbc.Alert([
                html.I(className="fas fa-edit me-2"),
                "Edit the baseline feature values below and click 'Update Analysis' to see the revised predictions."
            ], color="light", className="py-2 px-3 border",
               style={"fontSize": "0.85rem", "borderRadius": "8px",
                      "borderColor": "#bee2ff !important"}),
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
    prevent_initial_call=True
)
def toggle_update_button_visibility(index):
    if index is not None:
        return {"display": "block", "marginTop": "10px"}
    return {"display": "none"}

@app.callback(
    Output('alpha-container-updated', 'style'),
    Input('update-shap-button', 'n_clicks'),
    Input('memory-calibration', 'data'),
    State('current-patient-index', 'data'),
)

def toggle_slider_visibility(n_clicks,calib_data, index):
    if calib_data is not None and n_clicks is not None and index is not None:
        return {"display": "block", "marginTop": "20px"}
    return {"display": "none"}


@app.callback(
    Output('mortality-plot-updated', 'children'),
    Output('Current-coefficients', 'data'),
    Input('update-shap-button', 'n_clicks'),
    Input('alpha-slider-updated', 'value'),
    State('editable-table', 'data'),
    State('current-patient-index', 'data'),
    State('Current-mortality', 'data'),
    State('memory-predictions', 'data'),
    State('memory-calibration', 'data'),
    prevent_initial_call=True
)
def update_mortality(n_clicks, alpha_value,edited_data, index, current_mortality, memory_current,memory_calibration):
    if memory_calibration is not None:
        plot_interval = True
    else:
        plot_interval = False

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


    y = mortality[0,:10]
    y_current = np.array(current_mortality)
    
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
    
    if plot_interval:
        data_calibration_scaled_with_mask = np.array(memory_calibration['data_calibration_scaled_with_mask'])
        
        Event_time_calibration = np.array(memory_calibration['Event_time_calibration'])
        
        Event_status_calibration = np.array(memory_calibration['Event_status_calibration'])
        
        X_and_mask_test = df_raw_feature_scaled_with_mask 
        
        result = conformal_mortality_prediction(
            model_original=model_copy,
            X_and_mask=data_calibration_scaled_with_mask,
            E_train=Event_status_calibration,
            T_train=Event_time_calibration,
            X_and_mask_test=X_and_mask_test,
            max_horizon=10,
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
            max_horizon=10,
            alpha=alpha_value
        )
        lower = result_current['corrected_lower_bounds']
        upper = result_current['corrected_upper_bounds']

        fig.add_traces([
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),  # forward + reverse
                y=np.concatenate([upper, lower[::-1]]),  # upper bound followed by lower bound reversed
                fill='toself',
                fillcolor='rgba(0, 123, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name=f"Original {int((1 - alpha_value) * 100)}% Interval"
            ),
            go.Scatter(x=x, y=upper, line=dict(dash='dash', color='rgba(0, 123, 255, 0.2)'), mode='lines', showlegend=False,name='Upper Bound'),
            go.Scatter(x=x, y=lower, line=dict(dash='dash', color='rgba(0, 123, 255, 0.2)'), mode='lines', showlegend=False,name='Lower Bound')
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
                name=f"Updated {int((1 - alpha_value) * 100)}% Interval"
            ),
            go.Scatter(x=x, y=upper_update, line=dict(dash='dash', color='rgba(255, 0, 0, 0.2)'), mode='lines', showlegend=False,name='Upper Bound'),
            go.Scatter(x=x, y=lower_update, line=dict(dash='dash', color='rgba(255, 0, 0, 0.2)'), mode='lines', showlegend=False,name='Lower Bound')
        ])



    fig.update_layout(
        title=dict(text=f'Updated Cumulative Mortality — Modified Patient {index + 1}',
                   font=dict(size=15, color="#003087", family="Segoe UI, Arial")),
        xaxis_title='Year', yaxis_title='Cumulative Mortality',
        template='plotly_white',
        xaxis=dict(tickmode='linear', dtick=1, showgrid=True,
                   gridcolor="#e8eef4", gridwidth=1),
        yaxis=dict(range=[-0.01, 1.01], showgrid=True,
                   gridcolor="#e8eef4", gridwidth=1,
                   tickformat=".0%"),
        plot_bgcolor="rgba(248,251,255,0.9)", paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#444"),
        legend=dict(bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#dee2e6", borderwidth=1),
        margin=dict(t=60, b=50, l=60, r=20),
        hovermode="x unified",
    )
    return [dcc.Graph(figure=fig, config={'displayModeBar': False}), html.Hr()], {'coefficients': coefficients_np} 

@app.callback(
    Output('trajectory-plot-updated', 'children'),
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
    return [dcc.Graph(figure=fig, config={'displayModeBar': False}), html.Hr()]

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
        html.Div([
            html.I(className="fas fa-water me-2 text-danger"),
            html.Span(f"Updated SHAP Waterfall — Modified Patient {index + 1}",
                      style={"fontWeight": "600", "fontSize": "1rem", "color": "#C8102E"})
        ], style={"marginBottom": "10px"}),
        html.Img(src=img, style={
            'maxWidth': '100%', 'height': 'auto',
            'border': '1px solid #f5c2c7', 'borderRadius': '8px',
            'boxShadow': '0 2px 8px rgba(230,57,70,.12)'
        }),
        html.Hr(style={"borderColor": "#dee2e6"})
    ])



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  
    app.run(debug=False, host='0.0.0.0', port=port)

