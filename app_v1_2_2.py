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

def predict_flat(X_and_mask):

    X = X_and_mask[:, :68]
    mask = X_and_mask[:, 68:]
    X_tensor = paddle.to_tensor(X, dtype='float32')
    mask_tensor = paddle.to_tensor(mask, dtype='float32')

    survival_pred,_ = model.predict(X_tensor, mask_tensor)
    survival_pred = survival_pred[:, 0, :]
    survival_pred = survival_pred.numpy()

    return np.sum(survival_pred[:, 0:5], axis=1)

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
        height=200 * math.ceil(num_variables / 4),
        title_text=f"Predicted 3-Year Trajectories of Risk Factors for Patient {person_id+1}",
        template='plotly_white'
    )

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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Sickle Cell Disease Mortality Prediction"

# Layout: UI improvements only
app.layout = dbc.Container(fluid=True, children=[
    # Hidden stores
    dcc.Store(id='memory-predictions'),
    dcc.Store(id='memory-calibration'),
    dcc.Store(id='current-patient-index'),
    dcc.Store(id='edited-row'),
    dcc.Store(id='current-order'),
    dcc.Store(id='Current-coefficients'),
    dcc.Store(id='Current-mortality'),
    dcc.Store(id='Current-bounds'),

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
                    html.Br(),
                    html.Div(id='data-status', children='🟡 Patients data is not uploaded.'),
                    html.Br(),
                    html.Div(id='upload-data-status')
                ])
            ], className="shadow-sm mb-4"),
            dbc.Card([
                dbc.CardHeader(html.H5("Upload Calibration Data", className="mb-0")),
                dbc.CardBody([
                    dbc.Button(
                        "Download Example Rows",
                        id="btn-download-calibration-example",
                        color="secondary",
                        className="w-100 mb-3"
                    ),
                    dcc.Download(id="download-calibration-example-csv"),
                    dcc.Upload(
                        id='upload-calibration-data',
                        children=html.Div(['Click to upload patients CSV']),
                        style={
                            'width': '100%', 'height': '80px', 'lineHeight': '80px',
                            'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                            'textAlign': 'center', 'backgroundColor': '#f8f9fa'
                        },
                        multiple=False
                    ),
                    html.Br(),
                    html.Div(id='calibration-status', children='🟡 Prediction interval is not applied.'),
                    html.Br(),
                    html.Div(id='upload-calibration-status')
                ])
            ], className="shadow-sm mb-4"),           

            dbc.Card([
                dbc.CardHeader(html.H5("Instructions", className="mb-0")),
                dbc.CardBody([
                    html.P("1. Click 'Download Example Rows' to get a sample CSV format."),
                    html.P("2. Upload a CSV file with 68 numeric columns representing patient data."),
                    html.P("3. (Optional) Upload calibration data to enable conformal prediction intervals for mortality risk."),
                    html.P("4. View mortality risk predictions after uploading data."),
                    html.P("5. Click on a row in the patients data table to view:"),
                    html.Ul([
                        html.Li("Predicted cumulative mortality over 10 years"),
                        html.Li("Predicted trajectories of 12 risk factors over 3 years"),
                        html.Li("SHAP Analysis: Important baseline variables contributing to predicted 5-year mortality"),
                    ]),
                    html.P("6. Edit patient features and click 'Update Analysis' to see updated results and intervals.")
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
                        children=html.Div(id='output')
                    ),
                    
              
                    html.Div(
                        id='alpha-container',
                        children=[
                            html.Label("Select Conformal Prediction Level (1 - alpha):"),
                            dcc.Slider(
                                id='alpha-slider',
                                min=0.01,
                                max=0.5,
                                step=0.01,
                                value=0.05,
                                marks={
                                    0.01: '99%', 0.05: '95%', 0.1: '90%',
                                    0.2: '80%', 0.3: '70%', 0.5: '50%'
                                },
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ],
                        style={"display": "none"}  
                    ),
                    dcc.Loading(
                        id='loading-mortality',
                        type='circle',
                        children=html.Div(id='mortality-plot')
                    ),
              
                    dcc.Loading(
                        id='loading-trajectory',
                        type='circle',
                        children=html.Div(id='trajectory-plot')
                    ),
           
                    # SHAP analysis
                    dcc.Loading(
                        id='loading-shap',
                        type='circle',
                        children=html.Div(id='shap-plot')
                    ),
           
                    html.Div(id='feature-editor'),
                    dbc.Button("Update Analysis", id='update-shap-button', color="primary", className="mt-2",style={"display": "none"}),
           
                    html.Div(
                        id='alpha-container-updated',
                        children=[
                            html.Label("Select Conformal Prediction Level (1 - alpha):"),
                            dcc.Slider(
                                id='alpha-slider-updated',
                                min=0.01,
                                max=0.5,
                                step=0.01,
                                value=0.05,
                                marks={
                                    0.01: '99%', 0.05: '95%', 0.1: '90%',
                                    0.2: '80%', 0.3: '70%', 0.5: '50%'
                                },
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ],
                        style={"display": "none"}  
                    ),
                    dcc.Loading(
                        id='loading-update_mortality',
                        type='circle',
                        children=html.Div(id='mortality-plot-updated')
                    ),    
          
                    dcc.Loading(
                        id='loading-update_trajectory',
                        type='circle',
                        children=html.Div(id='trajectory-plot-updated')
                    ),      
                
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
                "Model powered by Multi-Task Deephit v2.14",
                html.Br(),
                "App designed by Gefei Lin",
                html.Br(),
                "Version 1.2.2"
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
    
    # if df_calibration does not have Event_status and Event_time columns, return an error message
    if 'Event_status' not in df_calibration.columns or 'Event_time' not in df_calibration.columns:
        return dash.no_update, html.Span("Error: Calibration data must contain 'Event_status' and 'Event_time' columns.", style={"color": "red"}), html.Span("🔴 Prediction interval is not applied.", style={"color": "red"})

    max_event_time = df_calibration['Event_time'].max()
    if max_event_time <16:
        return dash.no_update, html.Span("Error: 'Event_time' in calibration data must be provided in units of days.", style={"color": "red"}), html.Span("🔴 Prediction interval is not applied.", style={"color": "red"})

    df_calibration_no_outcome = df_calibration.drop(columns=['Event_status','Event_time'], errors='ignore')
    
    calibration_example_no_outcome = calibration_example.drop(columns=['Event_status','Event_time'], errors='ignore')
    calibration_example_columns = calibration_example_no_outcome.columns.tolist()
    # Check if the columns in df_calibration_no_outcome match those in calibration_example_no_outcome, and their ordering.
    if not all(col in calibration_example_columns for col in df_calibration_no_outcome.columns):
        return dash.no_update, html.Span("Error: Calibration data columns do not match the example data.", style={"color": "red"}), html.Span("🔴 Prediction interval is not applied.", style={"color": "red"})
    
    # check the order of columns in df_calibration_no_outcome
    if not all(df_calibration_no_outcome.columns[i] == calibration_example_no_outcome.columns[i] for i in range(len(df_calibration_no_outcome.columns))):
        return dash.no_update, html.Span("Error: Calibration data columns are not in the correct order.", style={"color": "red"}), html.Span("🔴 Prediction interval is not applied.", style={"color": "red"})

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
    

    return {'data_calibration_scaled_with_mask': data_calibration_scaled_with_mask.tolist(), 'Event_time_calibration': Event_time, 'Event_status_calibration': Event_status},html.Span(f"{filename} uploaded ✔️", style={"color": "green"}), html.Span("🟢 Prediction interval is enabled.", style={"color": "green"})

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
    
    if not all(col in X_example.columns for col in df.columns):
        return  dash.no_update, dash.no_update,  html.Span("🔴 Patients data is not uploaded.", style={"color": "red"}),html.Div("Error: Patients data columns do not match the example data.", style={"color": "red"})

    if not all(df.columns[i] == X_example.columns[i] for i in range(len(df.columns))):
        return dash.no_update, dash.no_update, html.Span("🔴 Patients data is not uploaded.", style={"color": "red"}), html.Div("Error: Patients data columns are not in the correct order.", style={"color": "red"})


    if df.shape[1] != 68:
        return dash.no_update, dash.no_update, html.Span("🔴 Patients data is not uploaded.", style={"color": "red"}), html.Div("Please make sure the patients data contains exactly 68 column of numeric values.", style={"color": "red"})

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
    return html.Div([
        html.H5(f"Predictions from: {filename}"),
        html.Div([
            html.Div([
                dash_table.DataTable(
                    id='x-table',
                    data=df_features.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in df_features.columns],
                    page_action='none',
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
                    columns = [
                        {'name': col, 'id': col} if col == 'Patient ID'
                        else {'name': col, 'id': col, 'type': 'numeric', 'format': Format(precision=5, scheme=Scheme.fixed)}
                        for col in pred_df.columns
                    ],
                    page_action='none',
                    style_table={'height': 'auto', 'overflowX': 'auto'},
                    style_cell={'minWidth': '100px', 'whiteSpace': 'normal'}
                )
            ], style={
                'height': '400px', 'overflowY': 'scroll', 'overflowX': 'auto',
                'width': '40%', 'display': 'inline-block'
            })
        ]),html.Br(),
        dbc.Button("Download Predicted Mortality Table", id='download-mortality-button', color="primary", className="mt-2",style={"display": "none"}), dcc.Download(id="download-mortality-table-csv"),
        html.Br(),html.Br(),
        html.Div("Click on a row to view mortality plot, trajectories of risk factors, and important baseline variables at individual level for that patient."),
        html.Hr()
    ]), {'df_features': df.to_dict('records'), 'pred_df': pred_df.to_dict('records'), 'scaled_df': df_scaled.to_dict('records'), 'mask': mask.values.tolist(), 'coefficients': coefficients_np},  html.Span("🟢 Patients data is uploaded.", style={"color": "green"}), html.Span(f"{filename} uploaded ✔️", style={"color": "green"})

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
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',name=f'Patient {i+1}'))

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
        

    fig.update_layout(title=f'Mortality Risk for Patient {i+1}',
                      xaxis_title='Year',
                      yaxis_title='Cumulative Mortality',
                      template='plotly_white',
                      xaxis=dict(tickmode='linear', dtick=1),
                      yaxis=dict(range=[-0.01, 1.01]))
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
            html.Div("You can edit the baseline variable values below and click 'Update Analysis' to see the updated results."),
            html.Hr()
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
    fig.add_trace(go.Scatter(x=x, y=y_current, mode='lines+markers',name='Original',line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',name='Updated',line=dict(color='red')))
    
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



    fig.update_layout(title=f'Updated Mortality Risk for Modified Patient {index+1}',
                      xaxis_title='Year',
                      yaxis_title='Cumulative Mortality',
                      template='plotly_white',
                      xaxis=dict(tickmode='linear', dtick=1),
                      yaxis=dict(range=[-0.01, 1.01]))
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
        html.H5(f"Updated SHAP Waterfall Plot for Modified Patient {index+1}"),
        html.Img(src=img, style={'maxWidth': '100%', 'height': 'auto', 'border': '1px solid lightgray'}),
        html.Hr()
    ])



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  
    app.run(debug=False, host='0.0.0.0', port=port)

