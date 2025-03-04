import sys
import os


# Calculate relative path to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, '..'))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
# Add project root to Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)

"""
Dash callbacks for DSSAT Viewer
"""
import os
import traceback
import pandas as pd
import numpy as np
import logging
import dash
from dash import callback_context
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import project modules
from data.data_processing import unified_date_convert

import config
from utils.dssat_paths import get_crop_details, prepare_folders
from data.dssat_io import (
    prepare_experiment, prepare_treatment, prepare_out_files, 
    read_file, read_observed_data, read_evaluate_file,
    create_batch_file, run_treatment
)
from data.data_processing import (
    handle_missing_xvar, get_variable_info, improved_smart_scale,
    get_evaluate_variable_pairs, get_all_evaluate_variables
)
from models.metrics import MetricsCalculator
from data.visualization import create_plot

logger = logging.getLogger(__name__)

def register_callbacks(app):
    """Register all callbacks with the Dash app."""
    
    @app.callback(
        [
            Output("sidebar", "is_open"),
            Output("visualization-container", "style"),
            Output("toggle-sidebar", "style"),
            Output("sidebar-state", "data"),
        ],
        [Input("toggle-sidebar", "n_clicks"), Input("close-sidebar", "n_clicks")],
        [State("sidebar", "is_open")],
    )
    def toggle_sidebar(toggle_clicks, close_clicks, is_open):
        base_viz_style = {
            "height": "100vh",
            "width": "80%",
            "padding": "30px",
            "transition": "margin-left 0.3s ease-in-out",
        }
        base_toggle_style = {
            "position": "fixed",
            "top": "13px",
            "z-index": "1001",
            "transition": "left 0.3s ease-in-out",
        }
        new_is_open = is_open
        if toggle_clicks or close_clicks:
            new_is_open = not is_open if toggle_clicks else False
            if new_is_open:
                viz_style = {**base_viz_style, "margin-left": "20%", "width": "80%"}
                toggle_style = {**base_toggle_style, "left": "calc(20% - 50px)"}
            else:
                viz_style = {**base_viz_style, "margin-left": "0", "width": "100%"}
                toggle_style = {**base_toggle_style, "left": "20px"}
            return new_is_open, viz_style, toggle_style, {"is_open": new_is_open}
        viz_style = {
            **base_viz_style,
            "width": "80%" if is_open else "100%",
            "margin-left": "20%" if is_open else "0",
        }
        toggle_style = {
            **base_toggle_style,
            "left": "calc(20% - 50px)" if is_open else "20px",
        }
        return is_open, viz_style, toggle_style, {"is_open": new_is_open}
    
    @app.callback(
        [Output("time-series-controls", "style"),
         Output("scatter-controls", "style")],
        [Input("viz-tabs", "active_tab")]
    )
    def toggle_control_sections(active_tab):
        if active_tab == "scatter":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    @app.callback(
        Output("preview-modal", "is_open"),
        [
            Input("open-preview", "n_clicks"),
            Input("close-preview-modal", "n_clicks"),
        ],
        [State("preview-modal", "is_open")],
    )
    def toggle_preview_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open
        
    @app.callback(
        Output("metrics-modal", "is_open"),
        [
            Input("open-metrics", "n_clicks"),
            Input("close-metrics-modal", "n_clicks"),
        ],
        [State("metrics-modal", "is_open")],
    )
    def toggle_metrics_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open
        
    @app.callback(
        [Output("folder-selector", "options"), Output("folder-selector", "value")],
        Input("folder-selector", "id"),
    )
    def initial_folder_load(_):
        folders = prepare_folders()
        options = [{"label": folder, "value": folder} for folder in folders]
        default_value = folders[0] if folders else None
        return options, default_value
    
    @app.callback(
        [
            Output("experiment-selector", "options"),
            Output("experiment-selector", "value"),
            Output("treatment-selector", "options"),
            Output("treatment-selector", "value"),
        ],
        [Input("folder-selector", "value"), Input("experiment-selector", "value")],
    )
    def update_experiment_and_treatment(
        selected_folder: str, selected_experiment: str
    ):
        ctx = callback_context
        triggered_id = (
            ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        )
        empty_return = [], None, [], None
        if not selected_folder:
            return empty_return
        try:
            if triggered_id == "folder-selector":
                experiments = prepare_experiment(selected_folder)
                # Create options with display name as label but filename as value
                exp_options = [{"label": exp_name, "value": filename} for exp_name, filename in experiments]
                selected_exp = experiments[0][1] if experiments else None  # Use filename as value
                if selected_exp:
                    treatments = prepare_treatment(selected_folder, selected_exp)
                    if treatments is not None and not treatments.empty:
                        treatments["TR"] = treatments["TR"].astype(str)
                        trt_options = [
                            {"label": f"{row.TR} - {row.TNAME}", "value": row.TR}
                            for _, row in treatments.iterrows()
                        ]
                        default_values = treatments["TR"].tolist()
                        return (
                            exp_options,
                            selected_exp,
                            trt_options,
                            default_values,
                        )
                    return exp_options, selected_exp, [], None
                return exp_options, None, [], None
            elif triggered_id == "experiment-selector":
                if not selected_experiment:
                    return dash.no_update, dash.no_update, [], None
                treatments = prepare_treatment(selected_folder, selected_experiment)
                if treatments is not None and not treatments.empty:
                    treatments["TR"] = treatments["TR"].astype(str)
                    trt_options = [
                        {"label": f"{row.TR} - {row.TNAME}", "value": row.TR}
                        for _, row in treatments.iterrows()
                    ]
                    default_values = treatments["TR"].tolist()
                    return (
                        dash.no_update,
                        dash.no_update,
                        trt_options,
                        default_values,
                    )
                return dash.no_update, dash.no_update, [], None
            return empty_return
        except Exception as e:
            logger.error(f"Error in update_experiment_and_treatment: {e}")
            return empty_return

    @app.callback(
        [
            Output("success-message", "is_open"),
            Output("success-message", "children"),
            Output("error-message", "is_open"),
            Output("error-message", "children"),
            Output("execution-status", "data"),
            Output("loading-treatment-run", "children"),
            Output("refresh-plot-button", "n_clicks"),
        ],
        [Input("run-button", "n_clicks")],
        [
            State("folder-selector", "value"),
            State("treatment-selector", "value"),
            State("experiment-selector", "value"),
        ],
    )
    def handle_run_treatment(
        n_clicks, selected_folder, selected_treatments, selected_experiment
    ):
        if not n_clicks:
            return False, "", False, "", {"completed": False}, None, 0
        try:
            if not all([selected_folder, selected_treatments, selected_experiment]):
                return (
                    False,
                    "",
                    True,
                    "Please select all required fields before running.",
                    {"completed": False},
                    None,
                    0,
                )
            crop_details = get_crop_details()
            if not any(crop['name'].upper() == selected_folder.upper() for crop in crop_details):
                return (
                    False,
                    "",
                    True,
                    f"Invalid crop folder: {selected_folder}",
                    {"completed": False},
                    None,
                    0,
                )
            input_data = {
                "folders": selected_folder,
                "executables": config.DSSAT_EXE,
                "experiment": selected_experiment,
                "treatment": selected_treatments,
            }
            batch_file_path = create_batch_file(input_data, config.DSSAT_BASE)
            run_treatment(input_data, config.DSSAT_BASE)
            treatment_str = (
                ", ".join(str(t) for t in selected_treatments)
                if isinstance(selected_treatments, list)
                else str(selected_treatments)
            )
            success_msg = f"Treatment(s) {treatment_str} executed successfully! Please select an output file to view results."
            return True, success_msg, False, "", {"completed": True}, None, 1
        except Exception as e:
            error_msg = f"Error executing treatment: {str(e)}"
            logger.error(f"Error in handle_run_treatment: {error_msg}")
            return False, "", True, error_msg, {"completed": False}, None, 0

    # Add this modification to the update_output_files callback in callbacks.py

    @app.callback(
        [
            Output("out-file-selector", "options"),
            Output("out-file-selector", "value"),
            Output("execution-status", "data", allow_duplicate=True),  # Add this output with allow_duplicate
        ],
        [
            Input("run-button", "n_clicks"),
            Input("treatment-selector", "value"),
            Input("execution-status", "data"),
            Input("folder-selector", "value"),
            Input("experiment-selector", "value"),
        ],
        [State("folder-selector", "value"), State("experiment-selector", "value")],
        prevent_initial_call=True  # Add this to prevent initial callback
    )
    def update_output_files(
        n_clicks,
        selected_treatments,
        execution_status,
        selected_folder_input,
        selected_experiment_input,
        selected_folder_state,
        selected_experiment_state,
    ):
        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        
        # Reset execution status when folder or experiment changes
        if triggered_id in ["folder-selector", "experiment-selector"]:
            return [], [], {"completed": False}
            
        # For other cases, maintain current behavior but return execution status unchanged
        if not execution_status or not execution_status.get("completed", False):
            return [], [], dash.no_update
            
        if not selected_folder_input:
            return [], [], dash.no_update
            
        try:
            out_files = prepare_out_files(selected_folder_input)
            logger.info(f"Output files found: {out_files}")
            if not out_files:
                return [], [], dash.no_update
                
            file_options = [{"label": file, "value": file} for file in out_files]
            selected_out_files = (
                ["PlantGro.OUT"] if "PlantGro.OUT" in out_files else [out_files[0]]
            )
            return file_options, selected_out_files, dash.no_update
        except Exception as e:
            logger.error(f"Error updating output files: {e}")
            return [], [], dash.no_update
                
    @app.callback(
        [
            Output("x-var-selector", "options"),
            Output("x-var-selector", "value"),
            Output("y-var-selector", "options"),
            Output("y-var-selector", "value"),
            Output("refresh-plot-button", "disabled"),
        ],
        [Input("out-file-selector", "value")],
        [State("folder-selector", "value"), State("y-var-selector", "value")],
    )
    def update_variables(selected_out_files, selected_folder, current_y_vars):
        """Update variable selectors based on selected OUT files."""
        if not selected_out_files or not selected_folder:
            return [], None, [], None, True
            
        try:
            # Get crop directory
            crop_details = get_crop_details()
            crop_info = next(
                (crop for crop in crop_details 
                if crop['name'].upper() == selected_folder.upper()),
                None
            )
            
            if not crop_info:
                logger.error(f"Could not find crop info for: {selected_folder}")
                return [], None, [], None, True

            # Collect columns from all selected files
            all_columns = set()
            for out_file in selected_out_files:
                file_path = os.path.join(crop_info['directory'], out_file)
                logger.info(f"Reading file: {file_path}")
                
                data = read_file(file_path)
                if data is not None and not data.empty:
                    all_columns.update(
                        col for col in data.columns 
                        if col not in ["TRT", "FILEX"]
                    )

            # Create variable options with labels
            var_options = []
            for col in sorted(all_columns):
                var_label, description = get_variable_info(col)
                var_options.append({
                    "label": var_label if var_label else col,
                    "value": col,
                })

            # Set default values
            default_x = "DATE" if "DATE" in all_columns else None
            default_y = []
            
            if "CWAD" in all_columns:
                default_y = ["CWAD"]
            elif current_y_vars:
                available_vars = [opt["value"] for opt in var_options]
                default_y = [var for var in current_y_vars 
                            if var in available_vars]
            elif var_options:
                default_y = [var_options[0]["value"]]

            disable_refresh = len(default_y) < 1
            
            return (
                var_options,    # x-var options
                default_x,      # x-var default
                var_options,    # y-var options
                default_y,      # y-var defaults
                disable_refresh # refresh button state
            )

        except Exception as e:
            logger.error(f"Error updating variables: {e}")
            return [], None, [], None, True
            
    @app.callback(
        [
            Output("data-plot", "figure"),
            Output("data-table", "data", allow_duplicate=True),
            Output("data-table", "columns", allow_duplicate=True),
            Output("metrics-table", "data", allow_duplicate=True),
            Output("metrics-table", "columns", allow_duplicate=True),
        ],
        [
            Input("refresh-plot-button", "n_clicks"),
            Input("treatment-selector", "value"),
            Input("sidebar-state", "data"),
            
            Input("out-file-selector", "value"),  # Add this input to watch for changes
            Input("x-var-selector", "value"),     # Add this input to watch for changes
            Input("y-var-selector", "value"),     # Add this input to watch for changes
            Input("execution-status", "data"),    # Add this to watch execution status
        ],
        [
            State("viz-tabs", "active_tab"), 
            State("x-var-selector", "value"),
            State("y-var-selector", "value"),
            State("folder-selector", "value"),
            State("out-file-selector", "value"),
            State("experiment-selector", "value"),
            State("treatment-selector", "options"),
            State("sidebar", "is_open"),
        ],
        prevent_initial_call=True,
    )
    def update_timeseries_plot(
        n_clicks,
        selected_treatments,
        sidebar_data,
        out_files_input,       # New input
        x_var_input,           # New input
        y_var_input,           # New input
        execution_status, 
        active_tab,     # New input
        x_var,
        y_var,
        selected_folder,
        selected_out_files,
        selected_experiment,
        treatment_options,
        is_open,
    ):
        # Define empty returns
        empty_fig = go.Figure()
        empty_table_data = []
        empty_table_columns = []
        empty_metrics_data = []
        empty_metrics_columns = []
        
        # Check if we have execution_status and required data
        if not execution_status or not execution_status.get("completed", False):
            return (
                empty_fig,
                empty_table_data,
                empty_table_columns,
                empty_metrics_data,
                empty_metrics_columns,
            )
        
        # Check if we have all required data for time series tab
        # Only process when on time-series tab
        if active_tab != "time-series":
            return empty_fig, empty_table_data, empty_table_columns, empty_metrics_data, empty_metrics_columns         
        try:
            all_data = []
            y_vars = y_var if isinstance(y_var, list) else [y_var]
            
            for selected_out_file in selected_out_files:
                file_path = os.path.join(
                    config.DSSAT_BASE, selected_folder, selected_out_file
                )
                sim_data = read_file(file_path)
                if sim_data is None or sim_data.empty:
                    continue
                    
                sim_data.columns = sim_data.columns.str.strip().str.upper()
                
                if "TRNO" in sim_data.columns and "TRT" not in sim_data.columns:
                    sim_data["TRT"] = sim_data["TRNO"]
                elif "TRT" not in sim_data.columns:
                    sim_data["TRT"] = "1"
                    
                sim_data["TRT"] = sim_data["TRT"].astype(str)
                
                for col in ["YEAR", "DOY"]:
                    if col in sim_data.columns:
                        sim_data[col] = (
                            pd.to_numeric(sim_data[col], errors="coerce")
                            .fillna(0)
                            .replace([np.inf, -np.inf], 0)
                        )
                    else:
                        sim_data[col] = 0
                        
                sim_data["DATE"] = sim_data.apply(
                    lambda row: unified_date_convert(row["YEAR"], row["DOY"]),
                    axis=1,
                )
                sim_data["DATE"] = sim_data["DATE"].dt.strftime("%Y-%m-%d")
                sim_data["source"] = "sim"
                sim_data["FILE"] = selected_out_file
                all_data.append(sim_data)
                
            if not all_data:
                return (
                    empty_fig,
                    empty_table_data,
                    empty_table_columns,
                    empty_metrics_data,
                    empty_metrics_columns,
                )
                
            sim_data = pd.concat(all_data, ignore_index=True)
            missing_values = {-99, -99.0, -99.9, -99.99}
            
            # Read observed data
            obs_data = None
            if selected_experiment:
                obs_data = read_observed_data(
                    selected_folder, selected_experiment, x_var, y_vars
                )
                if obs_data is not None and not obs_data.empty:
                    obs_data["source"] = "obs"
                    obs_data = handle_missing_xvar(obs_data, x_var, sim_data)
                    
                    if obs_data is not None:
                        if "TRNO" in obs_data.columns:
                            obs_data["TRNO"] = obs_data["TRNO"].astype(str)
                            obs_data = obs_data.rename(columns={"TRNO": "TRT"})
                            
                        for var in y_vars:
                            if var in obs_data.columns:
                                obs_data[var] = pd.to_numeric(
                                    obs_data[var], errors="coerce"
                                )
                                obs_data.loc[
                                    obs_data[var].isin(missing_values), var
                                ] = np.nan
                                obs_data = obs_data.rename(columns={var: f"{var}"})
                                
            # Calculate metrics
            hover_texts = {}
            metrics_data = []
            
            for selected_out_file in selected_out_files:
                file_sim_data = sim_data[sim_data["FILE"] == selected_out_file]
                
                if obs_data is not None and not obs_data.empty:
                    unique_treatments_obs = obs_data["TRT"].unique()
                    unique_treatments_sim = file_sim_data["TRT"].unique()
                    common_treatments = set(unique_treatments_obs) & set(
                        unique_treatments_sim
                    )
                    
                    for var in y_vars:
                        if var in obs_data.columns and var in file_sim_data.columns:
                            for treatment in common_treatments:
                                filtered_obs_data = obs_data[
                                    obs_data["TRT"] == treatment
                                ]
                                filtered_sim_data = file_sim_data[
                                    file_sim_data["TRT"] == treatment
                                ]
                                
                                if (
                                    not filtered_obs_data.empty
                                    and not filtered_sim_data.empty
                                ):
                                    common_dates = filtered_sim_data["DATE"].isin(
                                        filtered_obs_data["DATE"]
                                    )
                                    filtered_sim = filtered_sim_data[common_dates]
                                    sim_values = filtered_sim[var].to_numpy()
                                    obs_values = filtered_obs_data[var].to_numpy()
                                    
                                    if len(obs_values) > 0 and len(sim_values) > 0:
                                        var_metrics = (
                                            MetricsCalculator.calculate_metrics(
                                                sim_values, obs_values, treatment
                                            )
                                        )
                                        
                                        if var_metrics is not None:
                                            treatment_names = {
                                                opt["value"]: opt["label"]
                                                for opt in treatment_options
                                            }
                                            var_label, var_description = (
                                                get_variable_info(var)
                                            )
                                            rounded_n = round(var_metrics["n"])
                                            rounded_rmse = (
                                                round(var_metrics["RMSE"], 2)
                                                if var_metrics["RMSE"] is not None
                                                else None
                                            )
                                            rounded_nrmse = (
                                                round(var_metrics["NRMSE"], 2)
                                                if var_metrics["NRMSE"] is not None
                                                else None
                                            )
                                            rounded_d_stat = (
                                                round(
                                                    var_metrics[
                                                        "Willmott's d-stat"
                                                    ],
                                                    2,
                                                )
                                                if var_metrics["Willmott's d-stat"]
                                                is not None
                                                else None
                                            )

                                            metrics_data.append(
                                                {
                                                    "Treatment": f"{treatment_names.get(str(treatment), f'T{treatment}')}",
                                                    "Variable": var_label,
                                                    "n": rounded_n,
                                                    "RMSE": rounded_rmse,
                                                    "NRMSE": rounded_nrmse,
                                                    "Willmott's d-stat": rounded_d_stat,
                                                }
                                            )
            
            metrics_columns = [
                {"name": i, "id": i}
                for i in [
                    "Treatment",
                    "Variable",
                    "n",
                    "RMSE",
                    "NRMSE",
                    "Willmott's d-stat",
                ]
            ]
            
            # Scale data for visualization
            sim_scaling_factors = {}
            for var in y_vars:
                if var in sim_data.columns:
                    sim_values = (
                        pd.to_numeric(sim_data[var], errors="coerce")
                        .dropna()
                        .values
                    )
                    var_min, var_max = np.min(sim_values), np.max(sim_values)

                    if np.isclose(var_min, var_max):
                        midpoint = (10000 + 1000) / 2
                        sim_scaling_factors[var] = (1, midpoint)
                    else:
                        scale_factor = (10000 - 1000) / (var_max - var_min)
                        offset = 1000 - var_min * scale_factor
                        sim_scaling_factors[var] = (scale_factor, offset)
                        
            sim_scaled = improved_smart_scale(
                sim_data, y_vars, scaling_factors=sim_scaling_factors
            )
            
            for var in sim_scaled:
                sim_data[f"{var}_original"] = sim_data[var]
                sim_data[var] = sim_scaled[var]
                
            if obs_data is not None and not obs_data.empty:
                obs_scaled = improved_smart_scale(
                    obs_data, y_vars, scaling_factors=sim_scaling_factors
                )
                for var in obs_scaled:
                    obs_data[f"{var}_original"] = obs_data[var]
                    obs_data[var] = obs_scaled[var]
                    
            # Create scaling text for annotation
            scaling_text = "<br>".join(
                [
                    f"{var_label} = {round(scale_factor, 6):.6f} * {var_label} + {round(offset, 2):.2f}"
                    for var, (scale_factor, offset) in sim_scaling_factors.items()
                    for var_label, _ in [get_variable_info(var)]
                ]
            )
            
            # Create figure
            fig = go.Figure()
            line_styles = config.LINE_STYLES
            marker_symbols = config.MARKER_SYMBOLS
            colors = config.PLOT_COLORS
            
            treatment_names = {
                opt["value"]: opt["label"] for opt in treatment_options
            }
            
            for dataset in [sim_data, obs_data]:
                if dataset is not None and not dataset.empty:
                    for var in y_vars:
                        for trt_value, group in dataset.groupby("TRT"):
                            if (
                                trt_value in selected_treatments
                                and var in group.columns
                                and group[var].notna().any()
                            ):
                                source_type = group["source"].iloc[0]
                                legend_group = (
                                    "Simulated Data"
                                    if source_type == "sim"
                                    else "Observed Data"
                                )
                                var_index = y_vars.index(var) % len(line_styles)
                                treatment_index = int(trt_value) % len(colors)
                                color = colors[treatment_index]
                                line_style = line_styles[var_index]
                                marker_symbol = marker_symbols[
                                    (var_index + treatment_index)
                                    % len(marker_symbols)
                                ]
                                var_label, var_description = get_variable_info(var)
                                name_label = f"{var_label} | {treatment_names.get(str(trt_value), f'T{trt_value}')}"
                                fig.add_trace(
                                    go.Scatter(
                                        x=group[group[var].notna()][x_var],
                                        y=group[group[var].notna()][var],
                                        mode=(
                                            "markers"
                                            if source_type == "obs"
                                            else "lines"
                                        ),
                                        name=name_label,
                                        line=dict(
                                            dash=line_style, width=2, color=color
                                        ),
                                        marker=(
                                            dict(
                                                symbol=marker_symbol,
                                                size=8,
                                                color=color,
                                            )
                                            if source_type == "obs"
                                            else None
                                        ),
                                        legendgroup=legend_group,
                                        legendgrouptitle_text=legend_group,
                                        hoverlabel=dict(namelength=-1),
                                    )
                                )
            
            # Add scaling annotation
            annotation_x = (
                1 if sidebar_data and sidebar_data.get("is_open", True) else 1
            )
            
            # Update layout
            fig.update_layout(
                title="",
                xaxis_title={"text": x_var, "font": {"size": 18, "color": "blue"}},
                yaxis_title={
                    "text": ", ".join(
                        get_variable_info(var)[0]
                        for var in y_vars
                        if var in sim_data.columns
                    ),
                    "font": {"size": 18, "color": "blue"},
                },
                xaxis=dict(
                    title_font=dict(size=18),
                    tickfont=dict(size=14),
                    showgrid=True,
                    gridcolor="lightgrey",
                    linecolor="black",
                    linewidth=2,
                ),
                yaxis=dict(
                    title_font=dict(size=18),
                    tickfont=dict(size=14),
                    showgrid=True,
                    gridcolor="lightgrey",
                    linecolor="black",
                    linewidth=2,
                ),
                template="ggplot2",
                height=700,
                showlegend=True,
                legend_tracegroupgap=10,
                paper_bgcolor="rgba(245, 245, 245, 1)",
                margin=dict(t=50, b=50, l=50, r=50),
                annotations=[
                    dict(
                        text=f"Scaling Factors:<br>{scaling_text}",
                        x=annotation_x,
                        y=0.1,
                        xref="paper",
                        yref="paper",
                        xanchor="left",
                        yanchor="bottom",
                        showarrow=False,
                        font=dict(
                            size=12, family="Arial, sans-serif", color="black"
                        ),
                        align="center",
                        bgcolor="rgba(255, 255, 255, 0.9)",
                        bordercolor="rgba(0, 0, 0, 0.5)",
                        borderwidth=0,
                    )
                ],
            )
            
            # Prepare data for table preview
            sim_preview_data = sim_data.to_dict("records")
            obs_preview_data = (
                obs_data.to_dict("records")
                if obs_data is not None and not obs_data.empty
                else []
            )
            combined_preview_data = sim_preview_data + obs_preview_data
            sim_columns = [{"name": i, "id": i} for i in sim_data.columns]
            
            return (
                fig,
                combined_preview_data,
                sim_columns,
                metrics_data,
                metrics_columns,
            )
            
        except Exception as e:
            logger.error(f"Error in visualization update: {str(e)}")
            traceback.print_exc()
            return (
                empty_fig,
                empty_table_data,
                empty_table_columns,
                empty_metrics_data,
                empty_metrics_columns,
            )
    
    @app.callback(
        [
            Output("auto-pair-container", "style"),
            Output("custom-xy-container", "style"),
            Output("metrics-table", "data", allow_duplicate=True),
            Output("metrics-table", "columns", allow_duplicate=True),
            Output("scatter-var-selector", "options"),
            Output("scatter-var-selector", "value"),
            Output("scatter-x-var-selector", "options"),
            Output("scatter-x-var-selector", "value"),
            Output("scatter-y-var-selector", "options"),
            Output("scatter-y-var-selector", "value"),
            Output("scatter-plot", "figure"),
            Output("data-table", "data", allow_duplicate=True),
            Output("data-table", "columns", allow_duplicate=True),
        ],
        [
            Input("folder-selector", "value"),
            Input("scatter-mode", "value"),
            Input("scatter-var-selector", "value"),
            Input("scatter-x-var-selector", "value"),
            Input("scatter-y-var-selector", "value"),
            Input("viz-tabs", "active_tab"),
            Input("execution-status", "data"),  # Add execution status as input
        ],
        [
            State("treatment-selector", "options"),
        ],
        prevent_initial_call=True,
    )
    def update_scatter_plot(
        selected_folder, mode, auto_vars, x_var, y_vars, active_tab, execution_status, treatment_options
    ):
        """Update scatter plot based on selected mode and variables."""
        # Initialize empty metrics
        empty_metrics_data = []
        empty_metrics_columns = [
            {"name": i, "id": i}
            for i in [
                "Variable",
                "n",
                "R²", 
                "RMSE",
                "d-stat",
            ]
        ]
        
        # Full empty return
        empty_return = (
            {"display": "none"}, {"display": "none"},
            empty_metrics_data, empty_metrics_columns,
            [], None, [], None, [], None,
            go.Figure(), [], []
        )
        
        # Check if we have execution status and if we're on the scatter tab
        if not selected_folder or active_tab != "scatter":
            return empty_return
            
        # Check if treatment has been run
        if not execution_status or not execution_status.get("completed", False):
            return empty_return
            
        try:
            # Read evaluate data
            evaluate_data = read_evaluate_file(selected_folder)
            if evaluate_data is None or evaluate_data.empty:
                return empty_return

            
            data_preview_columns = [
            {
                "name": col, 
                "id": col, 
                "type": "numeric" if pd.api.types.is_numeric_dtype(evaluate_data[col]) else "text",
                "format": {
                    "specifier": ".3f" if pd.api.types.is_numeric_dtype(evaluate_data[col]) else None
                }
            } for col in evaluate_data.columns
        ]

            # Advanced data preview with processing
            data_preview = []
            for _, row in evaluate_data.iterrows():
                processed_row = {}
                for col in evaluate_data.columns:
                    # Handle NaN values
                    if pd.isna(row[col]):
                        processed_row[col] = 'N/A'
                    # Round numeric columns
                    elif pd.api.types.is_numeric_dtype(evaluate_data[col]):
                        processed_row[col] = round(row[col], 3)
                    else:
                        processed_row[col] = row[col]
                data_preview.append(processed_row)
                
            # Set container visibility based on mode
            auto_style = {"display": "block" if mode == "auto" else "none"}
            custom_style = {"display": "block" if mode == "custom" else "none"}
            
            # Get variable pairs for auto mode
            var_pairs = get_evaluate_variable_pairs(evaluate_data)
            auto_options = [
                {"label": display_name, "value": str((display_name, sim_var, meas_var))}
                for display_name, sim_var, meas_var in var_pairs
            ]
            
            # Get all variables for custom mode
            all_vars = []
            for col in evaluate_data.columns:
                if col not in ['RUN', 'EXCODE', 'TRNO', 'RN', 'CR']:
                    var_label, _ = get_variable_info(col)
                    display_name = var_label if var_label else col
                    all_vars.append({"label": display_name, "value": col})
            
            # Initialize figure and metrics data
            fig = go.Figure()
            metrics_data = []
            
            # Process data for auto mode
            if mode == "auto" and auto_vars:
                # Process auto pairs
                titles = []
                valid_pairs = []
                
                # First check which pairs have valid and meaningful data
                for auto_var in auto_vars:
                    selected_pair = eval(auto_var)
                    display_name, sim_var, meas_var = selected_pair
                    
                    # Get data and drop any rows where either value is NaN
                    valid_data = evaluate_data[[sim_var, meas_var, 'TRNO']].dropna()
                    
                    if not valid_data.empty:
                        # Check if all values are identical
                        if not (valid_data[sim_var] == valid_data[meas_var]).all():
                            titles.append(f"{display_name}")
                            valid_pairs.append((display_name, sim_var, meas_var))
                        else:
                            logger.warning(f"Skipping {display_name} - all simulated and measured values are identical")
                
                # Only create subplots for valid pairs
                n_valid = len(valid_pairs)
                if n_valid == 0:
                    return empty_return 
                    
                fig = go.Figure()
                    
                    # Determine grid layout based on number of valid pairs
                if n_valid == 1:
                        n_rows, n_cols = 1, 1  # 1 plot: full screen
                elif n_valid == 2:
                        n_rows, n_cols = 1, 2  # 2 plots: 1×2 grid
                elif n_valid <= 4:
                        n_rows, n_cols = 2, 2  # 3-4 plots: 2×2 grid
                elif n_valid <= 6:
                        n_rows, n_cols = 3, 3  # 5-6 plots: 2×3 grid
                elif n_valid <= 9:
                        n_rows, n_cols = 3, 3  # 7-9 plots: 3×3 grid
                elif n_valid <= 12:
                        n_rows, n_cols = 4, 4
                else:
                        n_rows, n_cols = 4, 4  # 10+ plots: 4×4 grid
                    
                    # ===== CRITICAL FIX: Create a proper subplot structure =====
                fig = make_subplots(
                        rows=n_rows,
                        cols=n_cols,
                        subplot_titles=titles,
                        horizontal_spacing=0.1,  # Increased spacing
                        vertical_spacing=0.1,    # Increased spacing
                        shared_xaxes=False,       # Important - no shared axes
                        shared_yaxes=False,       # Important - no shared axes
                        print_grid=False          # Don't print the grid to console
                    )
                
                colors = px.colors.qualitative.Set1
                marker_symbols = ["circle", "square", "diamond", "triangle-up", "star", "hexagon"]
                
                # Track if legend has been added for treatment numbers and 1:1 line
                legend_added = set()
                
                # Process each valid pair
                for idx, (display_name, sim_var, meas_var) in enumerate(valid_pairs):
                    if idx >= min(4, (n_valid+1)//2) * min(4, 2):
                        # Skip if we have more pairs than subplot spaces
                        logger.warning(f"Skipping plot {idx+1} ({display_name}) - not enough space in grid")
                        continue
                        
                    # Calculate row and column position
                    row_idx = (idx // n_cols) + 1
                    col_idx = (idx % n_cols) + 1
                    
                    valid_data = evaluate_data[[sim_var, meas_var, 'TRNO']].dropna()
                    
                    # Calculate range for 1:1 line
                    all_vals = pd.concat([valid_data[sim_var], valid_data[meas_var]])
                    min_val = all_vals.min()
                    max_val = all_vals.max()
                    
                    # Add padding and ensure we have a square domain
                    range_span = max_val - min_val
                    pad = max(range_span * 0.1, 0.01)  # 10% padding, minimum of 0.01
                    
                    range_min = min_val - pad
                    range_max = max_val + pad
                    
                    # Add 1:1 line
                    show_legend = '1:1 Line' not in legend_added
                    fig.add_trace(
                        go.Scatter(
                            x=[range_min, range_max],
                            y=[range_min, range_max],
                            mode='lines',
                            line=dict(dash='dash', color='red', width=1),
                            name='1:1 Line',
                            showlegend=show_legend
                        ),
                        row=row_idx, col=col_idx
                    )
                    if show_legend:
                        legend_added.add('1:1 Line')
                    
                    # Calculate statistics for all data points combined (not by treatment)
                    sim_values = valid_data[sim_var].to_numpy()
                    meas_values = valid_data[meas_var].to_numpy()
                    
                    # Calculate statistics text for the plot
                    correlation_matrix = np.corrcoef(sim_values, meas_values)
                    r2 = correlation_matrix[0, 1] ** 2
                    rmse = MetricsCalculator.rmse(meas_values, sim_values)
                    d_stat = MetricsCalculator.d_stat(meas_values, sim_values)
                    
                    stats_text = (
                        f'n = {len(meas_values)}<br>'
                        f'R² = {r2:.3f}<br>'
                        f'RMSE = {rmse:.3f}<br>'
                        f'd-stat = {d_stat:.3f}'
                    )
                    
                    # Add to metrics data for the table
                    metrics_data.append({
                        "Variable": display_name,
                        "n": len(meas_values),
                        "R²": round(r2, 3),
                        "RMSE": round(rmse, 3),
                        "d-stat": round(d_stat, 3) if d_stat is not None else None,
                    })
                    
                    # Add scatter plots for each treatment
                    for j, trno in enumerate(sorted(valid_data['TRNO'].unique())):
                        trno_data = valid_data[valid_data['TRNO'] == trno]
                        legend_key = f"Treatment {trno}"
                        show_legend = legend_key not in legend_added
                        
                        fig.add_trace(
                            go.Scatter(
                                x=trno_data[sim_var],
                                y=trno_data[meas_var],
                                
                                mode='markers',
                                name=legend_key,
                                marker=dict(
                                    size=10,
                                    color=colors[j % len(colors)],
                                    symbol=marker_symbols[j % len(marker_symbols)]
                                ),
                                showlegend=show_legend,
                                legendgroup=f"treatment_{trno}",
                                hovertemplate=(
                                    f"Treatment {trno}<br>" +
                                    f"Simulated: %{{x:.2f}}<br>" +
                                    f"Measured: %{{y:.2f}}<br>" +
                                    "<extra></extra>"
                                )
                            ),
                            row=row_idx, col=col_idx
                        )
                        
                        if show_legend:
                            legend_added.add(legend_key)
                    
                    # Update axes to force square shape
                    fig.update_yaxes(
                        title_text="Measured",
                        range=[range_min, range_max],
                        row=row_idx, 
                        col=col_idx,
                        gridcolor='lightgray',
                        scaleanchor=f"y{row_idx}{col_idx}",
                        scaleratio=1,
                        constrain="domain",
                        matches=None
                    )
                    
                    fig.update_xaxes(
                        title_text="Simulated",
                        range=[range_min, range_max],
                        row=row_idx, 
                        col=col_idx,
                        gridcolor='lightgray',
                        constrain="domain",
                        matches=None
                    )
                    
                    # Add title at the top of each subplot
                    fig.add_annotation(
                        text=f"{display_name}",
                        xref=f"x{idx+1}", 
                        yref=f"y{idx+1}",
                        x=0.5,
                        y=1.05,
                        xanchor='center', 
                        yanchor='bottom',
                        showarrow=False,
                        font=dict(size=12, color='black'),
                        borderwidth=0
                    )
                
                # Set figure dimensions based on the layout
                base_size = 420  # Base size for each subplot
                
                # Calculate width and height
                width = base_size * min(4, 2) + 150  # Add space for legend
                height = base_size * min(4, (n_valid+1)//2) + 100  # Add space for title
                
                # Set minimum dimensions
                width = max(1200, width)
                height = max(600, height)
                
                # Update layout
                fig.update_layout(
                    width=width,
                    height=height,
                    template="plotly_white",
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.02,
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='rgba(0, 0, 0, 0.2)',
                        borderwidth=1
                    ),
                    margin=dict(t=80, b=60, l=60, r=60)
                )
                    
            elif mode == "custom" and x_var and y_vars:
                # Handle custom X-Y mode with multiple Y variables
                for y_var in y_vars:
                    valid_data = evaluate_data[[x_var, y_var, 'TRNO']].dropna()
                    if not valid_data.empty:
                        # Get variable labels
                        x_label, _ = get_variable_info(x_var)
                        y_label, _ = get_variable_info(y_var)
                        display_name = f"{y_label or y_var} vs {x_label or x_var}"
                        
                        # Calculate statistics for all data points combined (not by treatment)
                        x_values = valid_data[x_var].to_numpy()
                        y_values = valid_data[y_var].to_numpy()
                        
                        # Calculate statistics for the plot and metrics table
                        correlation_matrix = np.corrcoef(x_values, y_values)
                        r2 = correlation_matrix[0, 1] ** 2

                        # Use your own RMSE function
                        rmse = MetricsCalculator.rmse(y_values, x_values)
                        d_stat = MetricsCalculator.d_stat(y_values, x_values)
                        
                        stats_text = (
                            f'n = {len(x_values)}<br>'
                            f'R² = {r2:.3f}<br>'
                            f'RMSE = {rmse:.3f}<br>'
                            f'd-stat = {d_stat:.3f}'
                        )
                        
                        # Add to metrics data for the table
                        metrics_data.append({
                            "Variable": display_name,
                            "n": len(x_values),
                            "R²": round(r2, 3),
                            "RMSE": round(rmse, 3),
                            "d-stat": round(d_stat, 3) if d_stat is not None else None,
                        })
                        
                        for trno in valid_data['TRNO'].unique():
                            trno_data = valid_data[valid_data['TRNO'] == trno]
                            
                            # Create legend label based on selection
                            legend_label = f"{y_label or y_var} - Treatment {trno}"
                            
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=trno_data[x_var],
                                    y=trno_data[y_var],
                                    mode='markers',
                                    name=legend_label,
                                    marker=dict(size=10),
                                    hovertemplate=(
                                        f"Treatment {trno}<br>" +
                                        f"{x_var}: %{{x}}<br>" +
                                        f"{y_var}: %{{y}}<br>" +
                                        "<extra></extra>"
                                    )
                                )
                            )
                        
                        
                # Update layout for custom mode
                x_label, _ = get_variable_info(x_var)
                fig.update_layout(
                    title=f"Multiple Variables vs {x_label or x_var}",
                    xaxis_title=x_label or x_var,
                    yaxis_title="Values",
                    showlegend=True,
                    height=600,
                    template="plotly_white",
                    legend=dict(
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )
            
            else:
                fig = go.Figure()
            
            # Final return with metrics data
            return (
                auto_style, custom_style,
                metrics_data, empty_metrics_columns,
                auto_options, auto_vars,
                all_vars, x_var,
                all_vars, y_vars,
                fig,
                data_preview,
                data_preview_columns
            )
            
        except Exception as e:
            logger.error(f"Error updating scatter plot: {str(e)}")
            traceback.print_exc()
            return empty_return

    # Add this new callback to automatically trigger the refresh plot button when switching tabs
    @app.callback(
        Output("refresh-plot-button", "n_clicks", allow_duplicate=True),
        [Input("viz-tabs", "active_tab")],
        [State("refresh-plot-button", "n_clicks"), State("execution-status", "data")],
        prevent_initial_call=True
    )
    def auto_refresh_on_tab_change(active_tab, current_clicks, execution_status):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        
        # Get the tab that triggered the callback
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Only increment click count once when switching to time-series tab
        if (trigger_id == "viz-tabs" and 
            active_tab == "time-series" and 
            execution_status and 
            execution_status.get("completed", False)):
            
            # Set a specific value instead of incrementing to avoid multiple refreshes
            return (current_clicks or 0) + 1
        
        return dash.no_update
    

    @app.callback(
        [
            Output("warning-message", "is_open"),
            Output("warning-message", "children")
        ],
        [
            Input("folder-selector", "value"),
            Input("experiment-selector", "value")
        ],
        [State("execution-status", "data")],
        prevent_initial_call=True
    )
    def show_run_treatment_alert(folder, experiment, execution_status):
        """Show warning alert when crop or experiment changes"""
        ctx = callback_context
        if not ctx.triggered:
            return False, ""

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id in ["folder-selector", "experiment-selector"]:
            message = ("You've changed the crop. " if trigger_id == "folder-selector" else "You've changed the experiment. ") + \
                     "Please run the treatment to update visualizations."
            return True, message
        
        return False, ""