"""
UI Layouts for DSSAT Viewer
"""
import os
import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

def create_scatter_tab_layout():
    """Create simplified layout for scatter plot tab with only the plot."""
    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id="scatter-plot",
                        style={"height": "80vh", "width": "100%"},
                        config={"responsive": True}
                    )
                ])
            ])
        ])
    )

def create_sidebar_layout():
    """Create the sidebar layout."""
    return dbc.Collapse(
        dbc.Card(
            [
                dbc.CardHeader(
                    [
                        html.H3("", className="fw-bold"),
                        dbc.Button(
                            "",
                            id="close-sidebar",
                            color="link",
                            className="ms-auto",
                            style={"font-size": "20px", "padding": "0"},
                        ),
                    ],
                    className="d-flex align-items-center",
                ),
                dbc.CardBody([
                    # Always visible controls
                    html.Label("Select Crop", className="fw-bold"),
                    dcc.Dropdown(
                        id="folder-selector",
                        placeholder="Select Folder",
                        className="mb-3",
                    ),
                    
                    html.Label("Select Experiment", className="fw-bold"),
                    dcc.Dropdown(
                        id="experiment-selector",
                        placeholder="Select Experiment",
                        className="mb-3",
                    ),
                    
                    html.Label("Select Treatments", className="fw-bold"),
                    dcc.Dropdown(
                        id="treatment-selector",
                        placeholder="Select Treatments",
                        multi=True,
                        className="mb-3",
                    ),
                    
                    dbc.Button(
                        "Run Treatment",
                        id="run-button",
                        color="primary",
                        className="mb-3 w-100",
                    ),

                    # Time Series Controls
                    html.Div(
                        id="time-series-controls",
                        children=[
                            html.Label("Select Output Files", className="fw-bold"),
                            dcc.Dropdown(
                                id="out-file-selector",
                                placeholder="Select .OUT Files",
                                multi=True,
                                className="mb-3",
                            ),
                            
                            html.Label("Select Variables", className="fw-bold"),
                            dcc.Dropdown(
                                id="x-var-selector",
                                placeholder="Select X Variable",
                                className="mb-2",
                            ),
                            dcc.Dropdown(
                                id="y-var-selector",
                                placeholder="Select Y Variables",
                                multi=True,
                                className="mb-3",
                            ),
                        ]
                    ),

                    # Scatter Plot Controls 
                    html.Div(
                        id="scatter-controls",
                        style={"display": "none"},
                        children=[
                            html.Label("Plot Mode", className="fw-bold"),
                            dcc.RadioItems(
                                id="scatter-mode",
                                options=[
                                    {'label': 'Simulated vs Measured (Auto-pair)', 'value': 'auto'},
                                    {'label': 'Custom X-Y Variables', 'value': 'custom'}
                                ],
                                value='auto',
                                className="mb-3"
                            ),
                            
                            html.Div(
                                id="auto-pair-container",
                                children=[
                                    html.Label("Select Variables to Compare", className="fw-bold"),
                                    dcc.Dropdown(
                                        id="scatter-var-selector",
                                        placeholder="Select Variables",
                                        multi=True,
                                        className="mb-3",
                                    ),
                                ]
                            ),
                            
                            html.Div(
                                id="custom-xy-container",
                                children=[
                                    html.Label("Select X Variable", className="fw-bold"),
                                    dcc.Dropdown(
                                        id="scatter-x-var-selector",
                                        placeholder="Select X Variable",
                                        className="mb-3"
                                    ),
                                    html.Label("Select Y Variables", className="fw-bold"),
                                    dcc.Dropdown(
                                        id="scatter-y-var-selector",
                                        placeholder="Select Y Variables",
                                        multi=True,
                                        className="mb-3"
                                    ),
                                ]
                            ),
                        ]
                    ),
                    
                    # Refresh button
                    dbc.Button(
                        "Refresh Plot",
                        id="refresh-plot-button",
                        color="info",
                        className="w-100",
                        style={"backgroundColor": "#007bff"},
                    ),
                ]),
            ],
            className="sidebar",
            style={
                "position": "fixed",
                "top": "0",
                "left": "0",
                "bottom": "0",
                "width": "22%",
                "overflow-y": "auto",
                "z-index": "1000",
                "background-color": "white",
                "box-shadow": "2px 0 5px rgba(0, 0, 0, 0.1)",
                "transition": "transform 0.3s ease-in-out",
            },
        ),
        id="sidebar",
        is_open=True,
    )

def create_preview_modal():
    """Create the data preview modal."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Data Preview")),
            dbc.ModalBody(
                dash_table.DataTable(
                    id="data-table",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "minWidth": "100px",
                        "maxWidth": "180px",
                        "whiteSpace": "normal",
                        "padding": "10px",
                    },
                    style_header={
                        "backgroundColor": "rgb(230, 230, 230)",
                        "fontWeight": "bold",
                        "padding": "10px",
                    },
                    page_current=0,
                    page_size=10,
                    page_action="native",
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                ),
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-preview-modal", className="ms-auto")
            ),
        ],
        id="preview-modal",
        size="xl",
        style={"maxWidth": "90%"},
    )
    
def create_metrics_modal():
    """Create the metrics modal."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Metrics Table")),
            dbc.ModalBody(
                dash_table.DataTable(
                    id="metrics-table",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "minWidth": "100px",
                        "maxWidth": "180px",
                        "whiteSpace": "normal",
                        "padding": "10px",
                    },
                    style_header={
                        "backgroundColor": "rgb(230, 230, 230)",
                        "fontWeight": "bold",
                        "padding": "10px",
                    },
                    page_current=0,
                    page_size=10,
                    page_action="native",
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                ),
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-metrics-modal", className="ms-auto")
            ),
        ],
        id="metrics-modal",
        size="xl",
        style={"maxWidth": "90%"},
    )

def create_app_layout():
    """Create the main application layout."""
    return html.Div(
        children=[
            # Loading overlay
            dcc.Loading(
                id="loading-treatment-run",
                type="default",
                children=[html.Div(id="treatment-run-output")],
                fullscreen=True,
                style={
                    "zIndex": "9999",
                    "backgroundColor": "rgba(255, 255, 255, 0.7)",
                },
            ),
            
            # State management
            dcc.Store(id="execution-status", data={"completed": False}),
            dcc.Store(id="sidebar-state", data={"is_open": True}),
            
            
            # Sidebar
            create_sidebar_layout(),
            
            # Main content area
            html.Div(
                [
                    # Toggle sidebar button
                    dbc.Button(
                        "☰",
                        id="toggle-sidebar",
                        color="primary",
                        size="sm",
                        style={
                            "position": "fixed",
                            "top": "13px",
                            "left": "calc(20% - 50px)",
                            "z-index": "1001",
                            "transition": "left 0.3s ease-in-out",
                        },
                    ),
                    
                    # Alert messages
                    dbc.Alert(
                        id="error-message",
                        is_open=False,
                        duration=4000,
                        color="danger",
                        style={
                            "position": "fixed",
                            "top": "10px",
                            "right": "10px",
                            "z-index": "1001",
                            "width": "300px",
                        },
                    ),
                    # Add this in the layout, next to your existing success and error alerts
                    dbc.Alert(
                        id="warning-message",
                        is_open=False,
                        duration=4000,  # Will auto-close after 4 seconds
                        color="warning",
                        style={
                            "position": "fixed",
                            "top": "10px",
                            "right": "10px",
                            "z-index": "1001",
                            "width": "300px",
                        },
                    ),
                    dbc.Alert(
                        id="success-message",
                        is_open=False,
                        duration=4000,
                        color="success",
                        style={
                            "position": "fixed",
                            "top": "10px",
                            "right": "10px",
                            "z-index": "1001",
                            "width": "300px",
                        },
                    ),
                    
                    # Visualization container
                    html.Div(
                        [
                            dcc.Loading(
                                id="loading-graph",
                                type="graph",  # Use the graph type for a more visual loading indicator
                                fullscreen=False,
                                children=[
                                    dbc.Tabs([
                                        dbc.Tab(
                                            dcc.Graph(
                                                id="data-plot",
                                                style={"height": "80vh", "width": "100%"},
                                            ),
                                            label="Time Series",
                                            tab_id="time-series"
                                        ),
                                        dbc.Tab(
                                            create_scatter_tab_layout(),
                                            label="Scatter Plot",
                                            tab_id="scatter"
                                        ),
                                    ], id="viz-tabs", active_tab="time-series")
                                ],
                            ),
                        ],
                        id="visualization-container",
                        style={
                            "transition": "margin-left 0.3s ease-in-out",
                            "height": "100vh",
                            "width": "90%",
                            "padding": "20px",
                            "marginLeft": "20%",
                        },
                    ),
                    
                    # Preview and metrics buttons
                    dbc.Button(
                        "Data Preview",
                        id="open-preview",
                        color="secondary",
                        className="me-2",
                        style={
                            "position": "fixed",
                            "bottom": "20px",
                            "right": "140px",
                            "z-index": "1001",
                        },
                    ),
                    dbc.Button(
                        "Show Metrics",
                        id="open-metrics",
                        color="info",
                        style={
                            "position": "fixed",
                            "bottom": "20px",
                            "right": "20px",
                            "z-index": "1001",
                        },
                    ),
                ],
                id="main-content",
            ),
            
            # Modal components
            create_preview_modal(),
            create_metrics_modal(),
            
            # Download component
            dcc.Download(id="download-data"),
        ]
    )