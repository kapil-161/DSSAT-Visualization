# DSSAT Viewer

A visualization tool for DSSAT crop model output data. This application provides an interface to run DSSAT simulations and visualize the results through time series and scatter plots.

## Features

- Run DSSAT simulations for different crops and treatments
- View time series plots of simulated and observed data
- Compare simulated vs. measured values through scatter plots
- Calculate and display performance metrics (RMSE, d-stat, etc.)
- Explore data in tabular format

## Project Structure

```
project/
│
├── main.py                    # Entry point, initializes and runs the application
├── ui/
│   ├── __init__.py
│   ├── app.py                 # DSSATViewer class and UI setup
│   ├── callbacks.py           # Dash callbacks
│   ├── layouts.py             # UI layout components
│   ├── dash_thread.py         # Thread for running Dash server
│
├── data/
│   ├── __init__.py
│   ├── dssat_io.py            # DSSAT file I/O operations
│   ├── data_processing.py     # Data processing and transformation functions
│   ├── visualization.py       # Visualization utilities
│
├── models/
│   ├── __init__.py
│   ├── metrics.py             # MetricsCalculator class
│
├── utils/
│   ├── __init__.py
│   ├── dssat_paths.py         # DSSAT path finding utilities
│
├── config.py                  # Configuration constants
└── requirements.txt           # Project dependencies
```

## Installation

1. Ensure you have DSSAT v4.8 installed on your system
2. Clone this repository
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application with:

```bash
python main.py
```

The application will:
1. Automatically detect your DSSAT installation
2. Launch a PyQt window with the Dash web interface
3. Allow you to select crops, experiments, and treatments
4. Run simulations and visualize the results

## Requirements

- Python 3.8 or higher
- DSSAT v4.8
- Qt5
- Dash and related packages (detailed in requirements.txt)

## How It Works

This application combines:
- PyQt5 for the desktop window
- Dash/Plotly for the interactive web interface
- Direct integration with DSSAT command-line interface

The architecture separates concerns into:
- UI components (layouts, callbacks)
- Data handling (file I/O, processing)
- Model integration (DSSAT paths, execution)
- Metrics calculation and visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
