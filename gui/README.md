# EEG Metrics Viewer

This module provides a graphical user interface (GUI) for visualizing metrics computed from EEG data and stored in the EEGAnalyzer.sqlite database.

## Features

- Select experiments and EEGs from the database
- Choose which metrics to display
- Select specific EEG channels to visualize with convenient options:
  - Alphabetically sorted channel list for easy navigation
  - Search filter for quickly finding channels
  - Select/deselect all channels with a single click
  - "Common Channels" button to select standard 10-20 system electrodes
  - Scrollable channel list with mouse wheel support for easy navigation
- Plot metrics over time for selected channels
- Statistical aggregation across channels:
  - Calculate and display mean across all selected channels
  - Calculate and display standard deviation across all selected channels
  - Calculate and display median across all selected channels
  - Visually distinguish aggregations with different colors and line styles
  - Option to show only aggregations without individual channels for clearer visualization
- Interactive time window selection:
  - Specify exact time ranges using input fields
  - Click and drag on the plot to zoom into a specific time range
  - Reset zoom with a single click
- Interactive plot with customizable display options

## Data Structure

The application is designed to work with the following data structure in the database tables:

- **eeg_id**: ID of the EEG recording
- **label**: Processing label
- **startDataRecord**: Starting time (in seconds) of the window
- **duration**: Length of the time window
- **metric**: Name of the metric represented in that row
- **[Channel Names]**: Remaining columns are EEG channels with computed metric values

## Requirements

- Python 3.6+
- Required packages:
  - customtkinter
  - matplotlib
  - numpy
  - pandas
  - sqlalchemy

## Usage

### Running the Application

You can run the application using the provided run script:

```bash
python run_metrics_viewer.py [path_to_database]
```

If no database path is provided, the script will look for the default database at "../example/EEGAnalyzer.sqlite".

### Using the Interface

1. **Select an Experiment**: Choose an experiment from the dropdown menu.
2. **Select an EEG**: Choose an EEG recording associated with the selected experiment.
3. **Select a Metric**: Choose a metric from the dropdown menu.
4. **Select Channels**:
   - Use the search box to filter channels by name
   - Click "Select All" to select all channels or "Deselect All" to clear all selections
   - Click "Common Channels" to select standard 10-20 system electrodes
   - Check the boxes for individual channels you want to visualize
   - Scroll through the list using the mouse wheel or scroll bar to see all available channels
5. **Select Aggregation Methods** (optional):
   - Check the boxes for the statistical aggregations you want to calculate across channels
   - Mean: Calculates the average value across all selected channels
   - Std Dev: Calculates the standard deviation across all selected channels
   - Median: Calculates the median value across all selected channels
   - Show Aggregation Only: When checked, only the aggregation lines will be displayed (without individual channels)
6. **Set Time Window** (optional):
   - Enter start and end times in seconds to focus on a specific time range
   - Alternatively, click and drag on the plot to select a time range visually
   - Click "Reset Zoom" to view the entire time range again
7. **Update Plot**: Click the "Update Plot" button to display the selected metric for the chosen channels and aggregations.

## Project Structure

### Main Files
- `run_metrics_viewer.py`: Entry point script to run the application
- `install_dependencies.py`: Script to install required dependencies

### Package Structure
- `metrics_viewer/`: Main package directory
  - `__init__.py`: Package initialization and exports
  - `app.py`: Main application class
  - `database_handler.py`: Database interaction functionality
  - `plot_frame.py`: Plotting and visualization components
  - `selection_frame.py`: UI controls for selection and configuration
  - `utils.py`: Utility functions and constants

### Reference Files
- `plotting_oneshot.py`: Example plotting application (for reference)

## Extending the Application

The modular structure makes it easy to extend the application with new features:

### Adding New Visualization Types
1. Modify the `plot_frame.py` file to add new visualization methods
2. Add new options in the `selection_frame.py` file for selecting the visualization type
3. Update the `update_plot` method to handle the new visualization options

### Adding New Data Processing Features
1. Add new data processing methods in the `database_handler.py` file
2. Expose the new functionality through the `SelectionFrame` class
3. Update the plotting logic to visualize the processed data

### Adding New UI Controls
1. Add new UI elements to the `SelectionFrame` class
2. Connect the UI elements to appropriate callback methods
3. Update the `update_plot` method to use the new settings

### Adding New Aggregation Methods
1. Add new aggregation methods to the `utils.py` file
2. Update the `SelectionFrame` class to include the new aggregation options
3. Implement the calculation logic in the `MetricsPlotFrame.update_plot` method
