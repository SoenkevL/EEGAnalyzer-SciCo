"""
EEG Metrics Viewer

This application provides a GUI for visualizing metrics computed from EEG data
and stored in the EEGAnalyzer.sqlite database.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from typing import List, Dict, Tuple, Optional, Any

# Add the parent directory to the path so we can import from OOP_Analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OOP_Analyzer import Alchemist

# Set the appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")


class DatabaseHandler:
    """
    Handles interactions with the SQLite database.
    """

    def __init__(self, db_path: str):
        """
        Initialize the database handler.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.engine = Alchemist.initialize_tables(db_path)
        self.session = Alchemist.Session(self.engine)

    def __del__(self):
        """Close the session when the object is deleted."""
        if hasattr(self, 'session'):
            self.session.close()

    def get_experiments(self) -> List[Dict]:
        """
        Get all experiments from the database.

        Returns:
            List of dictionaries containing experiment information
        """
        experiments = Alchemist.find_entries(self.session, Alchemist.Experiment)
        return [{'id': exp.id, 'name': exp.metric_set_name, 'run_name': exp.run_name} for exp in experiments]

    def get_eegs_for_experiment(self, experiment_id: str) -> List[Dict]:
        """
        Get all EEGs associated with a specific experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            List of dictionaries containing EEG information
        """
        experiment = self.session.get(Alchemist.Experiment, experiment_id)
        if not experiment:
            return []

        return [{'id': eeg.id, 'filename': eeg.filename, 'filepath': eeg.filepath} for eeg in experiment.eegs]

    def get_metrics_data(self, experiment_id: str, eeg_id: str) -> pd.DataFrame:
        """
        Get metrics data for a specific experiment and EEG.

        Args:
            experiment_id: ID of the experiment
            eeg_id: ID of the EEG

        Returns:
            DataFrame containing the metrics data
        """
        table_name = f"data_experiment_{experiment_id}"

        try:
            # Query the data for the specific EEG
            query = f"SELECT * FROM {table_name} WHERE eeg_id = '{eeg_id}'"
            df = pd.read_sql_query(query, self.engine)
            return df
        except Exception as e:
            print(f"Error retrieving metrics data: {e}")
            return pd.DataFrame()

    def get_available_metrics(self, experiment_id: str, eeg_id: str) -> List[str]:
        """
        Get the unique metric names available for a specific experiment and EEG.

        Args:
            experiment_id: ID of the experiment
            eeg_id: ID of the EEG

        Returns:
            List of unique metric names
        """
        df = self.get_metrics_data(experiment_id, eeg_id)

        if 'metric' in df.columns:
            return df['metric'].unique().tolist()
        return []

    def get_available_channels(self, experiment_id: str, eeg_id: str) -> List[str]:
        """
        Get the channel names available for a specific experiment and EEG.

        Args:
            experiment_id: ID of the experiment
            eeg_id: ID of the EEG

        Returns:
            List of channel names sorted alphabetically
        """
        df = self.get_metrics_data(experiment_id, eeg_id)

        # Exclude metadata columns
        metadata_cols = ['eeg_id', 'label', 'startDataRecord', 'duration', 'metric']
        channel_cols = [col for col in df.columns if col not in metadata_cols]

        # Sort channels alphabetically
        channel_cols.sort()

        return channel_cols


class MetricsPlotFrame(ctk.CTkFrame):
    """
    A frame containing a matplotlib figure for plotting metrics.
    """

    def __init__(self, master, title="Metrics Plot", **kwargs):
        """
        Initialize the plot frame.

        Args:
            master: The parent widget
            title: Title for the plot frame
            **kwargs: Additional arguments for the CTkFrame constructor
        """
        super().__init__(master, **kwargs)

        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Add title label
        self.title_label = ctk.CTkLabel(self, text=title, fg_color="gray30", corner_radius=6)
        self.title_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")

        # Create matplotlib figure with larger size
        self.figure = Figure(figsize=(12, 7), dpi=100)
        self.plot = self.figure.add_subplot(111)

        # Create canvas for the figure
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Add interactive zooming with mouse drag
        self.zoom_start = None
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)

        # Store the parent frame for callbacks
        self.parent_frame = master

        # Initialize with empty plot
        self.update_plot()

    def update_plot(self, data=None, metric=None, channels=None, title=None, xlabel=None, ylabel=None, time_window=None, aggregations=None, aggregation_only=False):
        """
        Update the plot with new data.

        Args:
            data: DataFrame containing the metrics data
            metric: The metric to plot
            channels: List of channels to plot
            title: Title for the plot
            xlabel: Label for the x-axis
            ylabel: Label for the y-axis
            time_window: Tuple of (start_time, end_time) to focus on a specific time range
            aggregations: List of aggregation methods to apply across channels (mean, std, median)
            aggregation_only: If True, only show aggregations without individual channels
        """
        self.plot.clear()

        if data is None or metric is None or (not channels and not aggregations):
            # Display a message if no data is provided
            self.plot.text(0.5, 0.5, "No data selected",
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=self.plot.transAxes)
        else:
            # Filter data for the selected metric
            metric_data = data[data['metric'] == metric]

            if metric_data.empty:
                self.plot.text(0.5, 0.5, f"No data for metric: {metric}",
                              horizontalalignment='center',
                              verticalalignment='center',
                              transform=self.plot.transAxes)
            else:
                # Sort by startDataRecord if available
                if 'startDataRecord' in metric_data.columns:
                    metric_data = metric_data.sort_values('startDataRecord')
                    x_values = metric_data['startDataRecord']
                    x_label = 'Time (s)'

                    # Apply time window filtering if specified
                    if time_window and any(x is not None for x in time_window):
                        start_time, end_time = time_window

                        # Filter by start time if specified
                        if start_time is not None:
                            metric_data = metric_data[metric_data['startDataRecord'] >= start_time]
                            if metric_data.empty:
                                self.plot.text(0.5, 0.5, f"No data in the specified time range",
                                              horizontalalignment='center',
                                              verticalalignment='center',
                                              transform=self.plot.transAxes)
                                x_label = 'Time (s)'
                                self.plot.set_xlabel(x_label)
                                self.plot.set_ylabel(metric if metric else 'Value')
                                self.figure.tight_layout()
                                self.canvas.draw()
                                return

                        # Filter by end time if specified
                        if end_time is not None:
                            metric_data = metric_data[metric_data['startDataRecord'] <= end_time]
                            if metric_data.empty:
                                self.plot.text(0.5, 0.5, f"No data in the specified time range",
                                              horizontalalignment='center',
                                              verticalalignment='center',
                                              transform=self.plot.transAxes)
                                x_label = 'Time (s)'
                                self.plot.set_xlabel(x_label)
                                self.plot.set_ylabel(metric if metric else 'Value')
                                self.figure.tight_layout()
                                self.canvas.draw()
                                return

                        # Update x_values after filtering
                        x_values = metric_data['startDataRecord']
                else:
                    x_values = range(len(metric_data))
                    x_label = 'Sample'

                # Get the channel columns for plotting
                channel_columns = [col for col in metric_data.columns if col in channels]

                # Plot each selected channel if not in aggregation_only mode
                if not aggregation_only:
                    for channel in channels:
                        if channel in metric_data.columns:
                            self.plot.plot(x_values, metric_data[channel], label=channel, alpha=0.7)

                # Calculate and plot aggregations if requested
                if aggregations and channel_columns:
                    # Define colors for aggregations
                    agg_colors = {
                        'mean': 'red',
                        'std': 'purple',
                        'median': 'green'
                    }

                    # Define line styles for aggregations
                    agg_styles = {
                        'mean': '-',
                        'std': '--',
                        'median': '-.'
                    }

                    # Calculate and plot each selected aggregation
                    for agg in aggregations:
                        if agg == 'mean':
                            # Calculate mean across channels
                            mean_values = metric_data[channel_columns].mean(axis=1)
                            self.plot.plot(x_values, mean_values,
                                          label='Mean',
                                          color=agg_colors['mean'],
                                          linestyle=agg_styles['mean'],
                                          linewidth=2.5)

                        elif agg == 'std':
                            # Calculate standard deviation across channels
                            std_values = metric_data[channel_columns].std(axis=1)
                            self.plot.plot(x_values, std_values,
                                          label='Std Dev',
                                          color=agg_colors['std'],
                                          linestyle=agg_styles['std'],
                                          linewidth=2.5)

                        elif agg == 'median':
                            # Calculate median across channels
                            median_values = metric_data[channel_columns].median(axis=1)
                            self.plot.plot(x_values, median_values,
                                          label='Median',
                                          color=agg_colors['median'],
                                          linestyle=agg_styles['median'],
                                          linewidth=2.5)

                # Add legend if there are multiple items to show
                if (not aggregation_only and len(channels) > 1) or \
                   (aggregations and len(aggregations) > 0) or \
                   (not aggregation_only and channels and aggregations):
                    self.plot.legend()

                # Set x-axis limits if time window is specified
                if 'startDataRecord' in metric_data.columns and time_window and any(x is not None for x in time_window):
                    start_time, end_time = time_window
                    x_min, x_max = None, None

                    if start_time is not None:
                        x_min = start_time
                    else:
                        x_min = min(x_values) if len(x_values) > 0 else 0

                    if end_time is not None:
                        x_max = end_time
                    else:
                        x_max = max(x_values) if len(x_values) > 0 else 1

                    # Add a small padding to the limits
                    padding = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
                    self.plot.set_xlim(x_min - padding, x_max + padding)

        # Set title and labels
        if title:
            self.plot.set_title(title)
        if xlabel:
            self.plot.set_xlabel(xlabel)
        else:
            self.plot.set_xlabel(x_label if 'x_label' in locals() else 'Sample')
        if ylabel:
            self.plot.set_ylabel(ylabel)
        else:
            self.plot.set_ylabel(metric if metric else 'Value')

        # Adjust layout and redraw
        self.figure.tight_layout()
        self.canvas.draw()

    def on_mouse_press(self, event):
        """Handle mouse press event for interactive zooming."""
        # Only handle left button clicks in the plot area
        if event.button != 1 or event.inaxes != self.plot:
            return

        # Store the starting point for the zoom box
        self.zoom_start = (event.xdata, event.ydata)

        # Create a rectangle for the zoom box if it doesn't exist
        if not hasattr(self, 'zoom_rect'):
            self.zoom_rect = self.plot.axvspan(event.xdata, event.xdata, alpha=0.3, color='gray')
            self.zoom_rect.set_visible(False)

    def on_mouse_motion(self, event):
        """Handle mouse motion event for interactive zooming."""
        # Only handle motion when we have a zoom start point and we're in the plot area
        if self.zoom_start is None or event.inaxes != self.plot or not hasattr(self, 'zoom_rect'):
            return

        # Update the zoom box
        x_start = self.zoom_start[0]
        x_current = event.xdata

        # Make sure we have valid coordinates
        if x_start is None or x_current is None:
            return

        # Set the zoom box coordinates
        x_min = min(x_start, x_current)
        x_max = max(x_start, x_current)

        # Update the zoom rectangle
        self.zoom_rect.set_visible(True)
        self.zoom_rect.set_xy([[x_min, 0], [x_min, 1], [x_max, 1], [x_max, 0], [x_min, 0]])

        # Redraw the canvas
        self.canvas.draw_idle()

    def on_mouse_release(self, event):
        """Handle mouse release event for interactive zooming."""
        # Only handle left button releases when we have a zoom start point
        if event.button != 1 or self.zoom_start is None or event.inaxes != self.plot:
            if hasattr(self, 'zoom_rect'):
                self.zoom_rect.set_visible(False)
                self.canvas.draw_idle()
            self.zoom_start = None
            return

        # Get the start and end points
        x_start = self.zoom_start[0]
        x_end = event.xdata

        # Make sure we have valid coordinates
        if x_start is None or x_end is None:
            self.zoom_start = None
            if hasattr(self, 'zoom_rect'):
                self.zoom_rect.set_visible(False)
                self.canvas.draw_idle()
            return

        # Reset the zoom rectangle
        if hasattr(self, 'zoom_rect'):
            self.zoom_rect.set_visible(False)
            self.canvas.draw_idle()

        # Only zoom if the drag distance is significant
        if abs(x_start - x_end) < 0.01:
            self.zoom_start = None
            return

        # Sort the coordinates
        x_min = min(x_start, x_end)
        x_max = max(x_start, x_end)

        # Update the time window in the selection frame
        if hasattr(self, 'parent_frame') and hasattr(self.parent_frame, 'selection_frame'):
            selection_frame = self.parent_frame.selection_frame
            if hasattr(selection_frame, 'start_time_var') and hasattr(selection_frame, 'end_time_var'):
                selection_frame.start_time_var.set(f"{x_min:.2f}")
                selection_frame.end_time_var.set(f"{x_max:.2f}")
                selection_frame.update_plot()

        # Reset the zoom start point
        self.zoom_start = None


class SelectionFrame(ctk.CTkFrame):
    """
    A frame containing controls for selecting experiments, EEGs, metrics, and channels.
    """

    def __init__(self, master, db_handler, plot_frame, **kwargs):
        """
        Initialize the selection frame.

        Args:
            master: The parent widget
            db_handler: DatabaseHandler instance for querying the database
            plot_frame: MetricsPlotFrame instance for displaying plots
            **kwargs: Additional arguments for the CTkFrame constructor
        """
        super().__init__(master, **kwargs)

        self.db_handler = db_handler
        self.plot_frame = plot_frame

        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Create a common font for labels and dropdowns - smaller for more compact layout
        label_font = ctk.CTkFont(size=12)
        dropdown_font = ctk.CTkFont(size=11)

        # Experiment selection - more compact
        self.experiment_label = ctk.CTkLabel(self, text="Select Experiment:", font=label_font)
        self.experiment_label.grid(row=0, column=0, padx=5, pady=(8, 0), sticky="w")

        self.experiments = self.db_handler.get_experiments()
        experiment_names = [f"{exp['name']} ({exp['run_name']})" for exp in self.experiments]

        self.experiment_var = ctk.StringVar(value=experiment_names[0] if experiment_names else "")
        self.experiment_dropdown = ctk.CTkOptionMenu(
            self,
            values=experiment_names,
            variable=self.experiment_var,
            command=self.on_experiment_selected,
            height=28,
            font=dropdown_font,
            dropdown_font=dropdown_font
        )
        self.experiment_dropdown.grid(row=0, column=1, padx=5, pady=(8, 0), sticky="ew")

        # EEG selection - more compact
        self.eeg_label = ctk.CTkLabel(self, text="Select EEG:", font=label_font)
        self.eeg_label.grid(row=1, column=0, padx=5, pady=(8, 0), sticky="w")

        self.eeg_var = ctk.StringVar()
        self.eeg_dropdown = ctk.CTkOptionMenu(
            self,
            values=[],
            variable=self.eeg_var,
            command=self.on_eeg_selected,
            height=28,
            font=dropdown_font,
            dropdown_font=dropdown_font
        )
        self.eeg_dropdown.grid(row=1, column=1, padx=5, pady=(8, 0), sticky="ew")

        # Metric selection - more compact
        self.metric_label = ctk.CTkLabel(self, text="Select Metric:", font=label_font)
        self.metric_label.grid(row=2, column=0, padx=5, pady=(8, 0), sticky="w")

        self.metric_var = ctk.StringVar()
        self.metric_dropdown = ctk.CTkOptionMenu(
            self,
            values=[],
            variable=self.metric_var,
            command=self.on_metric_selected,
            height=28,
            font=dropdown_font,
            dropdown_font=dropdown_font
        )
        self.metric_dropdown.grid(row=2, column=1, padx=5, pady=(8, 0), sticky="ew")

        # Channels selection with more compact layout
        self.channels_label = ctk.CTkLabel(self, text="Select Channels:", font=label_font)
        self.channels_label.grid(row=3, column=0, padx=5, pady=(8, 0), sticky="nw")

        # Create a frame for channel selection buttons - more compact layout
        self.channel_buttons_frame = ctk.CTkFrame(self)
        self.channel_buttons_frame.grid(row=3, column=1, padx=5, pady=(8, 0), sticky="ew")
        self.channel_buttons_frame.grid_columnconfigure(0, weight=1)
        self.channel_buttons_frame.grid_columnconfigure(1, weight=1)
        self.channel_buttons_frame.grid_columnconfigure(2, weight=1)

        # Add select all and deselect all buttons with smaller font and more compact design
        button_font = ctk.CTkFont(size=12)

        self.select_all_button = ctk.CTkButton(
            self.channel_buttons_frame,
            text="Select All",
            command=self.select_all_channels,
            height=22,
            font=button_font
        )
        self.select_all_button.grid(row=0, column=0, padx=2, pady=3, sticky="ew")

        self.deselect_all_button = ctk.CTkButton(
            self.channel_buttons_frame,
            text="Deselect All",
            command=self.deselect_all_channels,
            height=22,
            font=button_font
        )
        self.deselect_all_button.grid(row=0, column=1, padx=2, pady=3, sticky="ew")

        # Add select common channels button
        self.select_common_button = ctk.CTkButton(
            self.channel_buttons_frame,
            text="Common Channels",
            command=self.select_common_channels,
            height=22,
            font=button_font
        )
        self.select_common_button.grid(row=0, column=2, padx=2, pady=3, sticky="ew")

        # Create a scrollable frame for channel checkboxes - increased height for better usability
        self.channels_frame = ctk.CTkScrollableFrame(self, width=180, height=200)
        self.channels_frame.grid(row=4, column=1, padx=5, pady=(0, 8), sticky="ew")

        # Bind mouse wheel events to ensure scrolling works properly
        self.bind_mouse_wheel(self.channels_frame)

        self.channel_vars = {}  # Will hold the checkbox variables

        # Aggregation methods frame
        self.aggregation_frame = ctk.CTkFrame(self)
        self.aggregation_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=(5, 0), sticky="ew")
        self.aggregation_frame.grid_columnconfigure(0, weight=1)
        self.aggregation_frame.grid_columnconfigure(1, weight=3)
        self.aggregation_frame.grid_rowconfigure(0, weight=1)
        self.aggregation_frame.grid_rowconfigure(1, weight=1)

        # Aggregation label
        self.aggregation_label = ctk.CTkLabel(
            self.aggregation_frame,
            text="Aggregation:",
            font=label_font
        )
        self.aggregation_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Aggregation checkboxes frame
        self.aggregation_checkboxes_frame = ctk.CTkFrame(self.aggregation_frame, fg_color="transparent")
        self.aggregation_checkboxes_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.aggregation_checkboxes_frame.grid_columnconfigure(0, weight=1)
        self.aggregation_checkboxes_frame.grid_columnconfigure(1, weight=1)
        self.aggregation_checkboxes_frame.grid_columnconfigure(2, weight=1)

        # Aggregation method checkboxes
        self.aggregation_vars = {}

        # Mean checkbox
        self.aggregation_vars['mean'] = ctk.BooleanVar(value=False)
        self.mean_checkbox = ctk.CTkCheckBox(
            self.aggregation_checkboxes_frame,
            text="Mean",
            variable=self.aggregation_vars['mean'],
            onvalue=True,
            offvalue=False,
            font=dropdown_font,
            checkbox_width=16,
            checkbox_height=16
        )
        self.mean_checkbox.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        # Std checkbox
        self.aggregation_vars['std'] = ctk.BooleanVar(value=False)
        self.std_checkbox = ctk.CTkCheckBox(
            self.aggregation_checkboxes_frame,
            text="Std Dev",
            variable=self.aggregation_vars['std'],
            onvalue=True,
            offvalue=False,
            font=dropdown_font,
            checkbox_width=16,
            checkbox_height=16
        )
        self.std_checkbox.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        # Median checkbox
        self.aggregation_vars['median'] = ctk.BooleanVar(value=False)
        self.median_checkbox = ctk.CTkCheckBox(
            self.aggregation_checkboxes_frame,
            text="Median",
            variable=self.aggregation_vars['median'],
            onvalue=True,
            offvalue=False,
            font=dropdown_font,
            checkbox_width=16,
            checkbox_height=16
        )
        self.median_checkbox.grid(row=0, column=2, padx=5, pady=2, sticky="w")

        # Aggregation only checkbox
        self.aggregation_only_var = ctk.BooleanVar(value=False)
        self.aggregation_only_checkbox = ctk.CTkCheckBox(
            self.aggregation_frame,
            text="Show Aggregation Only",
            variable=self.aggregation_only_var,
            onvalue=True,
            offvalue=False,
            font=dropdown_font,
            checkbox_width=16,
            checkbox_height=16
        )
        self.aggregation_only_checkbox.grid(row=1, column=1, padx=5, pady=(0, 2), sticky="w")

        # Time window selection frame
        self.time_window_frame = ctk.CTkFrame(self)
        self.time_window_frame.grid(row=6, column=0, columnspan=2, padx=5, pady=(5, 0), sticky="ew")
        self.time_window_frame.grid_columnconfigure(0, weight=1)
        self.time_window_frame.grid_columnconfigure(1, weight=1)
        self.time_window_frame.grid_columnconfigure(2, weight=1)
        self.time_window_frame.grid_columnconfigure(3, weight=1)

        # Time window label
        self.time_window_label = ctk.CTkLabel(
            self.time_window_frame,
            text="Time Window (s):",
            font=label_font
        )
        self.time_window_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Start time entry
        self.start_time_var = ctk.StringVar(value="0")
        self.start_time_entry = ctk.CTkEntry(
            self.time_window_frame,
            textvariable=self.start_time_var,
            width=60,
            height=25,
            font=dropdown_font
        )
        self.start_time_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # To label
        self.to_label = ctk.CTkLabel(
            self.time_window_frame,
            text="to",
            font=label_font
        )
        self.to_label.grid(row=0, column=2, padx=2, pady=5)

        # End time entry
        self.end_time_var = ctk.StringVar(value="")
        self.end_time_entry = ctk.CTkEntry(
            self.time_window_frame,
            textvariable=self.end_time_var,
            width=60,
            height=25,
            font=dropdown_font
        )
        self.end_time_entry.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        # Reset zoom button
        self.reset_zoom_button = ctk.CTkButton(
            self,
            text="Reset Zoom",
            command=self.reset_time_window,
            height=25,
            font=dropdown_font
        )
        self.reset_zoom_button.grid(row=7, column=0, padx=5, pady=(5, 0), sticky="ew")

        # Update button - more compact
        self.update_button = ctk.CTkButton(
            self,
            text="Update Plot",
            command=self.update_plot,
            height=30,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.update_button.grid(row=7, column=1, padx=5, pady=(5, 8), sticky="ew")

        # Initialize with the first experiment if available
        if self.experiments:
            self.on_experiment_selected(experiment_names[0])

    def on_experiment_selected(self, selection):
        """
        Handle experiment selection change.

        Args:
            selection: Selected experiment name
        """
        # Find the selected experiment
        selected_exp = None
        for exp in self.experiments:
            if f"{exp['name']} ({exp['run_name']})" == selection:
                selected_exp = exp
                break

        if not selected_exp:
            return

        # Update EEG dropdown
        self.current_experiment_id = selected_exp['id']
        eegs = self.db_handler.get_eegs_for_experiment(self.current_experiment_id)
        eeg_names = [eeg['filename'] for eeg in eegs]

        self.eegs = eegs
        self.eeg_dropdown.configure(values=eeg_names)
        if eeg_names:
            self.eeg_var.set(eeg_names[0])
            self.on_eeg_selected(eeg_names[0])
        else:
            self.eeg_var.set("")
            self.clear_metrics()
            self.clear_channels()

    def on_eeg_selected(self, selection):
        """
        Handle EEG selection change.

        Args:
            selection: Selected EEG name
        """
        # Find the selected EEG
        selected_eeg = None
        for eeg in self.eegs:
            if eeg['filename'] == selection:
                selected_eeg = eeg
                break

        if not selected_eeg:
            return

        # Update metrics dropdown
        self.current_eeg_id = selected_eeg['id']
        self.update_metrics_dropdown()
        self.update_channels_checkboxes()

    def on_metric_selected(self, selection):
        """
        Handle metric selection change.

        Args:
            selection: Selected metric name
        """
        self.current_metric = selection

    def update_metrics_dropdown(self):
        """Update the metrics dropdown based on the selected experiment and EEG."""
        # Get available metrics
        metrics = self.db_handler.get_available_metrics(self.current_experiment_id, self.current_eeg_id)

        # Update dropdown
        self.metric_dropdown.configure(values=metrics)
        if metrics:
            self.metric_var.set(metrics[0])
            self.current_metric = metrics[0]
        else:
            self.metric_var.set("")
            self.current_metric = None

    def update_channels_checkboxes(self):
        """Update the channel checkboxes based on the selected experiment and EEG."""
        # Clear existing checkboxes
        self.clear_channels()

        # Get available channels
        self.available_channels = self.db_handler.get_available_channels(self.current_experiment_id, self.current_eeg_id)

        # Add a search entry at the top of the channels frame
        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", self.filter_channels)

        self.search_frame = ctk.CTkFrame(self.channels_frame)
        self.search_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.search_frame.grid_columnconfigure(0, weight=1)

        self.search_entry = ctk.CTkEntry(
            self.search_frame,
            placeholder_text="Search channels...",
            textvariable=self.search_var,
            height=25,
            font=ctk.CTkFont(size=12)
        )
        self.search_entry.grid(row=0, column=0, padx=5, pady=3, sticky="ew")

        # Create the channels container frame
        self.channels_container = ctk.CTkFrame(self.channels_frame, fg_color="transparent")
        self.channels_container.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # Display all channels initially
        self.display_channels(self.available_channels)

    def display_channels(self, channels):
        """Display the given channels as checkboxes."""
        # Clear existing checkboxes in the container
        for widget in self.channels_container.winfo_children():
            widget.destroy()

        # Configure the container for proper scrolling
        self.channels_container.grid_columnconfigure(0, weight=1)

        # Sort channels alphabetically to ensure consistent display
        sorted_channels = sorted(channels)

        # Create a checkbox for each channel
        for i, channel in enumerate(sorted_channels):
            var = self.channel_vars.get(channel, ctk.BooleanVar(value=False))
            self.channel_vars[channel] = var

            # Create more compact checkboxes
            checkbox = ctk.CTkCheckBox(
                self.channels_container,
                text=channel,
                variable=var,
                onvalue=True,
                offvalue=False,
                height=20,
                font=ctk.CTkFont(size=12),
                checkbox_width=16,
                checkbox_height=16
            )
            checkbox.grid(row=i, column=0, padx=8, pady=3, sticky="w")

            # Bind mouse wheel event to each checkbox for better scrolling
            checkbox.bind("<MouseWheel>", lambda event, w=self.channels_frame: self._on_mouse_wheel(event, w))
            checkbox.bind("<Button-4>", lambda event, w=self.channels_frame: self._on_mouse_wheel(event, w))
            checkbox.bind("<Button-5>", lambda event, w=self.channels_frame: self._on_mouse_wheel(event, w))

    def filter_channels(self, *args):
        """Filter channels based on search text."""
        search_text = self.search_var.get().lower()

        if not search_text:
            # If search is empty, show all channels
            filtered_channels = self.available_channels
        else:
            # Filter channels that contain the search text
            filtered_channels = [ch for ch in self.available_channels if search_text in ch.lower()]

            # Sort the filtered channels alphabetically
            filtered_channels.sort()

        # Update the displayed channels
        self.display_channels(filtered_channels)

    def bind_mouse_wheel(self, widget):
        """Bind mouse wheel events to the widget for scrolling."""
        # Bind for Windows and Linux (with mouse wheel)
        widget.bind_all("<MouseWheel>", lambda event: self._on_mouse_wheel(event, widget))
        # Bind for Linux (with touchpad)
        widget.bind_all("<Button-4>", lambda event: self._on_mouse_wheel(event, widget))
        widget.bind_all("<Button-5>", lambda event: self._on_mouse_wheel(event, widget))

    def _on_mouse_wheel(self, event, widget):
        """Handle mouse wheel events for scrolling."""
        # Get the widget under the cursor
        x, y = event.x_root, event.y_root
        target_widget = event.widget.winfo_containing(x, y)

        # Check if the cursor is over our scrollable frame or its children
        parent = target_widget
        while parent is not None:
            if parent == widget or parent == self.channels_container:
                break
            parent = parent.master

        # If cursor is not over our scrollable area, don't scroll
        if parent is None:
            return

        # Handle different event types
        if event.num == 4 or event.delta > 0:  # Scroll up
            widget._parent_canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:  # Scroll down
            widget._parent_canvas.yview_scroll(1, "units")

    def clear_metrics(self):
        """Clear the metrics dropdown."""
        self.metric_dropdown.configure(values=[])
        self.metric_var.set("")
        self.current_metric = None

    def clear_channels(self):
        """Clear all channel checkboxes and related widgets."""
        for widget in self.channels_frame.winfo_children():
            widget.destroy()

        self.channel_vars = {}
        self.available_channels = []

    def select_all_channels(self):
        """Select all channel checkboxes."""
        for var in self.channel_vars.values():
            var.set(True)

    def deselect_all_channels(self):
        """Deselect all channel checkboxes."""
        for var in self.channel_vars.values():
            var.set(False)

    def select_common_channels(self):
        """Select common EEG channels (10-20 system)."""
        # First deselect all
        self.deselect_all_channels()

        # Common 10-20 system channels (already in alphabetical order)
        common_channels = [
            'C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz',
            'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'Pz', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'
        ]

        # Map of alternative names (old -> new and new -> old)
        alternative_names = {
            'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
            'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'
        }

        # Select the channels if they exist in our available channels
        for channel in common_channels:
            if channel in self.channel_vars:
                self.channel_vars[channel].set(True)
            # Try alternative name if the channel doesn't exist
            elif channel in alternative_names and alternative_names[channel] in self.channel_vars:
                self.channel_vars[alternative_names[channel]].set(True)

        # If we have a search filter active, update the display
        if hasattr(self, 'search_var'):
            self.filter_channels()

    def reset_time_window(self):
        """Reset the time window to show all data."""
        self.start_time_var.set("0")
        self.end_time_var.set("")
        self.update_plot()

    def update_plot(self):
        """Update the plot with the selected metric and channels."""
        if not hasattr(self, 'current_experiment_id') or not hasattr(self, 'current_eeg_id') or not self.current_metric:
            return

        # Get selected channels
        selected_channels = [channel for channel, var in self.channel_vars.items() if var.get()]

        # Get selected aggregation methods
        selected_aggregations = [agg for agg, var in self.aggregation_vars.items() if var.get()]

        # Check if we should show only aggregations
        aggregation_only = self.aggregation_only_var.get()

        # Check if we have valid selections
        if (not selected_channels and not selected_aggregations) or \
           (aggregation_only and not selected_aggregations):
            self.plot_frame.update_plot(None, None, None, "No channels or aggregations selected")
            return

        # Get data for the selected experiment and EEG
        df = self.db_handler.get_metrics_data(self.current_experiment_id, self.current_eeg_id)

        if df.empty:
            self.plot_frame.update_plot(None, None, None, "No data available")
            return

        # Get time window values
        try:
            start_time = float(self.start_time_var.get()) if self.start_time_var.get() else None
        except ValueError:
            start_time = None
            self.start_time_var.set("0")

        try:
            end_time = float(self.end_time_var.get()) if self.end_time_var.get() else None
        except ValueError:
            end_time = None
            self.end_time_var.set("")

        # Update the plot
        experiment_name = next((exp['name'] for exp in self.experiments if exp['id'] == self.current_experiment_id), "")
        eeg_name = next((eeg['filename'] for eeg in self.eegs if eeg['id'] == self.current_eeg_id), "")

        title = f"{self.current_metric} for {experiment_name} - {eeg_name}"

        # Add aggregation-only info to title if specified
        if aggregation_only and selected_aggregations:
            title += " (Aggregation Only)"

        # Add time window info to title if specified
        if start_time is not None and end_time is not None:
            title += f" (Time: {start_time}s to {end_time}s)"
        elif start_time is not None:
            title += f" (Time: {start_time}s+)"

        self.plot_frame.update_plot(
            df,
            self.current_metric,
            selected_channels,
            title,
            time_window=(start_time, end_time),
            aggregations=selected_aggregations,
            aggregation_only=aggregation_only
        )


class App(ctk.CTk):
    """
    Main application class for the EEG Metrics Viewer.
    """

    def __init__(self, db_path):
        """
        Initialize the application.

        Args:
            db_path: Path to the SQLite database file
        """
        super().__init__()

        # Configure window
        self.title("EEG Metrics Viewer")
        self.geometry("1400x800")  # Larger default size

        # Configure grid layout - give much more space to the plot
        self.grid_columnconfigure(0, weight=1)    # Selection panel
        self.grid_columnconfigure(1, weight=8)    # Plot area (significantly increased weight)
        self.grid_rowconfigure(0, weight=1)

        # Initialize database handler
        self.db_handler = DatabaseHandler(db_path)

        # Create plot frame with more space
        self.plot_frame = MetricsPlotFrame(self, title="Metrics Visualization")
        self.plot_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")

        # Create selection frame with narrower fixed width
        self.selection_frame = SelectionFrame(self, self.db_handler, self.plot_frame, width=250)
        self.selection_frame.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")
        self.selection_frame.grid_propagate(False)  # Prevent the frame from resizing based on content


if __name__ == "__main__":
    # Check if a database path is provided as a command-line argument
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Default database path
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "example", "EEGAnalyzer.sqlite")

    # Ensure the database file exists
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        print("Please provide a valid path to the EEGAnalyzer.sqlite database.")
        sys.exit(1)

    # Start the application
    app = App(db_path)
    app.mainloop()
