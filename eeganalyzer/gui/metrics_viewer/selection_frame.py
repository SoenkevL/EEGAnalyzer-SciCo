"""
Copyright (C) <2025>  <Soenke van Loh>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

Selection frame for the EEG Metrics Viewer.

This module provides the SelectionFrame class for selecting experiments, EEGs,
metrics, channels, and other visualization options.
"""

from typing import List, Dict, Tuple, Optional, Any, Callable
import os
import sys
import subprocess
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import mne
import numpy as np

from .database_handler import DatabaseHandler
from .plot_frame import MetricsPlotFrame
from .utils import (
    get_label_font,
    get_dropdown_font,
    get_button_font,
    COMMON_CHANNELS,
    ALTERNATIVE_CHANNEL_NAMES
)


class SelectionFrame(ctk.CTkFrame):
    """
    A frame containing controls for selecting experiments, EEGs, metrics, channels, and other options.
    Also provides functionality to view the original EEG file using MNE, synchronized with the current time window.
    """

    def __init__(self, master, db_handler: DatabaseHandler, plot_frame: MetricsPlotFrame, **kwargs):
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
        self.grid_rowconfigure(8, weight=0)  # Row for the View EEG button

        # Create fonts
        label_font = get_label_font()
        dropdown_font = get_dropdown_font()

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
        button_font = dropdown_font

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
            font=get_button_font()
        )
        self.update_button.grid(row=7, column=1, padx=5, pady=(5, 0), sticky="ew")

        # View EEG button
        self.view_eeg_button = ctk.CTkButton(
            self,
            text="View EEG",
            command=self.view_eeg,
            height=30,
            font=get_button_font(),
            fg_color="#2a6099"  # Different color to distinguish it
        )
        self.view_eeg_button.grid(row=8, column=0, columnspan=2, padx=5, pady=(5, 8), sticky="ew")

        # Initialize with the first experiment if available
        if self.experiments:
            self.on_experiment_selected(experiment_names[0])

    def on_experiment_selected(self, selection: str):
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

    def on_eeg_selected(self, selection: str):
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

    def on_metric_selected(self, selection: str):
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
            font=get_dropdown_font()
        )
        self.search_entry.grid(row=0, column=0, padx=5, pady=3, sticky="ew")

        # Create the channels container frame
        self.channels_container = ctk.CTkFrame(self.channels_frame, fg_color="transparent")
        self.channels_container.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # Display all channels initially
        self.display_channels(self.available_channels)

    def display_channels(self, channels: List[str]):
        """
        Display the given channels as checkboxes.

        Args:
            channels: List of channel names to display
        """
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
                font=get_dropdown_font(),
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
        """
        Bind mouse wheel events to the widget for scrolling.

        Args:
            widget: The widget to bind mouse wheel events to
        """
        # Bind for Windows and Linux (with mouse wheel)
        widget.bind_all("<MouseWheel>", lambda event: self._on_mouse_wheel(event, widget))
        # Bind for Linux (with touchpad)
        widget.bind_all("<Button-4>", lambda event: self._on_mouse_wheel(event, widget))
        widget.bind_all("<Button-5>", lambda event: self._on_mouse_wheel(event, widget))

    def _on_mouse_wheel(self, event, widget):
        """
        Handle mouse wheel events for scrolling.

        Args:
            event: The mouse wheel event
            widget: The widget to scroll
        """
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

    def clear_channels(self):
        """Clear all channel checkboxes and related widgets."""
        for widget in self.channels_frame.winfo_children():
            widget.destroy()

        self.channel_vars = {}
        self.available_channels = []

    def clear_metrics(self):
        """Clear the metrics dropdown."""
        self.metric_dropdown.configure(values=[])
        self.metric_var.set("")
        self.current_metric = None

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

        # Select the channels if they exist in our available channels
        for channel in COMMON_CHANNELS:
            if channel in self.channel_vars:
                self.channel_vars[channel].set(True)
            # Try alternative name if the channel doesn't exist
            elif channel in ALTERNATIVE_CHANNEL_NAMES and ALTERNATIVE_CHANNEL_NAMES[channel] in self.channel_vars:
                self.channel_vars[ALTERNATIVE_CHANNEL_NAMES[channel]].set(True)

        # If we have a search filter active, update the display
        if hasattr(self, 'search_var'):
            self.filter_channels()

    def reset_time_window(self):
        """Reset the time window to show all data."""
        self.start_time_var.set("0")
        self.end_time_var.set("")
        self.update_plot()

    def view_eeg(self):
        """Open the current EEG file with MNE and display it in a new window using a subprocess."""
        if not hasattr(self, 'current_experiment_id') or not hasattr(self, 'current_eeg_id'):
            messagebox.showinfo("No EEG Selected", "Please select an experiment and EEG file first.")
            return

        # Find the selected EEG to get its filepath
        selected_eeg = None
        for eeg in self.eegs:
            if eeg['id'] == self.current_eeg_id:
                selected_eeg = eeg
                break

        if not selected_eeg or not selected_eeg.get('filepath'):
            messagebox.showerror("Error", "Could not find the EEG file path.")
            return

        eeg_filepath = selected_eeg['filepath']

        # Check if the file exists
        if not os.path.exists(eeg_filepath):
            messagebox.showerror("File Not Found", f"The EEG file was not found at:\n{eeg_filepath}")
            return

        # Get the current time window
        try:
            start_time = float(self.start_time_var.get()) if self.start_time_var.get() else None
        except ValueError:
            start_time = None

        try:
            end_time = float(self.end_time_var.get()) if self.end_time_var.get() else None
        except ValueError:
            end_time = None

        try:
            # Get the path to the helper script
            helper_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mne_plot_helper.py")

            # Make sure the helper script is executable
            if not os.access(helper_script_path, os.X_OK) and sys.platform != 'win32':
                os.chmod(helper_script_path, 0o755)

            # Build the command to run the helper script
            cmd = [sys.executable, helper_script_path, "--filepath", eeg_filepath]

            # Add time window parameters if specified
            if start_time is not None:
                cmd.extend(["--start-time", str(start_time)])
            if end_time is not None:
                cmd.extend(["--end-time", str(end_time)])

            # Add title
            window_title = f"MNE EEG Viewer - {selected_eeg['filename']}"
            cmd.extend(["--title", window_title])

            # Launch the subprocess
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Close the loading window after a short delay
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open the EEG file with MNE:\n{str(e)}")

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
