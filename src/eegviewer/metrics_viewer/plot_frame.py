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

Plot frame for the EEG Metrics Viewer.

This module provides the MetricsPlotFrame class for visualizing EEG metrics data.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .utils import AGGREGATION_COLORS, AGGREGATION_STYLES


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

    def update_plot(
        self,
        data: Optional[pd.DataFrame] = None,
        metric: Optional[str] = None,
        channels: Optional[List[str]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        time_window: Optional[Tuple[Optional[float], Optional[float]]] = None,
        aggregations: Optional[List[str]] = None,
        aggregation_only: bool = False
    ):
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
                    # Calculate and plot each selected aggregation
                    for agg in aggregations:
                        if agg == 'mean':
                            # Calculate mean across channels
                            mean_values = metric_data[channel_columns].mean(axis=1)
                            self.plot.plot(x_values, mean_values,
                                          label='Mean',
                                          color=AGGREGATION_COLORS['mean'],
                                          linestyle=AGGREGATION_STYLES['mean'],
                                          linewidth=2.5)

                        elif agg == 'std':
                            # Calculate standard deviation across channels
                            std_values = metric_data[channel_columns].std(axis=1)
                            self.plot.plot(x_values, std_values,
                                          label='Std Dev',
                                          color=AGGREGATION_COLORS['std'],
                                          linestyle=AGGREGATION_STYLES['std'],
                                          linewidth=2.5)

                        elif agg == 'median':
                            # Calculate median across channels
                            median_values = metric_data[channel_columns].median(axis=1)
                            self.plot.plot(x_values, median_values,
                                          label='Median',
                                          color=AGGREGATION_COLORS['median'],
                                          linestyle=AGGREGATION_STYLES['median'],
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

        # Update the zoom rectangle - use axvspan's xy parameter correctly
        self.zoom_rect.set_visible(True)
        # Instead of setting xy coordinates directly, update the span
        self.zoom_rect.remove()
        self.zoom_rect = self.plot.axvspan(x_min, x_max, alpha=0.3, color='gray')

        # Redraw the canvas
        self.canvas.draw_idle()

    def on_mouse_release(self, event):
        """Handle mouse release event for interactive zooming."""
        # Only handle left button releases when we have a zoom start point
        if event.button != 1 or self.zoom_start is None or event.inaxes != self.plot:
            if hasattr(self, 'zoom_rect'):
                self.zoom_rect.remove()
                delattr(self, 'zoom_rect')
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
                self.zoom_rect.remove()
                delattr(self, 'zoom_rect')
                self.canvas.draw_idle()
            return

        # Reset the zoom rectangle
        if hasattr(self, 'zoom_rect'):
            self.zoom_rect.remove()
            delattr(self, 'zoom_rect')
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