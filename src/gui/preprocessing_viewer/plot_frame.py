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

Plot frame for EEG preprocessing GUI visualization.

This module provides the plotting functionality for the EEG preprocessing GUI,
reusing the existing MNE plot helper functionality.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import sys
import os
import numpy as np

# Add the parent directory to import existing plot helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from gui.metrics_viewer.mne_plot_helper import MNEPlotHelper
except ImportError:
    # Fallback if the helper doesn't exist
    class MNEPlotHelper:
        @staticmethod
        def create_matplotlib_figure_from_raw(raw, **kwargs):
            """Fallback method for creating matplotlib figures from raw data."""
            return None


class PreprocessingPlotFrame(ttk.Frame):
    """Frame containing EEG visualization and plotting capabilities."""
    
    def __init__(self, parent):
        """
        Initialize the plot frame.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.preprocessing_pipeline = None
        self.current_plot_type = None
        self.current_parameters = {}
        
        # Create the UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Create control panel
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Plot type selection
        ttk.Label(control_frame, text="Plot Type:").pack(side=tk.LEFT)
        self.plot_type_var = tk.StringVar(value="raw")
        plot_type_combo = ttk.Combobox(
            control_frame,
            textvariable=self.plot_type_var,
            values=["raw", "psd", "topomap", "sensors"],
            state="readonly",
            width=15
        )
        plot_type_combo.pack(side=tk.LEFT, padx=5)
        
        # Refresh button
        refresh_button = ttk.Button(
            control_frame,
            text="Refresh Plot",
            command=self.refresh_plot
        )
        refresh_button.pack(side=tk.LEFT, padx=5)
        
        # Use MNE native plotting button
        native_plot_button = ttk.Button(
            control_frame,
            text="Open MNE Native Plot",
            command=self.open_native_plot
        )
        native_plot_button.pack(side=tk.LEFT, padx=5)
        
        # Create matplotlib frame
        self.create_matplotlib_frame()
        
    def create_matplotlib_frame(self):
        """Create the matplotlib plotting frame."""
        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(12, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Initial empty plot
        self.ax.text(0.5, 0.5, 'Load EEG data to view plots', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=16)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
    def set_preprocessing_pipeline(self, pipeline):
        """
        Set the preprocessing pipeline.
        
        Args:
            pipeline: EEGPreprocessingPipeline instance
        """
        self.preprocessing_pipeline = pipeline
        self.refresh_initial_plot()
        
    def refresh_initial_plot(self):
        """Refresh the plot with current data."""
        if self.preprocessing_pipeline and self.preprocessing_pipeline.raw:
            self.create_plot("raw", {"duration": 20.0, "n_channels": 20, "start": 0})
            
    def create_plot(self, plot_type: str, parameters: dict):
        """
        Create a plot of the specified type.
        
        Args:
            plot_type: Type of plot to create
            parameters: Parameters for the plot
        """
        if not self.preprocessing_pipeline or not self.preprocessing_pipeline.raw:
            messagebox.showwarning("Warning", "No EEG data loaded")
            return
            
        self.current_plot_type = plot_type
        self.current_parameters = parameters
        
        try:
            # Clear the current plot
            self.ax.clear()
            
            raw = self.preprocessing_pipeline.raw
            
            if plot_type == "raw":
                self._plot_raw_data(raw, parameters)
            elif plot_type == "psd":
                self._plot_psd(raw, parameters)
            elif plot_type == "topomap":
                self._plot_topomap(raw, parameters)
            elif plot_type == "sensors":
                self._plot_sensors(raw, parameters)
            else:
                self.ax.text(0.5, 0.5, f'Plot type "{plot_type}" not implemented', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=self.ax.transAxes, fontsize=16)
                
            # Update the canvas
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create plot: {str(e)}")
            self.ax.clear()
            self.ax.text(0.5, 0.5, f'Error creating plot: {str(e)}', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=self.ax.transAxes, fontsize=12, color='red')
            self.canvas.draw()
            
    def _plot_raw_data(self, raw, parameters):
        """Plot raw EEG data."""
        try:
            # Use MNE's plot functionality adapted for matplotlib
            duration = parameters.get("duration", 20)
            n_channels = parameters.get("n_channels", 20)
            start = parameters.get("start", 0)
            
            # Get data
            data, times = raw.get_data(start=int(np.floor(start)), stop=int(np.ceil(start + duration)), return_times=True)
            
            # Select channels to plot
            n_channels = min(n_channels, len(raw.ch_names))
            ch_names = raw.ch_names[:n_channels]
            data = data[:n_channels]
            
            # Plot data
            for i, (ch_data, ch_name) in enumerate(zip(data, ch_names)):
                self.ax.plot(times, ch_data + i * 100e-6, label=ch_name)
                
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude (V)')
            self.ax.set_title(f'EEG Raw Data (Duration: {duration}s)')
            self.ax.grid(True, alpha=0.3)
            
        except Exception as e:
            raise Exception(f"Error plotting raw data: {str(e)}")
            
    def _plot_psd(self, raw, parameters):
        """Plot power spectral density."""
        try:
            # Compute PSD
            psd_data = raw.compute_psd(fmax=100)
            
            # Plot PSD
            psd_data.plot(axes=self.ax, show=False)
            self.ax.set_title('Power Spectral Density')
            
        except Exception as e:
            raise Exception(f"Error plotting PSD: {str(e)}")
            
    def _plot_topomap(self, raw, parameters):
        """Plot channel topography."""
        try:
            # This is a placeholder - actual topomap requires channel positions
            self.ax.text(0.5, 0.5, 'Topomap plotting requires channel positions\nUse "Open MNE Native Plot" for full topomap functionality', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=self.ax.transAxes, fontsize=12)
            
        except Exception as e:
            raise Exception(f"Error plotting topomap: {str(e)}")
            
    def _plot_sensors(self, raw, parameters):
        """Plot sensor positions."""
        try:
            # This is a placeholder - actual sensor plot requires channel positions
            self.ax.text(0.5, 0.5, 'Sensor plotting requires channel positions\nUse "Open MNE Native Plot" for full sensor functionality', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=self.ax.transAxes, fontsize=12)
            
        except Exception as e:
            raise Exception(f"Error plotting sensors: {str(e)}")
            
    def open_native_plot(self):
        """Open MNE's native plotting interface."""
        if not self.preprocessing_pipeline or not self.preprocessing_pipeline.raw:
            messagebox.showwarning("Warning", "No EEG data loaded")
            return
            
        try:
            # Use the existing plot_eeg_data method from the pipeline
            plot_kwargs = {
                "duration": 20.0,
                "n_channels": 20,
                "start": 0.0,
                "block": True,  # Don't block the GUI
                "title": "EEG Data - Native MNE Plot"
            }
            
            self.preprocessing_pipeline.plot_eeg_data(**plot_kwargs)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open native plot: {str(e)}")
            
    def refresh_plot(self):
        """Refresh the current plot."""
        if self.current_plot_type:
            self.create_plot(self.current_plot_type, self.current_parameters)
        else:
            self.refresh_initial_plot()
            
    def update_plot(self):
        """Update the plot after preprocessing steps."""
        self.refresh_plot()