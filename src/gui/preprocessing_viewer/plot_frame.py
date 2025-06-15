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

This module provides embedded plotting functionality for the EEG preprocessing GUI,
focusing on analysis plots like PSD and ICA sources that are displayed inline.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np


class PreprocessingPlotFrame(ttk.Frame):
    """Frame containing embedded EEG analysis plots."""
    
    def __init__(self, parent, title="EEG Analysis", **kwargs):
        """
        Initialize the plot frame.
        
        Args:
            parent: Parent widget
            title: Title for the plot frame
            **kwargs: Additional arguments for the Frame constructor
        """
        super().__init__(parent, **kwargs)
        
        self.preprocessing_pipeline = None
        self.current_plot_type = None
        self.current_parameters = {}
        
        # Create the UI
        self.setup_ui(title)
        
    def setup_ui(self, title):
        """Set up the user interface."""
        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Create control panel
        control_frame = ttk.Frame(self)
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Title label
        title_label = ttk.Label(control_frame, text=title, font=("Arial", 14, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Plot type selection
        ttk.Label(control_frame, text="Analysis:").pack(side=tk.LEFT, padx=(20, 5))
        self.plot_type_var = tk.StringVar(value="psd")
        plot_type_combo = ttk.Combobox(
            control_frame,
            textvariable=self.plot_type_var,
            values=["psd", "ica_components"],
            state="readonly",
            width=18
        )
        plot_type_combo.pack(side=tk.LEFT, padx=5)
        
        # Refresh button
        refresh_button = ttk.Button(
            control_frame,
            text="Update Plot",
            command=self.refresh_plot
        )
        refresh_button.pack(side=tk.LEFT, padx=5)
        
        # Create matplotlib frame
        self.create_matplotlib_frame()
        
    def create_matplotlib_frame(self):
        """Create the matplotlib plotting frame."""
        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create a separate frame for the toolbar to avoid geometry manager conflicts
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=2, column=0, sticky="ew", padx=5)
        
        # Create toolbar in the separate frame
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        # The toolbar will automatically pack itself within toolbar_frame
        
        # Initial empty plot
        self.ax.text(0.5, 0.5, 'Load EEG data to view analysis plots\n\nUse View menu for raw data and direct MNE plots', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=12, alpha=0.7)
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
            self.create_plot("psd", {})

    def create_plot(self, plot_type: str, parameters: dict):
        """
        Create an embedded analysis plot using preprocessing pipeline functions.

        Args:
            plot_type: Type of analysis plot to create
            parameters: Parameters for the plot
        """
        if not self.preprocessing_pipeline or not self.preprocessing_pipeline.raw:
            messagebox.showwarning("Warning", "No EEG data loaded")
            return

        self.current_plot_type = plot_type
        self.current_parameters = parameters

        try:
            # Get figure from preprocessing pipeline (with default size)
            pipeline_fig = self._get_pipeline_figure(plot_type, parameters)

            if pipeline_fig is not None:
                # Resize the figure to match current canvas size
                self._resize_figure_to_canvas(pipeline_fig)

                # Replace the figure reference and redraw
                self.canvas.figure = pipeline_fig
                self.fig = pipeline_fig  # Keep reference
                self.canvas.draw()
            else:
                self._show_error_message(f'Could not generate {plot_type} plot')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create analysis plot: {str(e)}")
            self._show_error_message(f'Error creating plot: {str(e)}')

    def _resize_figure_to_canvas(self, figure):
        """
        Resize a matplotlib figure to match the current canvas size.

        Args:
            figure: matplotlib.Figure to resize
        """
        try:
            # Get current canvas size in pixels
            canvas_width_px = self.canvas.get_tk_widget().winfo_width()
            canvas_height_px = self.canvas.get_tk_widget().winfo_height()

            # Convert pixels to inches using the figure's DPI
            dpi = figure.dpi
            canvas_width_inches = canvas_width_px / dpi
            canvas_height_inches = canvas_height_px / dpi

            # Resize the figure to match canvas size
            figure.set_size_inches(canvas_width_inches, canvas_height_inches)

            # Ensure layout is updated
            figure.tight_layout()

        except Exception as e:
            print(f"Warning: Could not resize figure: {str(e)}")


    def _get_pipeline_figure(self, plot_type: str, parameters: dict):
        """
        Get figure from preprocessing pipeline based on plot type.

        Args:
            plot_type: Type of plot to create
            parameters: Parameters for the plot

        Returns:
            matplotlib.Figure or None: Figure object from pipeline
        """
        try:
            if plot_type == "psd":
                return self.preprocessing_pipeline.plot_power_spectral_density(
                    picks=parameters.get('picks', None),
                    fmin=parameters.get('fmin', 0.5),
                    fmax=parameters.get('fmax', 70.0),
                    title=parameters.get('title', 'Power Spectral Density'),
                )
            elif plot_type == "ica_components":
                return self.preprocessing_pipeline.plot_ica_components(
                    components=parameters.get('components', None),
                    title=parameters.get('title', 'ICA Components'),
                )
            else:
                return None

        except Exception as e:
            print(f"Error getting pipeline figure for {plot_type}: {str(e)}")
            return None

    def _show_error_message(self, message):
        """Show error message in the plot area."""
        # Create a simple error figure
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, message,
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

    def refresh_plot(self):
        """Refresh the current embedded analysis plot."""
        plot_type = self.plot_type_var.get()
        if plot_type:
            self.create_plot(plot_type, {})
        else:
            self.refresh_initial_plot()
            
    def update_plot(self):
        """Update the plot after preprocessing steps."""
        self.refresh_plot()