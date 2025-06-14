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
focusing on embedded matplotlib plots for analysis metrics and using MNE's 
native Qt plotting for raw data and components.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import subprocess
import sys
import os
import tempfile


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
        
        # Plot type selection - only for embedded matplotlib plots
        ttk.Label(control_frame, text="Analysis Plot:").pack(side=tk.LEFT)
        self.plot_type_var = tk.StringVar(value="psd")
        plot_type_combo = ttk.Combobox(
            control_frame,
            textvariable=self.plot_type_var,
            values=["psd", "channel_stats", "preprocessing_summary"],
            state="readonly",
            width=20
        )
        plot_type_combo.pack(side=tk.LEFT, padx=5)
        
        # Refresh button for embedded plots
        refresh_button = ttk.Button(
            control_frame,
            text="Update Analysis Plot",
            command=self.refresh_plot
        )
        refresh_button.pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(control_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # MNE Native plotting buttons
        ttk.Label(control_frame, text="MNE Plots:").pack(side=tk.LEFT)
        
        plot_raw_button = ttk.Button(
            control_frame,
            text="Plot Raw Data",
            command=self.plot_raw_data
        )
        plot_raw_button.pack(side=tk.LEFT, padx=5)
        
        plot_ica_button = ttk.Button(
            control_frame,
            text="Plot ICA Components",
            command=self.plot_ica_components
        )
        plot_ica_button.pack(side=tk.LEFT, padx=5)
        
        # Create matplotlib frame for analysis plots only
        self.create_matplotlib_frame()
        
    def create_matplotlib_frame(self):
        """Create the matplotlib plotting frame for analysis plots."""
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
        self.ax.text(0.5, 0.5, 'Load EEG data to view analysis plots\n\nUse "Plot Raw Data" button for EEG time series visualization', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=14, alpha=0.7)
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
        Create an analysis plot of the specified type (embedded matplotlib only).
        
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
            # Clear the current plot
            self.ax.clear()
            
            raw = self.preprocessing_pipeline.raw
            
            if plot_type == "psd":
                self._plot_psd(raw, parameters)
            elif plot_type == "channel_stats":
                self._plot_channel_stats(raw, parameters)
            elif plot_type == "preprocessing_summary":
                self._plot_preprocessing_summary(raw, parameters)
            else:
                self.ax.text(0.5, 0.5, f'Analysis plot type "{plot_type}" not implemented', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=self.ax.transAxes, fontsize=16)
                
            # Update the canvas
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create analysis plot: {str(e)}")
            self.ax.clear()
            self.ax.text(0.5, 0.5, f'Error creating plot: {str(e)}', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=self.ax.transAxes, fontsize=12, color='red')
            self.canvas.draw()
            
    def _plot_psd(self, raw, parameters):
        """Plot power spectral density (embedded matplotlib)."""
        try:
            # Compute PSD for a subset of channels to avoid overcrowding
            picks = raw.pick_types(eeg=True, meg=False, exclude='bads')
            n_channels = min(10, len(picks.ch_names))  # Limit to 10 channels
            selected_picks = picks.ch_names[:n_channels]
            
            # Compute PSD
            psd_data = raw.compute_psd(picks=selected_picks, fmax=100)
            freqs = psd_data.freqs
            psd_values = psd_data.get_data()
            
            # Plot PSD for each channel
            for i, ch_name in enumerate(selected_picks):
                self.ax.semilogy(freqs, psd_values[i], label=ch_name, alpha=0.7)
                
            self.ax.set_xlabel('Frequency (Hz)')
            self.ax.set_ylabel('Power Spectral Density (V²/Hz)')
            self.ax.set_title(f'Power Spectral Density - {n_channels} channels')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.fig.tight_layout()
            
        except Exception as e:
            raise Exception(f"Error plotting PSD: {str(e)}")
            
    def _plot_channel_stats(self, raw, parameters):
        """Plot channel statistics (embedded matplotlib)."""
        try:
            # Get data for statistics
            data = raw.get_data()
            ch_names = raw.ch_names
            
            # Compute basic statistics
            means = np.mean(data, axis=1)
            stds = np.std(data, axis=1)
            
            # Create subplots
            self.ax.clear()
            self.fig.clear()
            
            ax1 = self.fig.add_subplot(2, 1, 1)
            ax2 = self.fig.add_subplot(2, 1, 2)
            
            # Plot means
            x_pos = np.arange(len(ch_names))
            ax1.bar(x_pos, means * 1e6, alpha=0.7)  # Convert to microvolts
            ax1.set_ylabel('Mean Amplitude (µV)')
            ax1.set_title('Channel Mean Amplitudes')
            ax1.set_xticks(x_pos[::max(1, len(ch_names)//10)])
            ax1.set_xticklabels([ch_names[i] for i in range(0, len(ch_names), max(1, len(ch_names)//10))], rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Plot standard deviations
            ax2.bar(x_pos, stds * 1e6, alpha=0.7, color='orange')  # Convert to microvolts
            ax2.set_ylabel('Standard Deviation (µV)')
            ax2.set_xlabel('Channels')
            ax2.set_title('Channel Standard Deviations')
            ax2.set_xticks(x_pos[::max(1, len(ch_names)//10)])
            ax2.set_xticklabels([ch_names[i] for i in range(0, len(ch_names), max(1, len(ch_names)//10))], rotation=45)
            ax2.grid(True, alpha=0.3)
            
            self.fig.tight_layout()
            
        except Exception as e:
            raise Exception(f"Error plotting channel statistics: {str(e)}")
            
    def _plot_preprocessing_summary(self, raw, parameters):
        """Plot preprocessing summary information (embedded matplotlib)."""
        try:
            self.ax.clear()
            
            # Get preprocessing history
            history = getattr(self.preprocessing_pipeline, 'preprocessing_history', [])
            
            # Create summary text
            summary_text = f"EEG Data Summary\n\n"
            summary_text += f"Sampling Rate: {raw.info['sfreq']:.1f} Hz\n"
            summary_text += f"Number of Channels: {len(raw.ch_names)}\n"
            summary_text += f"Duration: {raw.times[-1]:.1f} seconds\n"
            summary_text += f"Bad Channels: {len(raw.info['bads'])}\n"
            if raw.info['bads']:
                summary_text += f"Bad Channel Names: {', '.join(raw.info['bads'])}\n"
            
            summary_text += f"\n\nPreprocessing Steps Applied:\n"
            if history:
                for i, step in enumerate(history, 1):
                    summary_text += f"{i}. {step}\n"
            else:
                summary_text += "No preprocessing steps applied yet.\n"
                
            # Display summary as text
            self.ax.text(0.05, 0.95, summary_text, 
                        horizontalalignment='left', verticalalignment='top',
                        transform=self.ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title('Preprocessing Summary')
            
        except Exception as e:
            raise Exception(f"Error creating preprocessing summary: {str(e)}")
            
    def plot_raw_data(self):
        """Plot raw EEG data using MNE's native Qt plotting via subprocess."""
        if not self.preprocessing_pipeline or not self.preprocessing_pipeline.raw:
            messagebox.showwarning("Warning", "No EEG data loaded")
            return
            
        try:
            # Create a temporary file for the raw data
            with tempfile.NamedTemporaryFile(suffix='.fif', delete=False) as tmp_file:
                temp_filepath = tmp_file.name
                
            # Save the current raw data to the temporary file
            self.preprocessing_pipeline.raw.save(temp_filepath, overwrite=True)
            
            # Get the path to the MNE plot helper script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            helper_script = os.path.join(script_dir, '..', 'metrics_viewer', 'mne_plot_helper.py')
            
            if not os.path.exists(helper_script):
                messagebox.showerror("Error", f"MNE plot helper script not found at: {helper_script}")
                return
                
            # Build the command to run the helper script
            cmd = [
                sys.executable,
                helper_script,
                '--filepath', temp_filepath,
                '--title', 'EEG Raw Data - Preprocessing Viewer'
            ]
            
            # Start the subprocess (non-blocking)
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Note: We don't delete the temp file immediately as the subprocess needs it
            # The OS will clean it up eventually
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open raw data plot: {str(e)}")
            # Clean up temp file on error
            try:
                if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                    os.unlink(temp_filepath)
            except:
                pass
                
    def plot_ica_components(self):
        """Plot ICA components using MNE's native Qt plotting."""
        if not self.preprocessing_pipeline:
            messagebox.showwarning("Warning", "No preprocessing pipeline loaded")
            return
            
        if not hasattr(self.preprocessing_pipeline, 'ica') or self.preprocessing_pipeline.ica is None:
            messagebox.showwarning("Warning", "ICA has not been computed yet. Run ICA first.")
            return
            
        try:
            # Use MNE's built-in ICA component plotting with non-blocking display
            self.preprocessing_pipeline.ica.plot_components(
                show=True,
                title="ICA Components - Preprocessing Viewer"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open ICA components plot: {str(e)}")
            
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