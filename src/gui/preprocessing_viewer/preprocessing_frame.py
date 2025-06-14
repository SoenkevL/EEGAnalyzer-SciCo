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

Preprocessing controls frame for EEG preprocessing GUI.

This module provides the GUI controls for EEG preprocessing operations.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Dict, Any


class PreprocessingFrame(ttk.Frame):
    """Frame containing preprocessing controls and options."""
    
    def __init__(self, parent, step_callback: Callable, plot_callback: Callable):
        """
        Initialize the preprocessing frame.
        
        Args:
            parent: Parent widget
            step_callback: Callback function for preprocessing steps
            plot_callback: Callback function for plot requests
        """
        super().__init__(parent)
        
        self.step_callback = step_callback
        self.plot_callback = plot_callback
        
        # Initialize control variables
        self.controls_enabled = False
        
        # Create the UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Create main container with scrollbar
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create preprocessing sections
        self.create_basic_info_section(scrollable_frame)
        self.create_filtering_section(scrollable_frame)
        self.create_resampling_section(scrollable_frame)
        self.create_artifact_section(scrollable_frame)
        self.create_montage_section(scrollable_frame)
        self.create_visualization_section(scrollable_frame)
        self.create_history_section(scrollable_frame)
        
    def create_basic_info_section(self, parent):
        """Create basic information section."""
        info_frame = ttk.LabelFrame(parent, text="Data Information", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Info display
        self.info_text = tk.Text(info_frame, height=4, width=50, state=tk.DISABLED)
        self.info_text.pack(fill=tk.X)
        
    def create_filtering_section(self, parent):
        """Create filtering controls section."""
        filter_frame = ttk.LabelFrame(parent, text="Filtering", padding=10)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # High-pass filter
        hp_frame = ttk.Frame(filter_frame)
        hp_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(hp_frame, text="High-pass (Hz):").pack(side=tk.LEFT)
        self.hp_var = tk.DoubleVar(value=0.5)
        hp_entry = ttk.Entry(hp_frame, textvariable=self.hp_var, width=10)
        hp_entry.pack(side=tk.LEFT, padx=5)
        
        self.hp_button = ttk.Button(
            hp_frame, 
            text="Apply High-pass", 
            command=self.apply_highpass_filter,
            state=tk.DISABLED
        )
        self.hp_button.pack(side=tk.LEFT, padx=5)
        
        # Low-pass filter
        lp_frame = ttk.Frame(filter_frame)
        lp_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(lp_frame, text="Low-pass (Hz):").pack(side=tk.LEFT)
        self.lp_var = tk.DoubleVar(value=50.0)
        lp_entry = ttk.Entry(lp_frame, textvariable=self.lp_var, width=10)
        lp_entry.pack(side=tk.LEFT, padx=5)
        
        self.lp_button = ttk.Button(
            lp_frame, 
            text="Apply Low-pass", 
            command=self.apply_lowpass_filter,
            state=tk.DISABLED
        )
        self.lp_button.pack(side=tk.LEFT, padx=5)
        
        # Notch filter
        notch_frame = ttk.Frame(filter_frame)
        notch_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(notch_frame, text="Notch (Hz):").pack(side=tk.LEFT)
        self.notch_var = tk.DoubleVar(value=50.0)
        notch_entry = ttk.Entry(notch_frame, textvariable=self.notch_var, width=10)
        notch_entry.pack(side=tk.LEFT, padx=5)
        
        self.notch_button = ttk.Button(
            notch_frame, 
            text="Apply Notch", 
            command=self.apply_notch_filter,
            state=tk.DISABLED
        )
        self.notch_button.pack(side=tk.LEFT, padx=5)
        
    def create_resampling_section(self, parent):
        """Create resampling controls section."""
        resample_frame = ttk.LabelFrame(parent, text="Resampling", padding=10)
        resample_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(resample_frame, text="New sampling rate (Hz):").pack(side=tk.LEFT)
        self.resample_var = tk.DoubleVar(value=250.0)
        resample_entry = ttk.Entry(resample_frame, textvariable=self.resample_var, width=10)
        resample_entry.pack(side=tk.LEFT, padx=5)
        
        self.resample_button = ttk.Button(
            resample_frame, 
            text="Resample", 
            command=self.apply_resampling,
            state=tk.DISABLED
        )
        self.resample_button.pack(side=tk.LEFT, padx=5)
        
    def create_artifact_section(self, parent):
        """Create artifact removal section."""
        artifact_frame = ttk.LabelFrame(parent, text="Artifact Removal", padding=10)
        artifact_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Bad channel detection
        bad_ch_frame = ttk.Frame(artifact_frame)
        bad_ch_frame.pack(fill=tk.X, pady=2)
        
        self.bad_ch_button = ttk.Button(
            bad_ch_frame, 
            text="Detect Bad Channels", 
            command=self.detect_bad_channels,
            state=tk.DISABLED
        )
        self.bad_ch_button.pack(side=tk.LEFT, padx=5)
        
        # ICA
        ica_frame = ttk.Frame(artifact_frame)
        ica_frame.pack(fill=tk.X, pady=2)
        
        self.ica_button = ttk.Button(
            ica_frame, 
            text="Run ICA", 
            command=self.run_ica,
            state=tk.DISABLED
        )
        self.ica_button.pack(side=tk.LEFT, padx=5)
        
    def create_montage_section(self, parent):
        """Create montage controls section."""
        montage_frame = ttk.LabelFrame(parent, text="Montage & Channels", padding=10)
        montage_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Montage selection
        montage_sel_frame = ttk.Frame(montage_frame)
        montage_sel_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(montage_sel_frame, text="Montage:").pack(side=tk.LEFT)
        self.montage_var = tk.StringVar(value="standard_1020")
        montage_combo = ttk.Combobox(
            montage_sel_frame, 
            textvariable=self.montage_var,
            values=["standard_1020", "standard_1005", "biosemi64", "biosemi128"],
            state="readonly"
        )
        montage_combo.pack(side=tk.LEFT, padx=5)
        
        self.montage_button = ttk.Button(
            montage_sel_frame, 
            text="Set Montage", 
            command=self.set_montage,
            state=tk.DISABLED
        )
        self.montage_button.pack(side=tk.LEFT, padx=5)
        
    def create_visualization_section(self, parent):
        """Create visualization controls section."""
        viz_frame = ttk.LabelFrame(parent, text="Visualization", padding=10)
        viz_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Plot parameters
        plot_params_frame = ttk.Frame(viz_frame)
        plot_params_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(plot_params_frame, text="Duration (s):").pack(side=tk.LEFT)
        self.duration_var = tk.DoubleVar(value=20.0)
        duration_entry = ttk.Entry(plot_params_frame, textvariable=self.duration_var, width=8)
        duration_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(plot_params_frame, text="Channels:").pack(side=tk.LEFT, padx=(10, 0))
        self.n_channels_var = tk.IntVar(value=20)
        channels_entry = ttk.Entry(plot_params_frame, textvariable=self.n_channels_var, width=8)
        channels_entry.pack(side=tk.LEFT, padx=5)
        
        # Plot buttons
        plot_buttons_frame = ttk.Frame(viz_frame)
        plot_buttons_frame.pack(fill=tk.X, pady=5)
        
        self.plot_raw_button = ttk.Button(
            plot_buttons_frame, 
            text="Plot Raw Data", 
            command=self.plot_raw_data,
            state=tk.DISABLED
        )
        self.plot_raw_button.pack(side=tk.LEFT, padx=5)
        
        self.plot_psd_button = ttk.Button(
            plot_buttons_frame, 
            text="Plot PSD", 
            command=self.plot_psd,
            state=tk.DISABLED
        )
        self.plot_psd_button.pack(side=tk.LEFT, padx=5)
        
    def create_history_section(self, parent):
        """Create preprocessing history section."""
        history_frame = ttk.LabelFrame(parent, text="Preprocessing History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # History listbox with scrollbar
        list_frame = ttk.Frame(history_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.history_listbox = tk.Listbox(list_frame)
        history_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.history_listbox.yview)
        self.history_listbox.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_listbox.pack(side="left", fill="both", expand=True)
        history_scrollbar.pack(side="right", fill="y")
        
    def enable_controls(self):
        """Enable all preprocessing controls."""
        self.controls_enabled = True
        
        # Enable all buttons
        buttons = [
            self.hp_button, self.lp_button, self.notch_button,
            self.resample_button, self.bad_ch_button, self.ica_button,
            self.montage_button, self.plot_raw_button, self.plot_psd_button
        ]
        
        for button in buttons:
            button.config(state=tk.NORMAL)
            
    def apply_preprocessing_step(self, pipeline, step_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Apply a preprocessing step to the pipeline.
        
        Args:
            pipeline: The preprocessing pipeline
            step_name: Name of the preprocessing step
            parameters: Parameters for the step
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if step_name == "highpass":
                pipeline.apply_filter(l_freq=parameters["freq"], h_freq=None)
            elif step_name == "lowpass":
                pipeline.apply_filter(l_freq=None, h_freq=parameters["freq"])
            elif step_name == "notch":
                pipeline.apply_notch_filter(freqs=parameters["freq"])
            elif step_name == "resample":
                pipeline.resample(sfreq=parameters["sfreq"])
            elif step_name == "fit_montage":
                pipeline.fit_montage(montage=parameters["montage"])
            elif step_name == "detect_bad_channels":
                pipeline.detect_bad_channels()
            elif step_name == "run_ica":
                pipeline.run_ica()
            else:
                return False
                
            # Update history
            self.update_history(pipeline.preprocessing_history)
            self.update_info_display(pipeline)
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply {step_name}: {str(e)}")
            return False
            
    def update_history(self, history_list):
        """Update the preprocessing history display."""
        self.history_listbox.delete(0, tk.END)
        for item in history_list:
            self.history_listbox.insert(tk.END, item)
            
    def update_info_display(self, pipeline):
        """Update the data information display."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        if pipeline and pipeline.raw:
            info_text = f"Sampling Rate: {pipeline.raw.info['sfreq']} Hz\n"
            info_text += f"Channels: {len(pipeline.raw.ch_names)}\n"
            info_text += f"Duration: {pipeline.raw.times[-1]:.2f} seconds\n"
            info_text += f"Bad Channels: {len(pipeline.raw.info['bads'])}"
            
            self.info_text.insert(1.0, info_text)
            
        self.info_text.config(state=tk.DISABLED)
        
    # Preprocessing step methods
    def apply_highpass_filter(self):
        """Apply high-pass filter."""
        freq = self.hp_var.get()
        self.step_callback("highpass", {"freq": freq})
        
    def apply_lowpass_filter(self):
        """Apply low-pass filter."""
        freq = self.lp_var.get()
        self.step_callback("lowpass", {"freq": freq})
        
    def apply_notch_filter(self):
        """Apply notch filter."""
        freq = self.notch_var.get()
        self.step_callback("notch", {"freq": freq})
        
    def apply_resampling(self):
        """Apply resampling."""
        sfreq = self.resample_var.get()
        self.step_callback("resample", {"sfreq": sfreq})
        
    def detect_bad_channels(self):
        """Detect bad channels."""
        self.step_callback("detect_bad_channels", {})
        
    def run_ica(self):
        """Run ICA."""
        self.step_callback("run_ica", {})
        
    def set_montage(self):
        """Set montage."""
        montage = self.montage_var.get()
        self.step_callback("fit_montage", {"montage": montage})
        
    def plot_raw_data(self):
        """Plot raw EEG data."""
        parameters = {
            "duration": self.duration_var.get(),
            "n_channels": self.n_channels_var.get(),
            "start": 0
        }
        self.plot_callback("raw", parameters)
        
    def plot_psd(self):
        """Plot power spectral density."""
        self.plot_callback("psd", {})