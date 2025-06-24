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

Preprocessing frame for EEG preprocessing GUI.

This module provides the PreprocessingFrame class containing all preprocessing
controls and options.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Any


class PreprocessingFrame(ttk.Frame):
    """Frame containing preprocessing controls and options."""
    
    def __init__(self, parent, step_callback: Callable, plot_callback: Callable, **kwargs):
        """
        Initialize the preprocessing frame.
        
        Args:
            parent: Parent widget
            step_callback: Callback function for preprocessing steps
            plot_callback: Callback function for plot requests
            **kwargs: Additional arguments for the Frame constructor
        """
        super().__init__(parent, **kwargs)
        
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
        self.create_montage_section(scrollable_frame)
        self.create_filtering_section(scrollable_frame)
        self.create_resampling_section(scrollable_frame)
        self.create_artifact_section(scrollable_frame)
        self.create_history_section(scrollable_frame)
        
    def create_basic_info_section(self, parent):
        """Create basic information section."""
        info_frame = ttk.LabelFrame(parent, text="Data Information", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Info display
        self.info_text = tk.Text(info_frame, height=4, width=40, state=tk.DISABLED)
        self.info_text.pack(fill=tk.X)
        
    def create_filtering_section(self, parent):
        """Create filtering controls section."""
        filter_frame = ttk.LabelFrame(parent, text="Filtering", padding=10)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # High-pass filter
        hp_frame = ttk.Frame(filter_frame)
        hp_frame.pack(fill=tk.X, pady=2)
        
        # Create horizontal layout with grid
        hp_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(hp_frame, text="High-pass (Hz):").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.hp_var = tk.DoubleVar(value=0.5)
        hp_entry = ttk.Entry(hp_frame, textvariable=self.hp_var, width=8)
        hp_entry.grid(row=0, column=1, sticky="w", padx=(0, 5))
        
        self.hp_button = ttk.Button(
            hp_frame, 
            text="Apply", 
            command=self.apply_highpass_filter,
            state=tk.DISABLED,
            width=8
        )
        self.hp_button.grid(row=0, column=2, sticky="w")
        
        # Low-pass filter
        lp_frame = ttk.Frame(filter_frame)
        lp_frame.pack(fill=tk.X, pady=2)
        
        # Create horizontal layout with grid
        lp_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(lp_frame, text="Low-pass (Hz):").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.lp_var = tk.DoubleVar(value=40.0)
        lp_entry = ttk.Entry(lp_frame, textvariable=self.lp_var, width=8)
        lp_entry.grid(row=0, column=1, sticky="w", padx=(0, 5))
        
        self.lp_button = ttk.Button(
            lp_frame, 
            text="Apply", 
            command=self.apply_lowpass_filter,
            state=tk.DISABLED,
            width=8
        )
        self.lp_button.grid(row=0, column=2, sticky="w")
        
        # # Notch filter
        # notch_frame = ttk.Frame(filter_frame)
        # notch_frame.pack(fill=tk.X, pady=2)
        #
        # # Create horizontal layout with grid
        # notch_frame.grid_columnconfigure(1, weight=1)
        #
        # ttk.Label(notch_frame, text="Notch (Hz):").grid(row=0, column=0, sticky="w", padx=(0, 5))
        # self.notch_var = tk.DoubleVar(value=50.0)
        # notch_entry = ttk.Entry(notch_frame, textvariable=self.notch_var, width=8)
        # notch_entry.grid(row=0, column=1, sticky="w", padx=(0, 5))
        #
        # self.notch_button = ttk.Button(
        #     notch_frame,
        #     text="Apply",
        #     command=self.apply_notch_filter,
        #     state=tk.DISABLED,
        #     width=8
        # )
        # self.notch_button.grid(row=0, column=2, sticky="w")
        
    def create_resampling_section(self, parent):
        """Create resampling controls section."""
        resample_frame = ttk.LabelFrame(parent, text="Resampling", padding=10)
        resample_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create horizontal layout
        controls_frame = ttk.Frame(resample_frame)
        controls_frame.pack(fill=tk.X)
        controls_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(controls_frame, text="Sampling rate (Hz):").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.resample_var = tk.DoubleVar(value=1450.0)
        resample_entry = ttk.Entry(controls_frame, textvariable=self.resample_var, width=8)
        resample_entry.grid(row=0, column=1, sticky="w", padx=(0, 5))
        
        self.resample_button = ttk.Button(
            controls_frame, 
            text="Apply", 
            command=self.apply_resampling,
            state=tk.DISABLED,
            width=8
        )
        self.resample_button.grid(row=0, column=2, sticky="w")
        
    def create_artifact_section(self, parent):
        """Create artifact removal section."""
        artifact_frame = ttk.LabelFrame(parent, text="Artifact Removal", padding=10)
        artifact_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # # Bad channel detection
        # self.bad_ch_button = ttk.Button(
        #     artifact_frame,
        #     text="Detect Bad Channels",
        #     command=self.detect_bad_channels,
        #     state=tk.DISABLED
        # )
        # self.bad_ch_button.pack(anchor=tk.W, pady=2)
        #
        # ICA controls in horizontal layout
        ica_frame = ttk.Frame(artifact_frame)
        ica_frame.pack(fill=tk.X, pady=2)
        
        self.fit_ica_button = ttk.Button(
            ica_frame, 
            text="Fit ICA", 
            command=self.fit_ica,
            state=tk.DISABLED,
            width=12
        )
        self.fit_ica_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.apply_ica_button = ttk.Button(
            ica_frame, 
            text="Apply ICA", 
            command=self.apply_ica,
            state=tk.DISABLED,
            width=12
        )
        self.apply_ica_button.pack(side=tk.LEFT)
        
    def create_montage_section(self, parent):
        """Create montage controls section."""
        montage_frame = ttk.LabelFrame(parent, text="Montage & Channels", padding=10)
        montage_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create horizontal layout
        controls_frame = ttk.Frame(montage_frame)
        controls_frame.pack(fill=tk.X)
        controls_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(controls_frame, text="Montage:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.montage_var = tk.StringVar(value="standard_1020")
        montage_combo = ttk.Combobox(
            controls_frame, 
            textvariable=self.montage_var,
            values=["standard_1020", "standard_1005", "biosemi64", "biosemi128"],
            state="readonly",
            width=12
        )
        montage_combo.grid(row=0, column=1, sticky="w", padx=(0, 5))
        
        self.montage_button = ttk.Button(
            controls_frame, 
            text="Apply", 
            command=self.set_montage,
            state=tk.DISABLED,
            width=8
        )
        self.montage_button.grid(row=0, column=2, sticky="w")
        
    def create_history_section(self, parent):
        """Create preprocessing history section."""
        history_frame = ttk.LabelFrame(parent, text="Preprocessing History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # History listbox with scrollbar
        list_frame = ttk.Frame(history_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.history_listbox = tk.Listbox(list_frame, height=6)
        history_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.history_listbox.yview)
        self.history_listbox.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_listbox.pack(side="left", fill="both", expand=True)
        history_scrollbar.pack(side="right", fill="y")
        
    def enable_controls(self):
        """Enable all preprocessing controls."""
        self.controls_enabled = True
        
        # Enable all buttons
        buttons = [
            self.hp_button, self.lp_button,
            self.resample_button, self.fit_ica_button,
            self.apply_ica_button, self.montage_button,
            # self.bad_ch_button
        ]
        
        for button in buttons:
            button.config(state=tk.NORMAL)
            
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
        
    # def apply_notch_filter(self):
    #     """Apply notch filter."""
    #     freq = self.notch_var.get()
    #     self.step_callback("notch", {"freq": freq})
        
    def apply_resampling(self):
        """Apply resampling."""
        sfreq = self.resample_var.get()
        self.step_callback("resample", {"sfreq": sfreq})
        
    def detect_bad_channels(self):
        """Detect bad channels."""
        self.step_callback("detect_bad_channels", {})
        
    def fit_ica(self):
        """Fit ICA and plot sources."""
        self.step_callback("fit_ica", {})
        # Request plotting of ICA sources after fitting
        self.plot_callback("ica_components", {})

    def apply_ica(self):
        """Apply ICA to remove selected components."""
        self.step_callback("apply_ica", {})
        
    def set_montage(self):
        """Set montage."""
        montage = self.montage_var.get()
        self.step_callback("fit_montage", {"montage": montage})