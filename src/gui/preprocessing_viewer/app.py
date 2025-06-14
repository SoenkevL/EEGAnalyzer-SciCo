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

Main EEG Preprocessing GUI Application.

This module provides the main application window for EEG preprocessing.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys

# Add the parent directory to the path to import the preprocessing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from gui.preprocessing_viewer.preprocessing_frame import PreprocessingFrame
from gui.preprocessing_viewer.plot_frame import PreprocessingPlotFrame
from eeganalyzer.preprocessing.eeg_preprocessing_pipeline import EEGPreprocessor as EEGPreprocessingPipeline


class PreprocessingApp:
    """Main application class for EEG preprocessing GUI."""
    
    def __init__(self, root):
        """
        Initialize the preprocessing application.
        
        Args:
            root: The main Tkinter window
        """
        self.root = root
        self.root.title("EEG Preprocessing Tool")
        self.root.geometry("1200x800")
        
        # Initialize the preprocessing pipeline
        self.preprocessing_pipeline = None
        self.current_file = None
        
        # Create the main GUI components
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create menu bar
        self.create_menu()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create preprocessing tab
        self.preprocessing_frame = PreprocessingFrame(
            self.notebook, 
            self.on_preprocessing_step_selected,
            self.on_plot_requested
        )
        self.notebook.add(self.preprocessing_frame, text="Preprocessing")
        
        # Create plot tab
        self.plot_frame = PreprocessingPlotFrame(self.notebook)
        self.notebook.add(self.plot_frame, text="EEG Visualization")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please load an EEG file")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_menu(self):
        """Create the application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load EEG File", command=self.load_file)
        file_menu.add_separator()
        file_menu.add_command(label="Save Preprocessed Data", command=self.save_preprocessed_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def load_file(self):
        """Load an EEG file for preprocessing."""
        file_path = filedialog.askopenfilename(
            title="Select EEG File",
            filetypes=[
                ("EEG files", "*.edf *.fif *.set *.bdf"),
                ("EDF files", "*.edf"),
                ("FIF files", "*.fif"),
                ("SET files", "*.set"),
                ("BDF files", "*.bdf"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_file = file_path
                self.preprocessing_pipeline = EEGPreprocessingPipeline(file_path)
                
                # Update status
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                
                # Enable preprocessing controls
                self.preprocessing_frame.enable_controls()
                
                # Update plot frame with the loaded data
                self.plot_frame.set_preprocessing_pipeline(self.preprocessing_pipeline)
                
                messagebox.showinfo("Success", f"Successfully loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                self.status_var.set("Error loading file")
                
    def save_preprocessed_data(self):
        """Save the preprocessed EEG data."""
        if not self.preprocessing_pipeline:
            messagebox.showwarning("Warning", "No data loaded")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Preprocessed Data",
            defaultextension=".fif",
            filetypes=[
                ("FIF files", "*.fif"),
                ("EDF files", "*.edf"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Save the preprocessed data
                self.preprocessing_pipeline.save_preprocessed_data(file_path)
                messagebox.showinfo("Success", f"Data saved successfully to: {os.path.basename(file_path)}")
                self.status_var.set(f"Saved: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
                
    def on_preprocessing_step_selected(self, step_name, parameters):
        """
        Handle preprocessing step selection.
        
        Args:
            step_name: Name of the preprocessing step
            parameters: Dictionary of parameters for the step
        """
        if not self.preprocessing_pipeline:
            messagebox.showwarning("Warning", "No data loaded")
            return
            
        try:
            # Apply the preprocessing step
            success = self.preprocessing_frame.apply_preprocessing_step(
                self.preprocessing_pipeline, step_name, parameters
            )
            
            if success:
                self.status_var.set(f"Applied: {step_name}")
                # Update plot if needed
                self.plot_frame.update_plot()
            else:
                self.status_var.set(f"Failed to apply: {step_name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply preprocessing step: {str(e)}")
            
    def on_plot_requested(self, plot_type, parameters):
        """
        Handle plot request.
        
        Args:
            plot_type: Type of plot requested
            parameters: Dictionary of plot parameters
        """
        if not self.preprocessing_pipeline:
            messagebox.showwarning("Warning", "No data loaded")
            return
            
        try:
            # Switch to plot tab
            self.notebook.select(1)
            
            # Request plot
            self.plot_frame.create_plot(plot_type, parameters)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create plot: {str(e)}")
            
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About EEG Preprocessing Tool",
            "EEG Preprocessing Tool\n\n"
            "A GUI tool for preprocessing EEG data using MNE-Python.\n\n"
            "Features:\n"
            "- Load various EEG file formats\n"
            "- Apply preprocessing steps\n"
            "- Visualize EEG data\n"
            "- Save preprocessed data\n\n"
            "Copyright (C) 2025 Soenke van Loh"
        )