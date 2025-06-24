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

Main application class for the EEG preprocessing GUI.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys

# Add the parent directory to the path to import the preprocessing modules

from gui.preprocessing_viewer.preprocessing_frame import PreprocessingFrame
from gui.preprocessing_viewer.plot_frame import PreprocessingPlotFrame
from eeganalyzer.preprocessing.eeg_preprocessing_pipeline import EEGPreprocessor as EEGPreprocessingPipeline

class PreprocessingViewerApp:
    """Main application class for EEG preprocessing GUI."""
    
    def __init__(self, root):
        """
        Initialize the preprocessing application.
        
        Args:
            root: The main Tkinter window
        """
        self.root = root
        self.root.title("EEG Preprocessing Tool")
        self.root.geometry("1400x800")  # Wider for side-by-side layout
        
        # Initialize the preprocessing pipeline
        self.preprocessing_pipeline = None
        self.current_file = None
        
        # Create the main GUI components
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface with side-by-side layout."""
        # Create menu bar
        self.create_menu()
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid layout - similar to metrics viewer
        main_frame.grid_columnconfigure(0, weight=1)    # Controls panel
        main_frame.grid_columnconfigure(1, weight=3)    # Plot area (more space)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Create preprocessing controls frame (left side)
        self.preprocessing_frame = PreprocessingFrame(
            main_frame, 
            self.on_preprocessing_step_selected,
            self.on_plot_requested,
            width=350
        )
        self.preprocessing_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.preprocessing_frame.grid_propagate(False)  # Prevent resizing based on content
        
        # Create plot frame (right side)
        self.plot_frame = PreprocessingPlotFrame(main_frame, title="EEG Analysis")
        self.plot_frame.grid(row=0, column=1, padx=(0, 5), pady=5, sticky="nsew")
        
        # Status bar at bottom
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please load an EEG file")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        
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
        
        # View menu for direct plots
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Plot Raw Data", command=self.plot_raw_data_direct)
        view_menu.add_command(label="Plot ICA Sources", command=self.plot_ica_sources_direct)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def load_file(self):
        """Load an EEG file for preprocessing."""
        file_path = filedialog.askopenfilename(
            title="Select EEG File",
            filetypes=[
                ("EEG files", "*.edf *.fif *.set *.bdf *.gdf *.vhdr"),
                ("EDF files", "*.edf"),
                ("FIF files", "*.fif"),
                ("SET files", "*.set"),
                ("BDF files", "*.bdf"),
                ("GDF files", "*.gdf"),
                ("BrainVision files", "*.vhdr"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_file = file_path
                self.preprocessing_pipeline = EEGPreprocessingPipeline(file_path)

                # create channel categories
                self.preprocessing_pipeline.categorize_channels(mark_unclassified_as_bad=True)

                # Mark flat channels as bad
                self.preprocessing_pipeline.mark_bad_channels(self.preprocessing_pipeline.find_flat_channels_psd())
                
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
                self.preprocessing_pipeline.save_preprocessed(file_path, overwrite=True)
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
            # Apply the preprocessing step through the pipeline directly
            success = self._apply_preprocessing_step(step_name, parameters)
            
            if success:
                self.status_var.set(f"Applied: {step_name}")
                # Update embedded plot
                self.plot_frame.update_plot()
                # Update info display in preprocessing frame
                self.preprocessing_frame.update_info_display(self.preprocessing_pipeline)
                self.preprocessing_frame.update_history(self.preprocessing_pipeline.preprocessing_history)
            else:
                self.status_var.set(f"Failed to apply: {step_name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply preprocessing step: {str(e)}")
            
    def _apply_preprocessing_step(self, step_name: str, parameters: dict) -> bool:
        """
        Apply a preprocessing step to the pipeline directly.
        
        Args:
            step_name: Name of the preprocessing step
            parameters: Parameters for the step
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if step_name == "highpass":
                self.preprocessing_pipeline.apply_filter(l_freq=parameters["freq"], h_freq=None)
            elif step_name == "lowpass":
                self.preprocessing_pipeline.apply_filter(l_freq=None, h_freq=parameters["freq"])
            # elif step_name == "notch":
            #     self.preprocessing_pipeline.apply_notch_filter(freqs=parameters["freq"])
            elif step_name == "resample":
                self.preprocessing_pipeline.resample_data(sfreq=parameters["sfreq"])
            elif step_name == "fit_montage":
                self.preprocessing_pipeline.fit_montage(montage=parameters["montage"])
            elif step_name == "detect_bad_channels":
                pass
            #     self.preprocessing_pipeline.detect_artifacts_automatic()
            elif step_name == "fit_ica":
                self.preprocessing_pipeline.run_ica_fitting(start=parameters['start'], duration=parameters['duration'])
                self.on_plot_requested("ica_sources", {})
            elif step_name == "apply_ica":
                self.preprocessing_pipeline.run_ica_selection()
            else:
                return False
                
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply {step_name}: {str(e)}")
            return False
            
    def on_plot_requested(self, plot_type, parameters):
        """
        Handle embedded plot request (for analysis plots like PSD, ICA sources).
        
        Args:
            plot_type: Type of plot requested
            parameters: Dictionary of plot parameters
        """
        if not self.preprocessing_pipeline:
            messagebox.showwarning("Warning", "No data loaded")
            return
            
        try:
            # Request embedded plot
            self.plot_frame.create_plot(plot_type, parameters)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create plot: {str(e)}")
            
    # Direct plotting methods (using pipeline methods directly)
    def plot_raw_data_direct(self):
        """Plot raw data directly through pipeline."""
        if not self.preprocessing_pipeline:
            messagebox.showwarning("Warning", "No data loaded")
            return
            
        try:
            self.preprocessing_pipeline.plot_eeg_data(
                duration=20.0,
                n_channels=20,
                start=0.0,
                block=True,
                title="Raw EEG Data"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot raw data: {str(e)}")
            
    def plot_ica_sources_direct(self):
        """Plot ICA sources directly through pipeline."""
        if not self.preprocessing_pipeline:
            messagebox.showwarning("Warning", "No data loaded")
            return
            
        if not hasattr(self.preprocessing_pipeline, 'ica') or self.preprocessing_pipeline.ica is None:
            messagebox.showwarning("Warning", "ICA has not been computed yet. Run ICA first.")
            return
            
        try:
            self.preprocessing_pipeline.plot_ica_sources()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot ICA sources: {str(e)}")
            
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


def main():
    """Main entry point for the preprocessing GUI."""
    root = tk.Tk()
    app = PreprocessingViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()