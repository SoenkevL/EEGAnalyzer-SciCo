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
from gui.preprocessing_viewer.ica_properties_popup_window import ICAPropertiesPopup
import numpy as np


class PreprocessingPlotFrame(ttk.Frame):
    """Frame containing embedded EEG analysis plots."""
    
    def __init__(self, parent, title="EEG Analysis", preprocessing_frame=None, **kwargs):
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
        self.preprocessing_frame = preprocessing_frame
        self.event_handlers_enabled = False  # Default to enabled

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
        
        # Separator
        separator = ttk.Separator(control_frame, orient='vertical')
        separator.pack(side=tk.LEFT, fill='y', padx=10)
        
        # Direct view buttons
        ttk.Label(control_frame, text="Direct Views:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.plot_raw_button = ttk.Button(
            control_frame,
            text="Raw Data",
            command=self.plot_raw_data_direct,
            state=tk.DISABLED
        )
        self.plot_raw_button.pack(side=tk.LEFT, padx=2)
        
        # Add checkbox for DC removal
        self.remove_dc_var = tk.BooleanVar(value=False)
        self.remove_dc_checkbox = ttk.Checkbutton(
            control_frame,
            text="Remove DC",
            variable=self.remove_dc_var,
            command=self.on_remove_dc_changed
        )
        self.remove_dc_checkbox.pack(side=tk.LEFT, padx=(2, 8))
        
        self.plot_ica_sources_button = ttk.Button(
            control_frame,
            text="ICA Sources",
            command=self.plot_ica_sources_direct,
            state=tk.DISABLED
        )
        self.plot_ica_sources_button.pack(side=tk.LEFT, padx=2)
        
        # Separator for PSD range controls
        separator2 = ttk.Separator(control_frame, orient='vertical')
        separator2.pack(side=tk.LEFT, fill='y', padx=10)
        
        # Add checkbox for using artifact detection time range in PSD
        self.use_artifact_range_var = tk.BooleanVar(value=False)
        self.use_artifact_range_checkbox = ttk.Checkbutton(
            control_frame,
            text="Use Artifact Range for PSD",
            variable=self.use_artifact_range_var,
            command=self.on_artifact_range_toggle
        )
        self.use_artifact_range_checkbox.pack(side=tk.LEFT, padx=(0, 5))
        
        # Create matplotlib frame
        self.create_matplotlib_frame()
    # Create matplotlibframe    
    def create_matplotlib_frame(self):
        """Create the matplotlib plotting frame."""
        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Make canvas focusable and set focus
        canvas_widget.configure(highlightthickness=1)
        canvas_widget.focus_set()
        
        # Create a separate frame for the toolbar to avoid geometry manager conflicts
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=2, column=0, sticky="ew", padx=5)
        
        # Create toolbar in the separate frame
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Initial empty plot
        self.ax.text(0.5, 0.5, 'Load EEG data to view analysis plots\n\nUse direct view buttons for raw data and ICA plots', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=12, alpha=0.7)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Set up event handlers AFTER canvas is created and configured
        self.setup_event_handlers()

    # Event handlers
    def setup_event_handlers(self):
        """Set up matplotlib event handlers for the canvas."""
        print("Setting up event handlers...")  # Debug print
        
        # Disconnect any existing connections first
        self.disconnect_event_handlers()
        
        # Only connect if event handlers are enabled
        if not getattr(self, 'event_handlers_enabled', True):
            print("Event handlers are disabled, skipping setup")
            return
        
        # Connect events to the canvas
        self.event_connections = {}
        
        try:
            self.event_connections['click'] = self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
            self.event_connections['release'] = self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
            self.event_connections['motion'] = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            
            print(f"Event handlers connected: {list(self.event_connections.keys())}")  # Debug print
            
        except Exception as e:
            print(f"Error connecting event handlers: {e}")

    def on_mouse_click(self, event):
        """
        Handle mouse click events on the plot.
        
        Args:
            event: matplotlib MouseEvent containing click information
        """
        print(f"Mouse click detected! Button: {event.button}, inaxes: {event.inaxes}")  # Debug print
        
        if event.inaxes is None:
            print("Click was outside axes")  # Debug print
            return  # Click was outside any axes
        
        # Only handle left mouse button clicks
        if event.button != 1:  # 1 = left mouse button
            print(f"Ignoring non-left click: {event.button}")  # Debug print
            return
        
        print(f"Processing click for plot type: {self.current_plot_type}")  # Debug print
        
        # Handle different plot types
        if self.current_plot_type == "ica_components":
            self.handle_ica_components_click(event)
        elif self.current_plot_type == "psd":
            self.handle_psd_click(event)
        else:
            print(f"Generic click at coordinates: ({event.xdata}, {event.ydata})")

        print(f"Mouse released at: x={event.xdata:.2f}, y={event.ydata:.2f}")

    def on_mouse_release(self, event):
        """Handle mouse release events."""
        if event.inaxes is None:
            return
        print(f"Mouse released at: x={event.xdata:.2f}, y={event.ydata:.2f}")

    def on_mouse_move(self, event):
        """Handle mouse move events (optional - can be used for hover effects)."""
        if event.inaxes is None:
            return

        # Uncomment the line below if you want to track mouse movement
        # print(f"Mouse moved to: x={event.xdata:.2f}, y={event.ydata:.2f}")

    def handle_psd_click(self, event):
        """
        Handle clicks on Power Spectral Density plots.
        
        Args:
            event: matplotlib MouseEvent
        """
        print("Handling PSD click...")  # Debug print
        
        try:
            if event.xdata is not None and event.ydata is not None:
                freq = event.xdata
                power = event.ydata
                
                print(f"PSD Click - Frequency: {freq:.2f} Hz, Power: {power:.2f}")
                
                # Show frequency information
                messagebox.showinfo(
                    "PSD Click", 
                    f"Clicked frequency: {freq:.2f} Hz\nPower: {power:.2f} dB"
                )
        
        except Exception as e:
            print(f"Error handling PSD click: {e}")

    def handle_ica_components_click(self, event):
        """
        Handle clicks on ICA component plots and open properties popup.
        
        Args:
            event: matplotlib MouseEvent
        """
        print("Handling ICA components click...")  # Debug print
        
        if not self.preprocessing_pipeline or not hasattr(self.preprocessing_pipeline, 'ica') or self.preprocessing_pipeline.ica is None:
            messagebox.showwarning("Warning", "ICA has not been computed yet.")
            return
        
        try:
            # Find which subplot/component was clicked
            clicked_axes = event.inaxes
            all_axes = self.fig.get_axes()
            
            component_index = None
            for i, ax in enumerate(all_axes):
                if ax == clicked_axes:
                    component_index = i
                    break
            
            print(f"Found component index: {component_index}")  # Debug print
            
            if component_index is not None:
                messagebox.showinfo("ICA Click", f"Clicked on ICA component {component_index}")
                # Show properties popup
                self.show_ica_properties_popup(component_index)
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to handle ICA component click: {str(e)}")
            print(f"Error handling ICA component click: {e}")

    def handle_generic_click(self, event, axes):
        """
        Handle clicks on generic plots.
        
        Args:
            event: matplotlib MouseEvent
            axes: The axes that was clicked
        """
        try:
            print(f"Generic Click - Position: x={event.xdata:.2f}, y={event.ydata:.2f}")

            # Add your custom logic here
            # For example, you could:
            # - Show data point information
            # - Add annotations
            # - Trigger other GUI actions

        except Exception as e:
            print(f"Error handling generic click: {e}")

    def show_ica_properties_popup(self, component_index):
        """
        Show ICA properties in a popup window.
        
        Args:
            component_index: Index of the ICA component to show properties for
        """
        try:
            print(f"Opening ICA properties popup for component {component_index}")  # Debug print
            
            # Get ICA properties figures from preprocessing pipeline
            properties_figures = self.preprocessing_pipeline.plot_ica_properties(component=component_index)
            
            if properties_figures:
                # Find the root window
                root_window = self.winfo_toplevel()
                
                # Create and show popup
                popup = ICAPropertiesPopup(
                    parent=root_window,
                    figures=properties_figures,
                    component_index=component_index
                )
                print(f"Opened ICA properties popup for component {component_index}")
            else:
                messagebox.showwarning(
                    "Warning", 
                    f"No properties data available for ICA component {component_index}"
                )
                
        except Exception as e:
            messagebox.showerror(
                "Error", 
                f"Failed to show ICA properties for component {component_index}: {str(e)}"
            )
            print(f"Error showing ICA properties popup: {e}")

    def disconnect_event_handlers(self):
        """Disconnect all event handlers (useful for cleanup)."""
        if hasattr(self, 'event_connections'):
            for name, connection_id in self.event_connections.items():
                try:
                    self.canvas.mpl_disconnect(connection_id)
                    print(f"Disconnected {name} event handler")  # Debug print
                except Exception as e:
                    print(f"Error disconnecting {name} handler: {e}")
            self.event_connections.clear()

    # Highlighting for selected subplots (not functional)
    # TODO: look into why this isnt working yet
    def add_subplot_click_highlight(self, axes, color='red', alpha=0.3):
        """
        Add a highlight overlay to a clicked subplot.

        Args:
            axes: The matplotlib axes to highlight
            color: Color of the highlight
            alpha: Transparency of the highlight
        """
        try:
            # Remove existing highlights
            self.remove_subplot_highlights()

            # Add new highlight
            xlim = axes.get_xlim()
            ylim = axes.get_ylim()

            highlight = axes.axvspan(xlim[0], xlim[1], alpha=alpha, color=color, zorder=1000)

            # Store reference to remove later
            if not hasattr(self, 'subplot_highlights'):
                self.subplot_highlights = []
            self.subplot_highlights.append(highlight)

            # Redraw canvas
            self.canvas.draw()

        except Exception as e:
            print(f"Error adding subplot highlight: {e}")

    def remove_subplot_highlights(self):
        """Remove all subplot highlights."""
        if hasattr(self, 'subplot_highlights'):
            for highlight in self.subplot_highlights:
                try:
                    highlight.remove()
                except:
                    pass
            self.subplot_highlights.clear()
            self.canvas.draw()

    # Initialize the preprocessing pipeline
    def set_preprocessing_pipeline(self, pipeline):
        """
        Set the preprocessing pipeline.
        
        Args:
            pipeline: EEGPreprocessingPipeline instance
        """
        self.preprocessing_pipeline = pipeline
        # Enable direct view buttons when pipeline is set
        self.plot_raw_button.config(state=tk.NORMAL)
        self.plot_ica_sources_button.config(state=tk.NORMAL)
        self.refresh_initial_plot()

    def get_artifact_range_values(self):
        """
        Get the start and stop values from the preprocessing frame.
        
        Returns:
            tuple: (start_time, stop_time) or (None, None) if not available
        """
        if not self.preprocessing_frame:
            return None, None
            
        try:
            start_time = self.preprocessing_frame.ica_start_var.get()
            stop_time = self.preprocessing_frame.ica_stop_var.get()
            
            # Convert to None if stop_time is 0 or empty (meaning use all data)
            if stop_time == 0:
                stop_time = None
                
            return start_time, stop_time
            
        except Exception as e:
            print(f"Error getting artifact range values: {e}")
            return None, None

    def on_artifact_range_toggle(self):
        """Handle changes to the artifact range checkbox."""
        # Refresh the plot when the checkbox state changes
        if self.current_plot_type == "psd":
            self.refresh_plot()

    # Plotting
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
        
        print(f"Creating plot of type: {plot_type}")  # Debug print

        try:
            # Get figure from preprocessing pipeline (with default size)
            pipeline_fig = self._get_pipeline_figure(plot_type, parameters)

            if pipeline_fig is not None:
                # Disconnect old event handlers
                self.disconnect_event_handlers()
                
                # Resize the figure to match current canvas size
                self._resize_figure_to_canvas(pipeline_fig)

                # Replace the figure reference and redraw
                self.canvas.figure = pipeline_fig
                self.fig = pipeline_fig  # Keep reference
                self.canvas.draw()
                
                # Reconnect event handlers for the new figure
                self.setup_event_handlers()
                
                # Ensure canvas has focus for event handling
                self.canvas.get_tk_widget().focus_set()
                
                print(f"Plot created successfully, type: {plot_type}")  # Debug print
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
                # Check if we should use artifact range for PSD
                if self.use_artifact_range_var.get():
                    start_time, stop_time = self.get_artifact_range_values()
                    if start_time is not None:
                        # Create a cropped version of the raw data for PSD calculation
                        return self.preprocessing_pipeline.plot_power_spectral_density(
                            picks=parameters.get('picks', None),
                            fmin=parameters.get('fmin', 0.5),
                            fmax=parameters.get('fmax', 70.0),
                            title=parameters.get('title', f'Power Spectral Density ({start_time}s - {stop_time if stop_time else "end"}s)'),
                            t_min=start_time,
                            t_max=stop_time
                        )
                
                # Default PSD plotting (full data)
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
        
    # Direct plotting methods (moved from app.py)
    def plot_raw_data_direct(self):
        """Plot raw data directly through pipeline."""
        if not self.preprocessing_pipeline:
            messagebox.showwarning("Warning", "No data loaded")
            return
        
        # Use checkbox value if remove_dc parameter is not explicitly provided
        remove_dc = self.remove_dc_var.get()
            
        try:
            self.preprocessing_pipeline.plot_eeg_data(
                duration=20.0,
                n_channels=20,
                start=0.0,
                block=True,
                remove_dc=remove_dc,
                title="Raw EEG Data"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot raw data: {str(e)}")
            
    def on_remove_dc_changed(self):
        """Handle changes to the Remove DC checkbox."""
        # This method will be called whenever the checkbox is toggled
        # You can add any immediate response here if needed
        pass

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

    def identify_clicked_subplot(self, event):
        """
        Identify which subplot was clicked in multi-subplot figures.
        
        Args:
            event: matplotlib MouseEvent
            
        Returns:
            tuple: (subplot_index, subplot_axes) or (None, None) if not found
        """
        if event.inaxes is None:
            return None, None

        all_axes = self.fig.get_axes()

        for i, ax in enumerate(all_axes):
            if ax == event.inaxes:
                return i, ax

        return None, None

    def set_event_handlers_enabled(self, enabled: bool):
        """
        Enable or disable event handlers.
        
        Args:
            enabled: True to enable event handlers, False to disable
        """
        self.event_handlers_enabled = enabled
        
        if enabled:
            self.setup_event_handlers()
        else:
            self.disconnect_event_handlers()