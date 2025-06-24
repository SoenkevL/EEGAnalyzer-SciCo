import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np


# Add this new class for popup windows
class ICAPropertiesPopup:
    """Popup window to display ICA component properties."""

    def __init__(self, parent, figures, component_index):
        """
        Initialize ICA properties popup.
        
        Args:
            parent: Parent widget
            figures: List of matplotlib figures from plot_ica_properties
            component_index: Index of the clicked ICA component
        """
        self.parent = parent
        self.figures = figures if isinstance(figures, list) else [figures]
        self.component_index = component_index

        # Create popup window
        self.popup = tk.Toplevel(parent)
        self.popup.title(f"ICA Component {component_index} Properties")
        self.popup.geometry("800x600")

        # Make popup modal
        self.popup.transient(parent)
        self.popup.grab_set()

        self.setup_popup_ui()

    def setup_popup_ui(self):
        """Set up the popup window UI."""
        # Configure grid
        self.popup.grid_columnconfigure(0, weight=1)
        self.popup.grid_rowconfigure(1, weight=1)

        # Title frame
        title_frame = ttk.Frame(self.popup)
        title_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        title_label = ttk.Label(
            title_frame,
            text=f"ICA Component {self.component_index} Properties",
            font=("Arial", 14, "bold")
        )
        title_label.pack(side=tk.LEFT)

        # Close button
        close_button = ttk.Button(
            title_frame,
            text="Close",
            command=self.close_popup
        )
        close_button.pack(side=tk.RIGHT)

        # Create matplotlib frame for the first figure
        if self.figures:
            self.create_popup_plot_frame()
        else:
            # Show error message if no figures
            error_label = ttk.Label(
                self.popup,
                text="No ICA properties data available",
                font=("Arial", 12)
            )
            error_label.grid(row=1, column=0, padx=10, pady=10)

    def create_popup_plot_frame(self):
        """Create the matplotlib plot frame in the popup."""
        # Use the first figure from the list
        figure = self.figures[0]

        # Create canvas for the popup
        canvas = FigureCanvasTkAgg(figure, self.popup)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Create toolbar frame
        toolbar_frame = ttk.Frame(self.popup)
        toolbar_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        # Create toolbar
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

    def close_popup(self):
        """Close the popup window."""
        self.popup.destroy()