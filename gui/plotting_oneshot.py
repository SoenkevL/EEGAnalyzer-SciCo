# This was created in close cooperation with proxyai and claude 3.7
import customtkinter as ctk  # CustomTkinter is a modern-looking UI library built on top of Tkinter
import numpy as np  # NumPy for numerical operations and array handling
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Special canvas to embed matplotlib in Tkinter
from matplotlib.figure import Figure  # Figure class to create matplotlib figures

# Set the overall appearance of the CustomTkinter application
# "dark" mode gives a modern dark theme (alternatives: "light", "system")
ctk.set_appearance_mode("dark")

# Set the color theme for interactive elements like buttons and sliders
# "green" gives green accents (alternatives: "blue", "dark-blue", etc.)
ctk.set_default_color_theme("green")


class MatplotlibFrame(ctk.CTkFrame):
    """A custom frame that contains a matplotlib figure

    This class creates a specialized frame that hosts a matplotlib plot.
    It inherits from CTkFrame which is CustomTkinter's enhanced version of Tkinter's Frame.
    """

    def __init__(self, master, title="Figure", **kwargs):
        """Initialize the frame

        Args:
            master: The parent widget (usually the main window)
            title: The title to display above the plot
            **kwargs: Additional arguments to pass to the CTkFrame constructor
        """
        # Call the parent class constructor with the master widget and any additional arguments
        super().__init__(master, **kwargs)

        # Configure grid layout for this frame
        # This makes column 0 expandable (weight=1), so it will grow when the window resizes
        self.grid_columnconfigure(0, weight=1)
        # This makes row 1 (where the plot will be) expandable
        self.grid_rowconfigure(1, weight=1)

        # Add a title label at the top of the frame
        # fg_color sets the background color of the label
        # corner_radius rounds the corners of the label
        self.title_label = ctk.CTkLabel(self, text=title, fg_color="gray30", corner_radius=6)
        # Place the label in row 0, column 0
        # padx adds horizontal padding, pady adds vertical padding
        # sticky="ew" makes the label stretch horizontally (east-west)
        self.title_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")

        # Create a matplotlib figure object
        # figsize sets the size in inches, dpi sets the resolution
        self.figure = Figure(figsize=(5, 4), dpi=100)
        # Add a subplot to the figure (1x1 grid, first subplot)
        self.plot = self.figure.add_subplot(111)

        # Create a special canvas that can display matplotlib figures in Tkinter
        # This bridges the gap between matplotlib and Tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        # Get the actual Tkinter widget from the canvas
        self.canvas_widget = self.canvas.get_tk_widget()
        # Place the canvas widget in row 1, column 0
        # sticky="nsew" makes it expand in all directions (north-south-east-west)
        self.canvas_widget.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Create an initial plot with default data
        self.update_plot()

    def update_plot(self, data=None):
        """Update the plot with new data or use default data

        Args:
            data: Optional list/tuple containing [x_values, y_values]
                 If None, default data will be used
        """
        # Clear the current plot to prepare for new data
        self.plot.clear()

        if data is None:
            # If no data is provided, create some default data
            # Create an array of x values from 0 to 3 with small steps
            t = np.arange(0, 3, .01)
            # Create sine wave data
            y = 2 * np.sin(2 * np.pi * t)
            # Plot the data
            self.plot.plot(t, y)
        else:
            # If data is provided, use it for the plot
            # data[0] should be x values, data[1] should be y values
            self.plot.plot(data[0], data[1])

            # Set appropriate axis limits based on the data
            if len(data[0]) > 0:
                x_min, x_max = min(data[0]), max(data[0])
                y_min, y_max = min(data[1]), max(data[1])

                # Add some padding to the limits
                x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
                y_padding = (y_max - y_min) * 0.1

                self.plot.set_xlim(x_min - x_padding, x_max + x_padding)
                self.plot.set_ylim(y_min - y_padding, y_max + y_padding)

        # Set the title and axis labels for the plot
        self.plot.set_title("Sample Plot")
        self.plot.set_xlabel("X axis")
        self.plot.set_ylabel("Y axis")
        # Adjust the layout to make everything fit nicely
        self.figure.tight_layout()
        # Redraw the canvas to show the updated plot
        self.canvas.draw()


class App(ctk.CTk):
    """Main application class

    This class creates the main window and organizes the different frames.
    """

    def __init__(self):
        """Initialize the main application window"""
        super().__init__()

        # Configure window
        self.title("CustomTkinter with Matplotlib")
        self.geometry("900x700")  # Increased height to accommodate bottom controls

        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)  # Main column expands
        self.grid_rowconfigure(0, weight=1)  # Plot row expands
        self.grid_rowconfigure(1, weight=0)  # Control row has fixed height

        # Create plot frame (takes up the entire top row)
        self.plot_frame = MatplotlibFrame(self, title="Interactive Plot")
        self.plot_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Create bottom control frame
        self.control_frame = BottomControlFrame(self, self.plot_frame)
        self.control_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")


class BottomControlFrame(ctk.CTkFrame):
    """A frame containing controls for the plot, arranged horizontally"""

    def __init__(self, master, plot_frame, **kwargs):
        """Initialize the control frame

        Args:
            master: The parent widget
            plot_frame: Reference to the MatplotlibFrame to update
            **kwargs: Additional arguments for the CTkFrame constructor
        """
        # Call the parent class constructor
        super().__init__(master, **kwargs)
        # Store reference to the plot frame so we can update it
        self.plot_frame = plot_frame

        # Configure grid layout - create 4 columns for the sliders and button
        # Each control (slider or button) gets its own column
        for i in range(4):
            self.grid_columnconfigure(i, weight=1)

        # Generate initial data
        self.t = np.arange(0, 3, .01)
        self.data_length = len(self.t)

        # Create variables with dynamic limits
        self.frequency = ctk.DoubleVar(value=2.0)
        self.start_index = ctk.IntVar(value=0)
        self.end_index = ctk.IntVar(value=self.data_length)

        # Create frames for each control to hold slider and entry together
        self.freq_frame = ctk.CTkFrame(self)
        self.freq_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        self.freq_frame.grid_columnconfigure(0, weight=3)
        self.freq_frame.grid_columnconfigure(1, weight=1)

        self.start_frame = ctk.CTkFrame(self)
        self.start_frame.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="ew")
        self.start_frame.grid_columnconfigure(0, weight=3)
        self.start_frame.grid_columnconfigure(1, weight=1)

        self.end_frame = ctk.CTkFrame(self)
        self.end_frame.grid(row=0, column=2, padx=10, pady=(10, 0), sticky="ew")
        self.end_frame.grid_columnconfigure(0, weight=3)
        self.end_frame.grid_columnconfigure(1, weight=1)

        # Frequency controls
        self.freq_slider = ctk.CTkSlider(self.freq_frame, from_=0.5, to=5.0,
                                         variable=self.frequency,
                                         command=self.update_plot)
        self.freq_slider.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")

        self.freq_entry = ctk.CTkEntry(self.freq_frame, width=60,
                                       textvariable=self.frequency)
        self.freq_entry.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ew")
        self.freq_entry.bind("<Return>", self.validate_and_update)
        self.freq_entry.bind("<FocusOut>", self.validate_and_update)

        # Frequency label
        self.freq_label = ctk.CTkLabel(self, text="Frequency")
        self.freq_label.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="n")

        # Start index controls
        self.start_slider = ctk.CTkSlider(self.start_frame, from_=0, to=self.data_length-2,
                                          variable=self.start_index,
                                          command=self.update_range)
        self.start_slider.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")

        self.start_entry = ctk.CTkEntry(self.start_frame, width=60,
                                        textvariable=self.start_index)
        self.start_entry.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ew")
        self.start_entry.bind("<Return>", self.validate_and_update)
        self.start_entry.bind("<FocusOut>", self.validate_and_update)

        # Start index label
        self.start_label = ctk.CTkLabel(self, text="Start Index")
        self.start_label.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="n")

        # End index controls
        self.end_slider = ctk.CTkSlider(self.end_frame, from_=1, to=self.data_length,
                                        variable=self.end_index,
                                        command=self.update_range)
        self.end_slider.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")

        self.end_entry = ctk.CTkEntry(self.end_frame, width=60,
                                      textvariable=self.end_index)
        self.end_entry.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ew")
        self.end_entry.bind("<Return>", self.validate_and_update)
        self.end_entry.bind("<FocusOut>", self.validate_and_update)

        # End index label
        self.end_label = ctk.CTkLabel(self, text="End Index")
        self.end_label.grid(row=1, column=2, padx=10, pady=(0, 10), sticky="n")

        # Refresh button
        self.button = ctk.CTkButton(self, text="Refresh", command=self.update_plot)
        self.button.grid(row=0, column=3, rowspan=2, padx=20, pady=15, sticky="ns")

        # Initial plot update
        self.update_plot()

    def validate_and_update(self, event=None):
        """Validate entry inputs and update if valid"""
        try:
            # Validate frequency (must be between 0.5 and 5.0)
            freq_value = float(self.freq_entry.get())
            if freq_value < 0.5 or freq_value > 5.0:
                self.freq_entry.delete(0, 'end')
                self.freq_entry.insert(0, f"{self.frequency.get():.1f}")
            else:
                self.frequency.set(freq_value)

            # Validate start index (must be between 0 and data_length-2)
            start_value = int(self.start_entry.get())
            if start_value < 0 or start_value >= self.data_length-1:
                self.start_entry.delete(0, 'end')
                self.start_entry.insert(0, str(self.start_index.get()))
            else:
                self.start_index.set(start_value)

            # Validate end index (must be between 1 and data_length)
            end_value = int(self.end_entry.get())
            if end_value <= 0 or end_value > self.data_length:
                self.end_entry.delete(0, 'end')
                self.end_entry.insert(0, str(self.end_index.get()))
            else:
                self.end_index.set(end_value)

            # Ensure start < end
            if self.start_index.get() >= self.end_index.get():
                self.end_index.set(self.start_index.get() + 1)
                self.end_entry.delete(0, 'end')
                self.end_entry.insert(0, str(self.end_index.get()))

            # Update the plot
            self.update_plot()

        except ValueError:
            # Reset to current values if input is not a valid number
            self.freq_entry.delete(0, 'end')
            self.freq_entry.insert(0, f"{self.frequency.get():.1f}")

    def update_range(self, *args):
        """Update the display range and ensure start < end"""
        # Get current values
        start = self.start_index.get()
        end = self.end_index.get()

        # Ensure start is less than end
        if start >= end:
            # If start is too high, adjust it to be one less than end
            if args and args[0] == self.start_slider:
                self.start_index.set(end - 1)
            # If end is too low, adjust it to be one more than start
            else:
                self.end_index.set(start + 1)

        # Update the labels with current values
        self.freq_label.configure(text=f"Frequency: {self.frequency.get():.1f}")
        self.start_label.configure(text=f"Start: {self.start_index.get()}")
        self.end_label.configure(text=f"End: {self.end_index.get()}")

        # Update the plot with new range
        self.update_plot()

    def update_plot(self, *args):
        self.t = self.t if self.t.any() else np.arange(0, 3, .01)
        freq = self.frequency.get()
        y = 2 * np.sin(freq * np.pi * self.t)

        if len(self.t) != self.data_length:
            self.data_length = len(self.t)
            self.update_data_limits()

        start = self.start_index.get()
        end = self.end_index.get()

        start = max(0, min(start, len(self.t)-2))
        end = max(start+1, min(end, len(self.t)))

        self.freq_label.configure(text=f"Frequency: {freq:.1f}")
        self.start_label.configure(text=f"Start: {start}")
        self.end_label.configure(text=f"End: {end}")

        t_range = self.t[start:end]
        y_range = y[start:end]

        self.plot_frame.update_plot([t_range, y_range])


if __name__ == "__main__":
    app = App()
    app.mainloop()