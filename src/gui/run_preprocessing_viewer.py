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

EEG Preprocessing GUI Entry Point.

This module provides the main entry point for the EEG preprocessing GUI application.
"""

import sys
import os
import tkinter as tk

# Add the parent directory to the path to import the preprocessing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gui.preprocessing_viewer.app import PreprocessingApp


def main():
    """Main entry point for the preprocessing GUI."""
    root = tk.Tk()
    app = PreprocessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()