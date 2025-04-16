#!/usr/bin/env python3
"""
Run script for the EEG Metrics Viewer.

This script provides a simple way to launch the EEG Metrics Viewer application
with a specified database path.

Usage:
    python run_metrics_viewer.py [path_to_database]

If no database path is provided, the script will look for the default database
at "../example/EEGAnalyzer.sqlite".
"""

import os
import sys
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = ["customtkinter", "matplotlib", "numpy", "pandas", "sqlalchemy", "mne"]
    missing_packages = []

    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)

    if missing_packages:
        print("The following required packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print("  python install_dependencies.py")
        print("or")
        print(f"  pip install {' '.join(missing_packages)}")
        return False

    return True

def main():
    """Run the EEG Metrics Viewer application."""
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    # Import App only after checking dependencies
    from metrics_viewer import App

    # Check if a database path is provided as a command-line argument
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Default database path
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "example", "EEGAnalyzer.sqlite")

    # Ensure the database file exists
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        print("Please provide a valid path to the EEGAnalyzer.sqlite database.")
        sys.exit(1)

    print(f"Starting EEG Metrics Viewer with database: {db_path}")

    # Start the application
    app = App(db_path)
    app.mainloop()

if __name__ == "__main__":
    main()
