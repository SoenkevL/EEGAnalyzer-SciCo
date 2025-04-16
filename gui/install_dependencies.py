#!/usr/bin/env python3
"""
Script to install the required dependencies for the EEG Metrics Viewer.

This script checks if the required packages are installed and installs them if needed.
"""

import sys
import subprocess
import importlib.util

def check_package(package_name):
    """Check if a package is installed."""
    return importlib.util.find_spec(package_name) is not None

def install_package(package_name):
    """Install a package using pip."""
    print(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    print(f"{package_name} installed successfully.")

def main():
    """Check and install required packages."""
    required_packages = [
        "customtkinter",
        "matplotlib",
        "numpy",
        "pandas",
        "sqlalchemy",
        "mne"
    ]

    missing_packages = []

    # Check which packages are missing
    for package in required_packages:
        if not check_package(package):
            missing_packages.append(package)

    # Install missing packages
    if missing_packages:
        print(f"The following packages need to be installed: {', '.join(missing_packages)}")
        try:
            for package in missing_packages:
                install_package(package)
            print("All required packages have been installed.")
        except Exception as e:
            print(f"Error installing packages: {e}")
            print("Please install the required packages manually:")
            for package in missing_packages:
                print(f"  pip install {package}")
            sys.exit(1)
    else:
        print("All required packages are already installed.")

if __name__ == "__main__":
    main()
