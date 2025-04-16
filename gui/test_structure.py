#!/usr/bin/env python3
"""
Test script to verify the package structure without running the full application.
This script doesn't require the external dependencies to be installed.
"""

import os
import sys

def main():
    """Test the package structure."""
    print("Testing package structure...")
    
    # Check if the package directory exists
    if not os.path.isdir("metrics_viewer"):
        print("Error: metrics_viewer directory not found")
        return False
    
    # Check if all required files exist
    required_files = [
        "metrics_viewer/__init__.py",
        "metrics_viewer/app.py",
        "metrics_viewer/database_handler.py",
        "metrics_viewer/plot_frame.py",
        "metrics_viewer/selection_frame.py",
        "metrics_viewer/utils.py"
    ]
    
    for file_path in required_files:
        if not os.path.isfile(file_path):
            print(f"Error: {file_path} not found")
            return False
        else:
            print(f"Found: {file_path}")
    
    print("\nPackage structure looks good!")
    return True

if __name__ == "__main__":
    main()
