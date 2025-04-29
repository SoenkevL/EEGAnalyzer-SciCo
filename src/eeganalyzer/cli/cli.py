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

Command-line interface for EEG analysis.

This module provides the main entry point for the EEG analysis command-line interface.
"""

import argparse
import sys
from typing import Dict, Any, Union

from eeganalyzer.core.processor import process_experiment
from eeganalyzer.utils.config import load_yaml_file, check_file_exists_and_create_path


def main() -> int:
    """
    Main entry point for the EEG analysis command-line interface.
    
    This function parses command-line arguments and runs the EEG analysis pipeline.
    
    Returns:
        int: Exit code (0 for success)
    """
    # Parse input arguments from the command line
    parser = argparse.ArgumentParser(
        description='Processes files from a BIDS folder structure based on a YAML configuration file.'
    )
    parser.add_argument('--yaml_config', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--logfile_path', type=str, required=False, default=False, help='Path to the log file (must end with .log).')

    args = parser.parse_args()
    yaml_file: str = args.yaml_config
    log_file: Union[str, bool] = args.logfile_path

    # Ensure the log file path exists and append a timestamp
    log_file = check_file_exists_and_create_path(log_file, append_datetime=True)

    # Load configuration from the YAML file
    config: Dict[str, Any] = load_yaml_file(yaml_file)

    # Process the experiments as defined in the configuration
    process_experiment(config, log_file)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())