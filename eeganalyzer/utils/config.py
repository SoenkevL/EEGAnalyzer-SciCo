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

Configuration utilities for EEG analysis.

This module provides functions for loading and validating configuration files.
"""

import os
import yaml
from datetime import datetime


def check_file_exists_and_create_path(log_file: str, append_datetime: bool = False) -> (bool|str):
    """
    Ensures the path for a log file exists and optionally appends the current date and time to the filename.

    Args:
        log_file (str): Path of the log file, expected to end with `.log`.
        append_datetime (bool): If True, appends the current date and time to the log file name.

    Returns:
        str: The updated log file path if that was possible otherwise an emtpy string
    """
    # check if log_file could be a valid path, otherwise return False
    if not isinstance(log_file, (str ,os.PathLike)):
        return False

    # Create directory for log file if it doesn't exist
    if os.path.dirname(log_file) and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Append timestamp to log file name if required
    if append_datetime:
        log_file = log_file.rstrip('.log')  # Remove `.log` extension for modification
        log_file = f'{log_file}__{datetime.today().strftime("%Y_%m_%d_%H_%M_%S")}.log'

    return log_file


def load_yaml_file(yaml_filepath: str) -> dict:
    """
    Loads a YAML configuration file into a dictionary.

    Args:
        yaml_filepath (str): The path to the YAML file.

    Returns:
        dict: A dictionary representation of the YAML configuration.
    """
    with open(yaml_filepath, 'r') as stream:
        data = yaml.safe_load(stream)
    return data