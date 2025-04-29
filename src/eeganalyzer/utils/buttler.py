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

Utility functions for EEG analysis.

This module provides utility functions for EEG analysis.
"""

import os
from typing import Tuple, Optional, Union, Dict, List, Any


class Buttler:
    """
    A class for utility functions.
    """
    
    def __init__(self) -> None:
        pass
    
    def check_outfile_name(self, outfile: str, file_exists_ok: bool = False) -> Tuple[bool, str]:
        """
        Checks if the outfile name is valid and if the directory exists.
        
        Args:
            outfile (str): Path to the output file.
            file_exists_ok (bool): If True, the function will return True even if the file already exists.
            
        Returns:
            tuple: (bool, str) - (True, '') if the outfile name is valid, (False, error_message) otherwise.
        """
        if not outfile:
            return False, 'No outfile name provided'
        
        # Create directory for outfile if it doesn't exist
        outfile_dir = os.path.dirname(outfile)
        if outfile_dir and not os.path.exists(outfile_dir):
            try:
                os.makedirs(outfile_dir, exist_ok=True)
            except Exception as e:
                return False, f'Could not create directory for outfile: {e}'
        
        # Check if file already exists
        if os.path.exists(outfile) and not file_exists_ok:
            return False, f'File {outfile} already exists and file_exists_ok is False'
        
        return True, ''
    
    def find_task_from_filename(self, filepath: Optional[str]) -> Optional[str]:
        """
        Extracts the task label from a filename.
        
        Args:
            filepath (str): Path to the file.
            
        Returns:
            str: Task label extracted from the filename, or None if no task label is found.
        """
        if not filepath:
            return None
        
        filename = os.path.basename(filepath)
        
        # Extract task label from filename
        # This is a simple implementation that assumes the task label is between 'task-' and '_'
        if 'task-' in filename:
            task_start = filename.find('task-') + 5
            task_end = filename.find('_', task_start)
            if task_end == -1:
                task_end = len(filename)
            return filename[task_start:task_end]
        
        return None
    
    def map_chaos_pipe_result_to_float(self, result: Any) -> Optional[float]:
        """
        Maps the result of the chaos pipeline to a float.
        
        Args:
            result: Result from the chaos pipeline.
            
        Returns:
            float: Mapped result.
        """
        if result is None:
            return None
        
        # If result is already a number, return it
        if isinstance(result, (int, float)):
            return float(result)
        
        # If result is a string, try to convert it to a float
        if isinstance(result, str):
            try:
                return float(result)
            except ValueError:
                pass
        
        # If result is a list or tuple, return the first element
        if isinstance(result, (list, tuple)) and len(result) > 0:
            return self.map_chaos_pipe_result_to_float(result[0])
        
        # If result is a dictionary, return the 'result' key if it exists
        if isinstance(result, dict) and 'result' in result:
            return self.map_chaos_pipe_result_to_float(result['result'])
        
        # If none of the above, return None
        return None