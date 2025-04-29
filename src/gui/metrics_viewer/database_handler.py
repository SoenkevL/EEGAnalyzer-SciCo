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

Database handler for the EEG Metrics Viewer.

This module provides functionality for interacting with the SQLite database
containing EEG metrics data.
"""

from typing import List, Dict, Any
import pandas as pd

# Import Alchemist from eeganalyzer.utils.database instead of OOP_Analyzer
from eeganalyzer.utils.database import Alchemist, Experiment

from .utils import METADATA_COLUMNS


class DatabaseHandler:
    """
    Handles interactions with the SQLite database containing EEG metrics.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the database handler.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.engine = Alchemist.initialize_tables(db_path)
        self.session = Alchemist.make_session(self.engine)
        
    def __del__(self):
        """Close the session when the object is deleted."""
        if hasattr(self, 'session'):
            self.session.close()
    
    def get_experiments(self) -> List[Dict[str, Any]]:
        """
        Get all experiments from the database.
        
        Returns:
            List of dictionaries containing experiment information
        """
        experiments = Alchemist.find_entries(self.session, Experiment)
        return [{'id': exp.id, 'name': exp.metric_set_name, 'run_name': exp.run_name} for exp in experiments]
    
    def get_eegs_for_experiment(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        Get all EEGs associated with a specific experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            List of dictionaries containing EEG information
        """
        experiment = self.session.get(Experiment, experiment_id)
        if not experiment:
            return []
        
        return [{'id': eeg.id, 'filename': eeg.filename, 'filepath': eeg.filepath} for eeg in experiment.eegs]
    
    def get_metrics_data(self, experiment_id: str, eeg_id: str) -> pd.DataFrame:
        """
        Get metrics data for a specific experiment and EEG.
        
        Args:
            experiment_id: ID of the experiment
            eeg_id: ID of the EEG
            
        Returns:
            DataFrame containing the metrics data
        """
        table_name = f"data_experiment_{experiment_id}"
        
        try:
            # Query the data for the specific EEG
            query = f"SELECT * FROM {table_name} WHERE eeg_id = '{eeg_id}'"
            df = pd.read_sql_query(query, self.engine)
            return df
        except Exception as e:
            print(f"Error retrieving metrics data: {e}")
            return pd.DataFrame()
    
    def get_available_metrics(self, experiment_id: str, eeg_id: str) -> List[str]:
        """
        Get the unique metric names available for a specific experiment and EEG.
        
        Args:
            experiment_id: ID of the experiment
            eeg_id: ID of the EEG
            
        Returns:
            List of unique metric names
        """
        df = self.get_metrics_data(experiment_id, eeg_id)
        
        if 'metric' in df.columns:
            return df['metric'].unique().tolist()
        return []
    
    def get_available_channels(self, experiment_id: str, eeg_id: str) -> List[str]:
        """
        Get the channel names available for a specific experiment and EEG.
        
        Args:
            experiment_id: ID of the experiment
            eeg_id: ID of the EEG
            
        Returns:
            List of channel names sorted alphabetically
        """
        df = self.get_metrics_data(experiment_id, eeg_id)
        
        # Exclude metadata columns
        channel_cols = [col for col in df.columns if col not in METADATA_COLUMNS]
        
        # Sort channels alphabetically
        channel_cols.sort()
        
        return channel_cols