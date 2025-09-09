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

Core processing functionality for EEG analysis.

This module provides the main processing functions for EEG analysis.
"""

import os
import shutil
from pprint import pformat
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from sqlalchemy.orm import Mapped
import logging
from eeganalyzer.utils.LoggingConfiguration import setup_logging
import time
from eeganalyzer.utils.config import load_yaml_file, check_file_exists_and_create_path

from eeganalyzer.core.eeg_processor import EEG_processor
from eeganalyzer.utils.database import Alchemist

class Processor:
    
    def __init__(self, config_path, log_file=None) -> None:
        # Initialize logging
        setup_logging(log_level=os.getenv('LOG_LEVEL', logging.INFO),
                      log_file=log_file)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Processor initialized")
        # initialize variables
        self.current_experiment = None
        self.session = None
        self.config_path = config_path
        self.log_file = log_file
        self.config = load_yaml_file(config_path)

    def get_files_dataframe(self) -> pd.DataFrame:
        """
        Creates a DataFrame containing valid file paths, their corresponding output paths,
        and the processed status (whether the output file already exists).

        Args:
            bids_folder (str): Path to the BIDS folder containing the files to process.
            outfile_ending (str): The expected output file ending.
            folder_extensions (str): The folder extension to be appended to the output folder name.
            infile_ending (str): The expected input file ending.
            session: Database session object.
            experiment_entry: Experiment object to associate with files.
            dataset_id (int): ID of the dataset to associate with files.

        Returns:
            pd.DataFrame: A DataFrame where:
                - The first column ('file_path') contains absolute file paths of valid files.
                - The second column ('outpath') contains the absolute path of the metrics output.
                - The third column ('already_processed') is a boolean indicating whether the output file exists.
        """
        bids_folder = self.current_experiment['bids_folder']
        infile_ending = self.current_experiment['input_file_ending']
        outfile_ending = self.current_experiment['outfile_ending']
        folder_extensions = '/'+'/'.join(
            [self.current_experiment.get('name').replace(' ', '_'),
             self.current_experiment.get('preprocessing_name').replace(' ', '_'),
             self.current_experiment.get('metric_name').replace(' ', '_')])
        valid_files = []

        # Walk through the BIDS folder structure
        for base, dirs, files in os.walk(bids_folder):
            splitbase = base.split('/')
            for file in files:
                if not infile_ending or file.endswith(infile_ending):
                    full_path = os.path.join(base, file)

                    # Construct the output path based on file naming conventions
                    outfile = file.replace(infile_ending, outfile_ending)
                    outpath = os.path.join(
                        *splitbase[:-1],
                        f'metrics{folder_extensions}',
                        outfile,
                    )
                    # Check if the output file exists
                    already_processed = os.path.exists(outpath)
                    valid_files.append({'file_path': full_path, 'outpath': outpath, 'already_processed': already_processed})

        # Create the DataFrame from the collected information
        df = pd.DataFrame(valid_files, columns=['file_path', 'outpath', 'already_processed'])
        base_output_path = os.path.join(
            *splitbase[:-1],
            f'metrics{folder_extensions}'
        )
        mapping_path = os.path.join(
            base_output_path,
            'mapping.csv',
        )
        df.to_csv(mapping_path, index=False)
        shutil.copy(self.config_path, base_output_path)
        logging.debug(f"Generated DataFrame with {len(df)} files")
        return df

    @staticmethod
    def process_file(row: pd.Series, experiment) -> None:
        """
        Processes a single file.

        Args:
            row (pd.Series): A row from the DataFrame containing file information.
        """
        t_start = time.time()
        processing_config = {'annotations': experiment['annotations_of_interest'],
                             'outpath': row['outpath'],
                             'start_time': experiment['epoching']['start_time'],
                             'stop_time': experiment['epoching']['stop_time'],
                             'duration': experiment['epoching']['duration'],
                             'overlap': experiment['epoching']['overlap'],
                             'recompute': experiment['recompute'],
                             'preprocessing_name': experiment['preprocessing_name'],
                             'preprocessing_params': experiment['preprocessing_params'],
                             'metric_name': experiment['metric_name'],
                             'metric_params': experiment['metric_params'],
                             }
        logging.debug(f"Processing config: \n{pformat(processing_config, indent=4, width=100, compact=True)}")
        file_path = row['file_path']
        outpath = row['outpath']
        already_processed = row['already_processed']

        if not already_processed or processing_config['recompute']:
            logging.info(f"Attempting to process file: {file_path}")
            logging.info(f"Results will be saved to: {outpath}")

            # Initialize EEG_processor and compute metrics
            if file_path.endswith(".fif") or file_path.endswith(".edf"):
                current_eeg_processor = EEG_processor(file_path, processing_config)
                current_eeg_processor.compute_metrics()
            else:
                logging.warning('Result not computed. Output file ending not recognized.')
        else:
            logging.info(f"Skipping already processed file: {file_path}")

    def process_experiment(self) -> None:
        """
        Processes experiments and their respective runs as specified in the YAML configuration.
        """

        # Iterate through experiments defined in the configuration
        for experiment in self.config['experiments']:
            self.current_experiment = experiment
            logging.info(
                f'{"#" * 20}'
                f' Running experiment "{experiment["name"]}"'
                f' on folder "{experiment["bids_folder"]}"'
                f' {"#" * 20}\n')
            files_df = self.get_files_dataframe()
            if len(files_df) == 0:
                logging.warning('No valid files found for processing.')
                # Dispose engine before continuing, to release resources/threads
                return None
            files_df.apply(self.process_file, experiment=experiment, axis=1)
        # Print a final message indicating completion
        logging.info(f"All processing complete.")
        return None
