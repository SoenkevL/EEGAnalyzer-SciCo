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
from pprint import pprint
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Mapped
import logging
from eeganalyzer.utils.LoggingConfiguration import setup_logging

from eeganalyzer.core.eeg_processor import EEG_processor
from eeganalyzer.utils.database import Alchemist
from custom_files import metrics, pipeline_preprocessing

class Processor:
    
    def __init__(self, config, log_file=None, max_processors_used=1) -> None:
        # Initialize logging
        #TODO: change this up to use .env file
        setup_logging(log_level=config.get('log_level', logging.INFO),
                      log_file=log_file)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Processor initialized")
        # initialize variables
        self.session = None
        self.config = config
        self.log_file = log_file
        self.num_processes = max_processors_used
        self.current_eeg_processor = None
        self.current_csv_processor = None
        self.current_experiment = None
        self.current_dataset_id = None
        self.current_run = None
        self.current_experiment_entry = None
        
    def add_or_update_dataset(self) -> Mapped[str]:
        """
        Add or update a dataset in the database.

        Args:
            config (dict): Configuration dictionary containing dataset information.

        Returns:
            datset_id: The id of the DataSet object that was added or updated.
        """
        dataset = Alchemist.add_or_update_dataset(
            self.session,
            dataset_name=self.current_experiment['name'],
            dataset_path=self.current_experiment['bids_folder'],
            dataset_description=self.current_experiment['description']
        )
        logging.debug(f"Added or updated dataset: {dataset.id}")
        return dataset.id

    def add_or_update_eeg(self, filepath, dataset_id: int=None) -> Any:
        """
        Add or update an eeg in the database.

        Args:
            session: Database session object
            dataset_id: ID of the dataset to associate with this EEG
            filepath: Path to the EEG file

        Returns:
            eeg_id: The id of the eeg object that was added or updated.
        """
        dataset_id = dataset_id if dataset_id else self.current_dataset_id
        full_path = os.path.normpath(filepath)
        basename = os.path.basename(full_path)
        file_name, ext = os.path.splitext(basename)
        eeg = Alchemist.add_or_update_eeg_entry(
                self.session,
                dataset_id=dataset_id,
                filepath=full_path,
                filename=file_name,
                file_extension=ext,
            )
        logging.debug(f"Added or updated eeg: {eeg.id}")
        return eeg

    def add_or_update_experiment(self, experiment: Dict[str, Any]=None, run: Dict[str, Any]=None) -> Any:
        experiment = experiment if experiment else self.current_experiment
        experiment_entry = Alchemist.add_or_update_experiment(
                self.session,
                metric_set_name=experiment.get('metric_name', ''),
                run_name=experiment['name'],
                fs=pipeline_preprocessing.SFREQ,
                start=experiment['epoching']['start_time'],
                stop=experiment['epoching']['stop_time'],
                window_len=experiment['epoching']['duration'],
                window_overlap=experiment['epoching']['overlap'],
                lower_cutoff=pipeline_preprocessing.L_FREQ_CUTOFF,
                upper_cutoff=pipeline_preprocessing.H_FREQ_CUTOFF,
                montage=pipeline_preprocessing.REFERENCE,
        )
        logging.debug(f"Added or updated experiment: {experiment_entry.id}")
        return experiment_entry

    def populate_data_tables(self, experiment_entry: Any=None, table_exists: str = 'append') -> Optional[str]:
        experiment_entry = experiment_entry if experiment_entry else self.current_experiment_entry
        experiment_id = experiment_entry.id
        table_name = None
        for eeg in experiment_entry.eegs:
            eeg_id = eeg.id
            result_path = Alchemist.get_result_path_from_ids(self.session, experiment_id=experiment_id, eeg_id=eeg_id)
            if result_path:
                data = pd.read_csv(result_path)
                table_name = Alchemist.add_metric_data_table(self.session, experiment_id, eeg_id, data, table_exists)
        self.session.commit()
        logging.debug(f"populated data table: {table_name}")
        return table_name

    def get_files_dataframe(self) -> pd.DataFrame:
        bids_folder = self.current_experiment['bids_folder']
        infile_ending = self.current_experiment['input_file_ending']
        outfile_ending = self.current_experiment['outfile_ending']
        folder_extensions = '/'+'/'.join(
            [self.current_experiment.get('preprocessing_name'), self.current_experiment.get('metric_name')])
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
        valid_files = []

        # Walk through the BIDS folder structure
        for base, dirs, files in os.walk(bids_folder):
            for file in files:
                if not infile_ending or file.endswith(infile_ending):
                    full_path = os.path.join(base, file)

                    # Construct the output path based on file naming conventions
                    outfile = file.replace(infile_ending, outfile_ending)
                    splitbase = base.split('/')
                    outpath = os.path.join(
                        *splitbase[:-1],
                        f'metrics{folder_extensions}',
                        outfile,
                    )
                    # Check if the output file exists
                    already_processed = os.path.exists(outpath)

                    # Add eeg to experiment in database
                    eeg = self.add_or_update_eeg(full_path)
                    #TODO: cannot have sql object in self due to paralel processing I think
                    if not eeg in self.current_experiment_entry.eegs:
                        self.current_experiment_entry.eegs.append(eeg)
                    Alchemist.add_result_path(self.session, self.current_experiment_entry.id, eeg.id, outpath)
                    # Append file data to list
                    valid_files.append({'file_path': full_path, 'outpath': outpath, 'already_processed': already_processed})

        # Create the DataFrame from the collected information
        df = pd.DataFrame(valid_files, columns=['file_path', 'outpath', 'already_processed'])
        logging.debug(f"Generated DataFrame with {len(df)} files")
        return df

    @staticmethod
    def process_file(row: pd.Series, experiment, run) -> None:
        """
        Processes a single file.

        Args:
            row (pd.Series): A row from the DataFrame containing file information.
        """
        processing_config = {'annotations': experiment['annotations_of_interest'],
                             'outpath': row['outpath'],
                             'start_time': experiment['epoching']['start_time'],
                             'stop_time': experiment['epoching']['stop_time'],
                             'duration': experiment['epoching']['duration'],
                             'overlap': experiment['epoching']['overlap'],
                             'recompute': experiment['recompute'],
                             'metric_name': experiment['metric_name'],
                             'preprocessing_name': experiment['preprocessing_name'],
                             }
        logging.debug(f"Processing config: {pprint(processing_config, indent=4, width=100, compact=True)}")
        file_path = row['file_path']
        outpath = row['outpath']
        already_processed = row['already_processed']

        if not already_processed or processing_config['recompute']:
            logging.info(f"Processing file: {file_path}")
            logging.info(f"Output path: {outpath}")

            # Initialize EEG_processor and compute metrics
            if file_path.endswith(".fif") or file_path.endswith(".edf"):
                current_eeg_processor = EEG_processor(file_path, processing_config)
                result = current_eeg_processor.compute_metrics()
            else:
                result = 'Result not computed. Output file ending not recognized.'
            logging.info(f"Result: {result}")
        else:
            logging.info(f"Skipping already processed file: {file_path}")

    def process_experiment(self) -> None:
        """
        Processes experiments and their respective runs as specified in the YAML configuration.
        """

        # Redirect all print outputs to the log file
        logging.info(f'{"*" * 102}\n{"*" * 40} {datetime.today().strftime("%Y-%m-%d %H:%M:%S")} {"*" * 40}\n{"*" * 102}\n')
            # Iterate through experiments defined in the configuration
        for experiment in self.config['experiments']:
            # make sure we can access our sqlite base
            self.current_experiment = experiment
            engine = Alchemist.initialize_tables(experiment['sqlite_path'])
            with Alchemist.make_session(engine) as session:
                self.session = session
                # Extract experiment-level configuration
                # add or update dataset in sqlite database
                self.current_dataset_id = self.add_or_update_dataset()
                logging.info(f"Using dataset ID: {self.current_dataset_id}")

                logging.info(
                    f'{"#" * 20}'
                    f' Running experiment "{self.current_experiment["name"]}"'
                    f' on folder "{self.current_experiment["bids_folder"]}"'
                    f' {"#" * 20}\n')

                self.current_experiment_entry = self.add_or_update_experiment()
                # create first experiment, then files df and add experiment to each eeg
                # Create DataFrame of valid files to process (also adds the eegs to the database)
                files_df = self.get_files_dataframe()

                if len(files_df) == 0:
                    logging.warning('No valid files found for processing.')
                    return None
                files_df.apply(self.process_file, experiment=self.current_experiment, run=self.current_run,
                                        axis=1)
                self.populate_data_tables(self.current_experiment_entry)

        # Print a final message indicating completion
        logging.info(f"All processing complete. Results stored in database: {self.current_experiment['sqlite_path']}")
        return None
