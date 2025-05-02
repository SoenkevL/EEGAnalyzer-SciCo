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
import sys
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from datetime import datetime
from multiprocesspandas import applyparallel

from eeganalyzer.core.eeg_processor import EEG_processor
from eeganalyzer.core.csv_processor import CSVProcessor
from eeganalyzer.utils.database import Alchemist


def add_or_update_dataset(session: Any, config: Dict[str, Any]) -> int:
    """
    Add or update a dataset in the database.

    Args:
        config (dict): Configuration dictionary containing dataset information.

    Returns:
        datset_id: The id of the DataSet object that was added or updated.
    """
    dataset = Alchemist.add_or_update_dataset(
        session,
        dataset_name=config['name'],
        dataset_path=config['bids_folder'],
        dataset_description=config['description']
    )
    return dataset.id


def add_or_update_eeg(session: Any, dataset_id: int, filepath: str) -> Any:
    """
    Add or update an eeg in the database.

    Args:
        session: Database session object
        dataset_id: ID of the dataset to associate with this EEG
        filepath: Path to the EEG file

    Returns:
        eeg_id: The id of the eeg object that was added or updated.
    """
    full_path = os.path.normpath(filepath)
    basename = os.path.basename(full_path)
    file_name, ext = os.path.splitext(basename)
    eeg = Alchemist.add_or_update_eeg_entry(
            session,
            dataset_id=dataset_id,
            filepath=full_path,
            filename=file_name,
            file_extension=ext,
        )
    return eeg


def add_or_update_experiment(session: Any, experiment: Dict[str, Any], run: Dict[str, Any]) -> Any:
    experiment = Alchemist.add_or_update_experiment(
            session,
            metric_set_name=experiment['name'],
            run_name=run['name'],
            fs=run['sfreq'],
            start=experiment['epoching']['start_time'],
            stop=experiment['epoching']['stop_time'],
            window_len=experiment['epoching']['duration'],
            window_overlap=experiment['epoching']['overlap'],
            lower_cutoff=run['filter']['l_freq'],
            upper_cutoff=run['filter']['h_freq'],
            montage=run['montage']
    )
    return experiment


def populate_data_tables(session: Any, experiment: Any, table_exists: str = 'append') -> Optional[str]:
    experiment_id = experiment.id
    table_name = None
    for eeg in experiment.eegs:
        eeg_id = eeg.id
        result_path = Alchemist.get_result_path_from_ids(session, experiment_id=experiment_id, eeg_id=eeg_id)
        if result_path:
            data = pd.read_csv(result_path)
            table_name = Alchemist.add_metric_data_table(session, experiment_id, eeg_id, data, table_exists)
    session.commit()
    return table_name


def get_files_dataframe(bids_folder: str, infile_ending: str, outfile_ending: str, folder_extensions: str,
                        session: Any, experiment: Any, dataset_id: int) -> pd.DataFrame:
    """
    Creates a DataFrame containing valid file paths, their corresponding output paths,
    and the processed status (whether the output file already exists).

    Args:
        bids_folder (str): Path to the BIDS folder containing the files to process.
        outfile_ending (str): The expected output file ending.
        folder_extensions (str): The folder extension to be appended to the output folder name.
        infile_ending (str): The expected input file ending.
        session: Database session object.
        experiment: Experiment object to associate with files.
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
                eeg = add_or_update_eeg(session, dataset_id, full_path)
                if not eeg in experiment.eegs:
                    experiment.eegs.append(eeg)
                Alchemist.add_result_path(session, experiment.id, eeg.id, outpath)
                # Append file data to list
                valid_files.append({'file_path': full_path, 'outpath': outpath, 'already_processed': already_processed})
                 # Add file to database

    # Create the DataFrame from the collected information
    df = pd.DataFrame(valid_files, columns=['file_path', 'outpath', 'already_processed'])

    return df


def process_file(row: pd.Series, metric_set_name: str, metric_path: str, annotations: List[str], lfreq: Optional[Union[int, float]],
                 hfreq: Optional[Union[int, float]], montage: str, ep_start: Optional[int], ep_stop: Optional[int], 
                 ep_dur: Optional[int], ep_overlap: int, sfreq: Union[int, float], recompute: bool) -> None:
    """
    Processes a single file.

    Args:
        row (pd.Series): A row from the DataFrame containing file information.
        metric_set_name (str): The name of the metric set to compute.
        annotations (list): The annotations of interest.
        lfreq (int or float): Lower cutoff frequency for filtering.
        hfreq (int or float): Upper cutoff frequency for filtering.
        montage (str): The montage to apply.
        ep_start (int): Epoching start time.
        ep_stop (int): Epoching stop time.
        ep_dur (int): Epoch duration.
        ep_overlap (int): Overlap of epochs.
        sfreq (int or float): Sampling frequency.
        recompute (bool): Whether to recompute metrics.
    """
    file_path = row['file_path']
    outpath = row['outpath']
    already_processed = row['already_processed']

    if not already_processed or recompute:
        print(f"Processing file: {file_path}")
        print(f"Output path: {outpath}")

        # Initialize EEG_processor and compute metrics
        if file_path.endswith(".fif") or file_path.endswith(".edf"):
            eeg_processor = EEG_processor(file_path)
            result = eeg_processor.compute_metrics(
                metric_set_name,
                metric_path,
                annotations,
                outpath,
                lfreq,
                hfreq,
                montage,
                ep_start,
                ep_stop,
                ep_dur,
                ep_overlap,
                sfreq,
                recompute,
            )
        elif file_path.endswith(".csv"):
            csv_processor = CSVProcessor(file_path, sfreq=sfreq)
            result = csv_processor.compute_metrics(
                metric_set_name,
                metric_path,
                outpath,
                lfreq,
                hfreq,
                ep_start,
                ep_stop,
                ep_dur,
                ep_overlap,
                sfreq,
                recompute,
            )
        else:
            result = 'Result not computed. Output file ending not recognized.'
        print(f"Result: {result}")
    else:
        print(f"Skipping already processed file: {file_path}")


def process_experiment(config: Dict[str, Any], log_file: Optional[str], num_processes: int = 4) -> None:
    """
    Processes experiments and their respective runs as specified in the YAML configuration.

    Args:
        config (dict): The dictionary representation of the YAML configuration file.
        log_file (str): The path to the log file where outputs and logs will be saved.
        num_processes (int): Number of processes to use for parallel processing.
    """
    # Redirect all print outputs to the log file
    if log_file:
        log_stream = open(log_file, 'w')
        sys.stdout = log_stream  # Redirect print statements to log file
    print(f'{"*" * 102}\n{"*" * 40} {datetime.today().strftime("%Y-%m-%d %H:%M:%S")} {"*" * 40}\n{"*" * 102}\n')
        # Iterate through experiments defined in the configuration
    for experiment in config['experiments']:
        # make sure we can access our sqlite base
        engine = Alchemist.initialize_tables(experiment['sqlite_path'])
        with Alchemist.make_session(engine) as session:
            # Extract experiment-level configuration
            exp_name = experiment['name']
            input_file_ending = experiment['input_file_ending']
            bids_folder = experiment['bids_folder']
            annotations = experiment['annotations_of_interest']
            outfile_ending = experiment['outfile_ending']
            recompute = experiment['recompute']
            epoching = experiment['epoching']
            ep_start, ep_dur, ep_stop, ep_overlap = (
                epoching['start_time'],
                epoching['duration'],
                epoching['stop_time'],
                epoching['overlap'],
            )
            metric_set_name = experiment['metric_set_name']
            metric_path = experiment['metric_path']

            # add or update dataset in sqlite database
            dataset_id = add_or_update_dataset(session, experiment)
            print(f"Using dataset ID: {dataset_id}")

            # Iterate through runs for each experiment
            for run in experiment['runs']:
                # Extract run-level configuration
                run_name = run['name']
                lfreq = run['filter']['l_freq']
                hfreq = run['filter']['h_freq']
                sfreq = run['sfreq']
                montage = run['montage']
                folder_extensions = run['metrics_prefix']

                print(
                    f'{"#" * 20} Running experiment "{exp_name}" and run "{run_name}" on folder "{bids_folder}" {"#" * 20}\n')

                experiment_object = add_or_update_experiment(session, experiment, run)
                # create first experiment, then files df and add experiment to each eeg
                # Create DataFrame of valid files to process (also adds the eegs to the database)
                files_df = get_files_dataframe(bids_folder, input_file_ending, outfile_ending, folder_extensions,
                                               session, experiment_object, dataset_id)
                print(f"Generated DataFrame with {len(files_df)} files:")
                # print(files_df.head())


                n_chunks = max(len(files_df) // num_processes, 1)
                num_processes = min(n_chunks, num_processes)
                files_df.apply_parallel(
                    process_file,
                    metric_set_name=metric_set_name,
                    metric_path=metric_path,
                    annotations=annotations,
                    lfreq=lfreq,
                    hfreq=hfreq,
                    montage=montage,
                    ep_start=ep_start,
                    ep_stop=ep_stop,
                    ep_dur=ep_dur,
                    ep_overlap=ep_overlap,
                    sfreq=sfreq,
                    recompute=recompute,
                    axis=0,
                    num_processes=num_processes,
                    n_chunks=n_chunks,
                )

                # Add the computed result frames to the database by iterating over the eegs of the experiment
                populate_data_tables(session, experiment_object)

    # Print a final message indicating completion
    print(f"\n{'*' * 50}")
    print(f"All processing complete. Results stored in database: {experiment['sqlite_path']}")
    print(f"{'*' * 50}\n")

    if log_file:
        log_stream.close()