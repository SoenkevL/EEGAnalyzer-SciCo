import argparse
import os
import sys

import pandas as pd
import yaml
from Compute_metrics_from_annotation import compute_metrics
from datetime import datetime
from icecream import ic
from multiprocesspandas import applyparallel

# This file is one of the two main files for computation. It is supposed to the executed from the command line with
# with arguments beeing the config file to use and the logfile where the log is written to.
# This file is used when computing files from a csv files, this is pretty specific to the csv file used for the coma
# patients on the synology server


def check_file_exists_and_create_path(log_file, append_datetime=False):
    '''
     utility function to make sure path or logfile exists and can append the current date and time for easy reference
     input:
     -log_file: path of the logfile, should end in .log
     -append_datetime: boolean if date and time should be incorportated in filename
     returns:
     -new logfile path
     '''
    if os.path.dirname(log_file) and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if append_datetime:
        log_file = log_file.split('.log')[0]
        log_file = f'{log_file}__{datetime.today().strftime("%Y_%m_%d_%H_%M_%S")}.log'
    return log_file


def load_yaml_file(yaml_filepath: str) -> dict:
    '''
    loads a yaml file into a dictionary
    input:
    -yaml_filepath: filepath of the yaml file
    returns:
    -data: dictionary from yaml file
    '''
    with open(yaml_filepath, 'r') as stream:
        data = yaml.safe_load(stream)
    return data

def apply_to_datarow(current_row, mapping, metric_set_name, annotations, lfreq, hfreq,
                     montage, ep_start, ep_stop, ep_dur, sfreq, recompute):
    '''
    function which takes in a datarow from a pandas dataframe. Specifically designed for the csv of coma patients from
    the CNPH group at the uni twente.
    input:
    - current_row: row of the dataframe
    - mapping: a mapping file to find the edf file of the EEG, also specific to the Synology server and csv file
    - all other arguments are just passed through to compute_metrics as experiment parameters
                see config file for explanation of the parameters
    return:
    - no return
    '''
    try:
        snummer = current_row['Studienummer']
        cpc = current_row['CPC (6mnd)']
        t_after_ca_start = current_row['Starttijd cEEG (uren na CA)']
        t_after_ca_stop = current_row['Stoptijd cEEG (uren na CA)']
        datapath = None
        center = None
        for key, value in mapping.items():
            if key in snummer:
                datapath = value
                center = key
                break
        if datapath:
            sub = '-'.join([center, snummer.split(center)[-1]])
            print(f'sub: {sub}')
            datapath = f'{Synology_root}/{datapath}/{snummer}'
            for root, dirs, files in os.walk(datapath):
                for file in files:
                    if '.edf' in file and '012_' in file: # the 012_ is to assure only taca 12 subjects are included
                        filepath = os.path.join(root, file)
                        time_after_ca = int(file.split('_')[0])
                        if t_after_ca_start <= time_after_ca <= t_after_ca_stop:
                            outpath = os.path.normpath(f'{result_path}/{sub}/metrics{folder_extensions}/{sub}_task-'
                                                       f'taca{time_after_ca}-cpc{cpc}_{outfile_ending}')
                            if outfile_ending.endswith("metrics.csv"):
                                print(f'{"-"*10} Calculating metrics for {file} {"-"*10}')
                                print(f'outputs will be saved to: \n{outpath}')
                                print(compute_metrics(filepath, metric_set_name, annotations, outpath, lfreq, hfreq,
                                                      montage, ep_start, ep_stop, ep_dur, sfreq, recompute))
                            else:
                                print('the given script_name is not valid, please see help for details')
                                sys.exit('the given script_name is not valid, please see help for details')
                        else:
                            print('the extracted time was not valid as it is outside the interval'
                                  ' specified in the csv')
    except TypeError:
        print(f'\n{"!"*10}Something went wrong when computing row {current_row} of the dataframe{"!"*10}\n')


if __name__ == "__main__":
    # initialize argument parsing
    parser = argparse.ArgumentParser(description='Takes as input a bids folder containing _raw.fif files'
                                                 ' and a name for a script')
    parser.add_argument('--yaml_config', type=str, help='the yaml file containing all relevant information')
    parser.add_argument('--logfile_path', type=str, help='path where to log to (should end in .log)')
    args = parser.parse_args()
    yaml_file = args.yaml_config
    log_file = args.logfile_path
    # initalize logging
    log_file = check_file_exists_and_create_path(log_file, True)
    with open(log_file, 'w') as sys.stdout:
        print(f'{"*"*102}\n{"*"*40} {datetime.today().strftime("%Y-%m-%d %H:%M:%S")} {"*"*40}\n{"*"*102}\n\n')
        config = load_yaml_file(yaml_file)
        # load general info from the config
        # go through each experiment defined in the file
        for experiment in config['experiments']:
            # load the experiment wide options
            exp_name: str = experiment['name']
            csv: str = experiment['csv']
            mapping_file: str = experiment['mapping_file']
            Synology_root: str = experiment['Synology_root']
            result_path: str = experiment['result_path']
            annotations: list[str] = experiment['annotations_of_interest']
            outfile_ending: str = experiment['outfile_ending']
            recompute: bool = experiment['recompute']
            ep_start: int = experiment['epoching']['start_time']
            ep_dur: int = experiment['epoching']['duration']
            ep_stop: int = experiment['epoching']['stop_time']
            metric_set_name = experiment['metric_set_name']
            num_processes: int = experiment['numProcesses']
            # go through the runs for the experiment
            for run in experiment['runs']:
                # load parameters from the run configs
                run_name: str = run['name']
                lfreq: int = run['filter']['l_freq']
                hfreq: int = run['filter']['h_freq']
                sfreq: int = run['sfreq']
                montage: str = run['montage']
                folder_extensions = run['metrics_prefix']
                # iterate over all files in the bids folder
                print(f'{"#"*20} Running experiment "{exp_name}" and run "{run_name}" {"#"*20}\n')
                patientdf = pd.read_csv(csv)
                mapping = load_yaml_file(mapping_file)
                # go through the rows of the dataframe on multiple cores and compute the metrics
                patientdf.apply_parallel(apply_to_datarow, mapping=mapping, metric_set_name=metric_set_name,
                                         annotations=annotations, lfreq=lfreq, hfreq=hfreq,
                                         montage=montage, ep_start=ep_start, ep_stop=ep_stop, ep_dur=ep_dur,
                                         sfreq=sfreq, recompute=recompute, axis=0, num_processes=num_processes,
                                         n_chunks=int(len(patientdf)/3))
