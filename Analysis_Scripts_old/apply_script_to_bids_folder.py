import argparse
import os
import sys
import yaml
from Compute_metrics_from_annotation import compute_metrics
from datetime import datetime

# This file is one of the two main files for computation. It is supposed to the executed from the command line with
# with arguments beeing the config file to use and the logfile where the log is written to.
# This file is used when computing files from a bids folder structure


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


if __name__ == "__main__":
    # main part of the file, will be exectued if file is called
    # initialize parsing of input arguments from command line
    parser = argparse.ArgumentParser(description='Takes as input a bids folder containing _raw.fif files'
                                                 ' and a name for a script')
    parser.add_argument('--yaml_config', type=str, help='the yaml file containing all relevant information')
    parser.add_argument('--logfile_path', type=str, help='path where to log to (should end in .log)')
    args = parser.parse_args()
    yaml_file = args.yaml_config
    log_file = args.logfile_path
    # initalize logging
    log_file = check_file_exists_and_create_path(log_file, True)
    # instead of priniting to the console we capture the outputs into a log
    with open(log_file, 'w') as sys.stdout:
        print(f'{"*"*102}\n{"*"*40} {datetime.today().strftime("%Y-%m-%d %H:%M:%S")} {"*"*40}\n{"*"*102}\n\n')
        # we load the config from the .yaml file
        config = load_yaml_file(yaml_file)
        # go through each experiment defined in the file
        for experiment in config['experiments']:
            # load the experiment wide options
            exp_name: str = experiment['name']
            bids_folder: str = experiment['bids_folder']
            annotations: list[str] = experiment['annotations_of_interest']
            outfile_ending: str = experiment['outfile_ending']
            recompute: bool = experiment['recompute']
            ep_start: int = experiment['epoching']['start_time']
            ep_dur: int = experiment['epoching']['duration']
            ep_stop: int = experiment['epoching']['stop_time']
            ep_overlap: int = experiment['epoching']['overlap']
            metric_set_name = experiment['metric_set_name']
            # we go through the runs for the experiment
            for run in experiment['runs']:
                # load parameters from the run configs
                run_name: str = run['name']
                lfreq: int = run['filter']['l_freq']
                hfreq: int = run['filter']['h_freq']
                sfreq: int = run['sfreq']
                montage: str = run['montage']
                folder_extensions = run['metrics_prefix']
                print(f'{"#"*20} Running experiment "{exp_name}" and run "{run_name}" on folder "{bids_folder}" {"#"*20}\n')
                # iterate over all files in the bids folder
                for base, dirs, files in os.walk(bids_folder):
                    for file in files:
                        # only raw.fif are of interest for the analysis, could be changed if needed
                        print(file)
                        if file.endswith("raw.fif"):
                            # instantiate files, a check that the outpath exists and is properly created is done in
                            # compute metrics itself
                            full_path = os.path.join(base, file)
                            # files need to be saved with the name 'preprocessed-raw.fif' for this to work
                            outpath = full_path.replace("preprocessed-raw.fif", outfile_ending)
                            outpath = outpath.replace("eeg", f"metrics{folder_extensions}")
                            print(outpath)
                            if outfile_ending.endswith("metrics.csv"):
                                print(f'{"-"*10} Calculating metrics for {file} {"-"*10}')
                                print(f'outputs will be saved to: \n{outpath}')
                                # main computational part where compute_metrics from Compute_metrics_from_annotation.py is called
                                print(compute_metrics(full_path, metric_set_name, annotations, outpath, lfreq, hfreq,
                                                      montage, ep_start, ep_stop, ep_dur, ep_overlap, sfreq, recompute))
                            else:
                                print('the given script_name is not valid, please see help for details')
                                sys.exit('the given script_name is not valid, please see help for details')
