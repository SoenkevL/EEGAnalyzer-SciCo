# EEGAnalyzer

A comprehensive tool for analyzing EEG data with focus on customizable metric analysis. Originally designed to analyze the
chaotic dynamics, complexity assessment, fractal-properties, and entropy of EEG signals.  
The idea of this project is to bring together some of the most powerful python libraries for biomedical timeseries analysis. These
include mne for eeg processing, pandas to handle dataframes, neurokit2 and edge-of-py to provide the metrics for the analysis.

## Features

- **Analysis Capabilities**:
  - Custom python functions
  - Chaotic dynamics analysis
  - Complexity metrics evaluation
  - Fractal property detection
  - Entropy calculation methods

- **Modular Architecture**:
  - Preset metric sets to start analysis right away 
  - Extensible design for easily adding custom functions as metrics
  - Supports 1D and 2D timeseries data
  - Primarily written for EEG analysis while also providing functionality for csv data

- **Configuration**:
  - Adjustable parameters for analysis focused on timeseries.
  - Full folder (and subfolder) structures can be processed at once.
  - Ability to include only certain files based on name
  - Full control over the output directories of the computed metrics
  - Parameters for epoching like start, stop, duration and window overlap
  - Filtering
  - Integration with annotations of edf files to include only annotated data segments and exclude bad channels
  - Choosing predefined sets of metrics (python functions) to be applied to the data

- **Logging System**:
  - Tracks processing information for better debugging and monitoring

- **Output Format**:
  - Metrics saved in standardized CSV format
  - Additionally creates a relational database allowing easy comparison of metrics with each other and the original eeg files
  - Provides a simple GUI to visualize the metrics and compare them to the original eeg files

## Requirements
- Python 3.8+
- QT
  - on ubuntu see the following link: https://web.stanford.edu/dept/cs_edu/resources/qt/install-linux
  - on arch install using pacman: `sudo pacman -S qtcreator`
- Tk
  - on ubuntu install using apt: `sudo apt-get -y install tk`
  - on arch install using pacman: `sudo pacman -S tk`

## Installation

We recommend installing the package within a virtual environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### From PyPiTest
```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple eeganalyzer
```

### From source
```bash
git clone https://github.com/SoenkevL/EEGAnalyzer.git
cd EEGAnalyzer
```
```bash
pip install .
```

## Using the Command-line Interface
Once installed as a package, you can use the command-line interface:

To analyze eegs:
```bash
eeganalyzer --yaml_config <path_to_config_file> --logfile_path <path_to_log_file>
```

Arguments:
- `--yaml_config`: Path to the YAML configuration file (required)
- `--logfile_path`: Path to the log file (optional)

and to visualize the metrics and compare them to the original eeg files:
```bash
eegviwer --sql_path <path_to_sqlite_database>
```
Arguments:
- `--sql_path`: Path to the SQLite database (required) 
## Using the Python API

You can also use the package as a Python library:

```python
from src.eeganalyzer.core.processor import process_experiment
from src.eeganalyzer.utils.config import load_yaml_file

# Load configuration
config = load_yaml_file('config.yaml')

# Process experiments
process_experiment(config, 'results/analysis.log')
```

## Installation with example (Development)
### 1. Follow the installation instructions
Make sure you have installed the package as described above. If you decide you install the package via pip and not from
source please make sure you get the folder example from the github and add it to your current project.

### 2. Prepare example data directory structure

```
mkdir -p example/eeg
```

### 3. Move downloaded file to correct location

Now we need to download an example eeg to test the pipeline on.
Any eeg file in edf format could be used but some changes to the scripts may be necessary then.
For the example I recommed using the file PN00-1.edf from the following link.
https://www.kaggle.com/datasets/abhishekinnvonix/seina-scalp-epilepsy-dataset?resource=download (14.10.2024)  
The data comes from a platform called kaggle which provides open source data and code exchange for people interested
in the machine learning community. An account may be required to download the data but is completely without cost.
The downloaded file will be a zip, extract it to the root directory in which you currently are.
Additional edf files could be downloaded from the link. We focus only on using one in this example. \
After downloading the file move it to the right folder with the following command
```
mv PN00-1.edf example/eeg/PN001-original.edf
```

This will also rename the file so that it is properly handled by the config (touched upon in step 7)

### 4. Preprocessing the file

There are multiple possible approaches for Preprocessing the eeg in the folder *eeg_reading_and_preprocessing*.
1. Navigate to the folder and use the ipython notebook for an interactive experience
2. Use the *preprocess_and_inspect.py* file to inspect the data without relying on ipython notebooks
```
python3 eeg_reading_and_preprocessing/preprocess_and_inspect.py
```
3. Use the *preprocess_fast.py* to do the bare minimum preprocessing to execute the last step without any visual or textual outputs
```
python3 eeg_reading_and_preprocessing/preprocess_fast.py
```

### 5. Inspecting the configuration file

In order to process the files we use configuration files. An example of such a file can be found on github in
*example/example_config_eeg.yaml* \
Within the configuration file alot of parameters can be set which are important for the processing steps. This includes
things like: Data folder, metric set, epoching, annotation usage and more. All parameters are described within the
example config file.

### 6. Run the main script with configuration

In order to calculate the metrics on our eeg files we can run the example configuration using the following command.    
This will create the metrics and a log file, alternatively the command can be run without the logfile argument to see the output in the terminal.  
Make sure you are inside the project root with the virtual environment activated and the requirements installed.

```
    eeganalyzer \
    --yaml_config example/example_config_eeg.yaml \
    --logfile_path example/test.log
```

### 7. Analysing the produced metrics

Now that we created two csv files and saved our data to our database we can go ahead and do some analysis on it.
This can either be done in the GUI or in python.
For the gui simply run
```bash
eegviewer example/EEGAnalyzer.sqlite
```
In case you have problems with QT at this point and did not do preprocessing, please see the section above for possible fixes.

Alternatively you can use python to do the analysis directly on the database or the computed csv files.
```
python3 example/metric_analysis_example.py
```
### 8. Summary

The provided example is intended as guidance to this pipeline. It cant show all the things that it can do and now is the point
where you need to start experimenting with it yourself.

To end on a personal note: In data analysis it is hard to find a one fits all approach. On the one hand the analysis of your data
should be a carefully pre-planed step of the final analysis. On the other hand it can be very exciting and interesting to
explore Data, its sampling parameters and different metrics to try understanding a part of the world from it.  
Data, and especially the way it is displayed, is very much subject to the experimenter and analyst.
Therefore, I want to remind anyone that you should not make your data fit your thesis but test your
thesis with your data by closely thinking about what you analyze and why. 
