# EEGAnalyzer

A comprehensive pipeline for analyzing timeseries data focused on chaoticity, complexity, fractality, and entropy in EEG data.

## Features

- Analyzes EEG data for:
  - Chaotic dynamics
  - Complexity assessment
  - Fractal properties
  - Entropy calculations
- Next to the allready provided metrics one can choose any kind of metric applicable to a 1-d or 2-d timeseries

- Modular and extensible architecture
- Configurable analysis parameters
- Logging system for tracking processing information
- Metrics output in standardized CSV format
- Visualization capabilities for key findings

## Requirements

- Python 3.8+
- Virtual environment (recommended)
- Dependencies listed in `requirements.txt`

## Installation with example
### 1. Clone the repository:
```
git clone https://github.com/SoenkevL/EEGAnalyzer.git 
cd EEGAnalyzer
```

### 2. Create and activate virtual environment
```
python -m venv .eeg_venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Prepare example data directory structure
```
mkdir -p examples/eeg
```

### 5. Move downloaded file to correct location
Now we need to download an example eeg to test the pipeline on. For this we use an open source edf file from kaggle
https://www.kaggle.com/datasets/abhishekinnvonix/seina-scalp-epilepsy-dataset?resource=download (14.10.2024)  
One can use all 5 files or just one of them, but we focus only on using one in this example, after downloading the file
move it to the right folder with the following command
```
mv PN00-1.edf examples/eeg/PN001-original.edf
```

### 6. Launch Jupyter notebook for preprocessing
Now we can run the notebook in the *eeg_reading_and_preprocessing* folder to do an inspection of the example eeg

### 7. Inspecting the configuration file
In order to process the files we use configuration files. An example of such a file can be found in *example/example_config_eeg.yaml*
Within the configuration file alot of parameters can be set which are important for the processing steps. This includes
things like: Data folder, metric set, epoching, annotation usage and more. All parameters are described within the example.

### 8. Run the main script with configuration#
In order to calculate the metrics on our eeg files we can run the example configuration using the following command
```
python3 OOP_Analyzer/apply_script_to_bids_folder_oop.py \
    --yaml_config example/example_config_eeg.yaml \
    --logfile_path example/test.log
```

