# EEGAnalyzer
Summarized code of my master thesis EEG processing pipeline. Focused on chaoticity, complexity, fractality and entropy in EEG data.


## How to run an experiment (example)
- I recommend creating and activating a virtual environment before continuing with the next steps!
- install the required packages using *pip install -r requirements.txt*
- download edf file from kaggle: https://www.kaggle.com/datasets/abhishekinnvonix/seina-scalp-epilepsy-dataset?resource=download (14.10.2024)
- paste edf into example folder into a subfolder called eeg
- rename file to *PN001-original.edf*
- you should end up with something like *examples/eeg/PN001-original.edf*
- run the *eeg_reading_and_preprocessing/Inspect_and_preprocess_eeg.ipynb* notebook to get a feeling for the eeg and do some inspection
- Analysis is run using config files to set the parameters of the analysis, an example for the downloaded eeg is provided under *example/example_config_eeg.yaml*
- to do some analysis run *python3 OOP_Analyzer/apply_script_to_bids_folder_oop.py --yaml_config example/example_config_eeg.yaml --logfile_path example/test.log* in the root project folder
- Once the script ran you should have a folder called *metrics* in your example folder which contains the csv with the processed metrics
- Next to the metrics there is also a logfile containing information on the processing and parameters used in the run