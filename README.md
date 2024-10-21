# EEGAnalyzer
Summarized code of my master thesis EEG processing pipeline. Focused on chaoticity, complexity, fractality and entropy in EEG data.


## How to run an experiment (example)
- download edf file from kaggle: https://www.kaggle.com/datasets/abhishekinnvonix/seina-scalp-epilepsy-dataset?resource=download (14.10.2024)
- paste edf into example folder into a subfolder called eeg
- rename file to *PN001-original.edf*
- you should end up with something like *examples/eeg/PN00-1-original.edf*
- run the *Inspect_and_preprocess_eeg.ipynb* notebook to get a feeling for the eeg and do some inspection
- Now lets see what is in the config file to check the analysis we would like to do
- to do some analysis run *python3 Analysis_Scripts/apply_script_to_bids_folder.py --yaml_config example/example_config_fromFolder.yaml --logfile_path example/test.log* in the root project folder
