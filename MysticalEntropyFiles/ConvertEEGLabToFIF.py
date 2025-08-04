# A file to use the .set, .fdt and .yaml metadata
# Additionally can combine the data back with the orignal channels
# to load a preprocessed eeg and convert it to properly annotated fif format
import os
import mne
import yaml
import logging

CLEANED_DATA_PATH = '/Users/soenkevanloh/PycharmProjects/MysticalEntropyData/Thesis_Cleaned_Pilot_1_2_3'
# I renamed some things compared to surfdrive
# 1. I removed the whitespace between Pilot and the number in the folders
# 2. I renamed files in Pilot2 based on the order in the yaml file
RAW_DATA_PATH = '/Users/soenkevanloh/PycharmProjects/MysticalEntropyData/Chakra_EEG_RawData'
# I uncommented the YRS vs SRS section in pilot3.yaml
METADATA_PATH = '/Users/soenkevanloh/PycharmProjects/MysticalEntropyData/metadata'

logger = logging.getLogger(__name__)

class Converter:

    def __init__(self, set_file, bdf_file=None, yaml_file=None):
        logger.info(f"Initializing Converter for {set_file}")
        self.set_file = set_file
        self.fdt_file = self.set_file.replace(".set", ".fdt")
        self.yaml_file = yaml_file if yaml_file else self.infer_yaml_filepath_from_set_file()

        try:
            self.config = self.load_yaml_config()
            self.raw_cleaned = self.load_raw_cleaned_data(preload=True)
            self.pilot, self.task_label = self.extract_task_label_from_filename()
            self.bdf_file = bdf_file if bdf_file else self.infer_bdf_filepath_from_set_file()
            self.raw_bdf = mne.io.read_raw_bdf(self.bdf_file, preload=True)
            logger.info(f"Successfully initialized converter for task: {self.task_label}")
        except Exception as e:
            logger.exception(f"Failed to initialize converter for {set_file}")
            raise

    def extract_task_label_from_filename(self):
        filename = self.set_file.split("/")[-1]
        split_filename = filename.split("_")
        task_label = '_'.join(split_filename[1:4])
        pilot = split_filename[0]
        logger.info(f"Extracted task label: {task_label}")
        return pilot, task_label

    def infer_bdf_filepath_from_set_file(self):
        basepath = RAW_DATA_PATH
        task = self.task_label
        pilot = self.pilot
        task = task.replace("_", " ")
        bdf_filename = f"{task}.bdf"
        bdf_path = os.path.join(basepath, pilot, bdf_filename)
        logger.info(f"Inferred bdf file path: {bdf_path}")
        return bdf_path



    def infer_yaml_filepath_from_set_file(self):
        '''
        function which can infer the metadata file based on set file if it is named correctly and a metadata folder exists
        where the rest of the files are stored
        '''
        basepath, filename = os.path.split(self.set_file)
        yaml_filename = filename.split('_')[0].lower()+'.yaml'
        yaml_path = os.path.join(METADATA_PATH, yaml_filename)
        logger.info(f"Inferred yaml file path: {yaml_path}")
        return yaml_path

    def load_yaml_config(self):
        try:
            with open(self.yaml_file, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded YAML config from {self.yaml_file}")
            return config
        except FileNotFoundError:
            logger.error(f"YAML config file not found: {self.yaml_file}")
            raise
        except yaml.YAMLError:
            logger.exception(f"Error parsing YAML file: {self.yaml_file}")
            raise

    def load_raw_cleaned_data(self, preload=False):
        raw = mne.io.read_raw_eeglab(self.set_file, preload=preload)
        info = raw.info
        logger.info(f"Loaded raw data with {info['nchan']} channels")
        return raw

    def annotate_aux_channels(self):
        #TODO: implement a check if channels exist
        self.raw_bdf.set_channel_types(self.config['channel_types'])
        logger.info("Annotated aux channels")

    def annotate_bad_channels(self):
        bad_channels = self.config['bad_channels'][self.task_label]
        self.raw_bdf.info['bads'] = bad_channels
        logger.info(f"Annotated bad channels: {bad_channels}")

    def add_timings_as_annotations(self):
        timings = self.config['timings'][self.task_label]
        onsets = []
        durations = []
        descriptions = []
        for name, times in timings.items():
            stop_time = times[1]
            start_time = times[0]
            descriptions.append(name)
            onsets.append(start_time)
            durations.append(stop_time - start_time)
        self.raw_cleaned.annotations.append(onsets, durations, descriptions)
        self.raw_bdf.annotations.append(onsets, durations, descriptions)
        logger.info(f"Added timings as annotations: {timings}")

    def add_aux_channels_to_cleaned_data(self):
        # This conversion follows assumptions:
        # 1. the start time of 0 is synched in both recordings and data was removed at the end
        cleaned_length = self.raw_cleaned.times[-1]
        raw_bdf_copy = self.raw_bdf.copy()
        raw_bdf_copy.pick_types(eeg=False, eog=True, ecg=True, misc=True, emg=True)
        raw_bdf_copy.resample(self.raw_cleaned.info['sfreq'])
        raw_bdf_copy.crop(tmin=0, tmax=cleaned_length)
        self.raw_cleaned.add_channels([raw_bdf_copy])

    def convert(self):
        self.annotate_aux_channels()
        # self.annotate_bad_channels()
        self.add_timings_as_annotations()
        self.add_aux_channels_to_cleaned_data()
        logger.info("Converted data")

    def save_converted_file(self, fif_file, overwrite=True):
        os.makedirs(os.path.dirname(fif_file), exist_ok=overwrite)
        self.raw_cleaned.save(fif_file, overwrite=overwrite)
        logger.info(f"Saved converted file to {fif_file}")

def main(plot=True):
    # Import logging config at the start
    from eeganalyzer.utils.LoggingConfiguration import setup_logging
    setup_logging(log_level=logging.INFO, log_file="logs/converter.log")

    logger.info("Starting EEG conversion process")

    processed_files = 0

    for root, dirs, files in os.walk(CLEANED_DATA_PATH):
        for file in files:
            if file.endswith(".set"):
                try:
                    set_file = os.path.join(root, file)
                    converter = Converter(set_file)
                    converter.convert()
                    fif_file = os.path.join(root, 'converted', file.split('.')[0] + '-raw.fif')
                    converter.save_converted_file(fif_file)
                    if plot:
                        converter.raw_bdf.plot()
                        converter.raw_cleaned.plot(block=True)
                    processed_files += 1
                    logger.info(f"Successfully processed {file}")
                except Exception as e:
                    logger.exception(f"Failed to process {file}")
                    continue

    logger.info(f"Conversion process completed. Processed {processed_files} files")


if __name__ == "__main__":
    # Configure logging
    main(plot=False)
    logging.shutdown() #only needed here as this is the only logged file right now