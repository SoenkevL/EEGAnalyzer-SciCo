from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gui.metrics_viewer.database_handler import DatabaseHandler


class Plotter():
    def __init__(self, dbpath):
        # db init
        self.dbhandler = DatabaseHandler(dbpath)
        self.available_experiments = self.dbhandler.get_experiments()
        # experiment init
        self.experiment_id = None
        self.available_eegs = None
        # data init
        self.eeg_id = None
        self.data = None
        self.channels = None
        self.metrics = None

    def load_data(self, experiment_id=None, eeg_id=None):
        experiment_id = experiment_id if experiment_id else self.experiment_id
        eeg_id = eeg_id if eeg_id else self.eeg_id
        df = self.dbhandler.get_metrics_data(experiment_id, eeg_id)
        self.data = df
        return df

    def set_current_experiment(self, experiment_idx=None, experiment_id=None, experiment_name=None, run_name=None):
        """
        provide either the index in the experiment list, the experiment id or the combination of experiment name and run name
        """
        if experiment_id:
            self.experiment_id = experiment_id
            return experiment_id
        elif isinstance(experiment_idx, int):
            self.experiment_id = self.available_experiments[experiment_idx].get('id', None)
            return self.experiment_id
        elif isinstance(experiment_name, str) and isinstance(run_name, str):
            for experiment in self.available_experiments:
                if experiment.get('name', None) == experiment_name and experiment.get('runName', None) == run_name:
                    self.experiment_id = experiment.get('id', None)
                    return self.experiment_id
        return None

    def set_available_eegs(self, experiment_id=None):
        experiment_id = experiment_id if experiment_id else self.experiment_id
        self.available_eegs = self.dbhandler.get_eegs_for_experiment(experiment_id)
        return self.available_eegs

    def set_current_eeg(self, eeg_idx=None, eeg_id=None):
        """
        provide either the index in the experiment list, the experiment id or the combination of experiment name and run name
        """
        if eeg_id:
            self.eeg_id = eeg_id
            return eeg_id
        elif isinstance(eeg_idx, int):
            self.eeg_id = self.available_eegs[eeg_idx].get('id', None)
            return self.eeg_id
        return None

    def print_current_experiments(self):
        pprint(self.available_experiments)

    def print_current_eegs(self):
        pprint(self.available_eegs)


if __name__ == "__main__":
    plotter = Plotter('EEGAnalyzer.sqlite')
    plotter.print_current_experiments()
    plotter.set_current_experiment(0)
    plotter.set_available_eegs()
    plotter.print_current_eegs()
    plotter.set_current_eeg(0)
    plotter.load_data()
    pass