"""
Advanced EEG Preprocessing Pipeline

This module provides a comprehensive EEG preprocessing pipeline designed to handle
various EEG/MEG data formats and preprocessing steps including filtering, resampling,
artifact removal using ICA, and automatic channel categorization.

Author: Soenke van Loh
Date: 2025-06-03
"""

from eeganalyzer.preprocessing.PreprocessingFunctions import *
import re
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from typing import Dict, List, Optional, Union, Any
import matplotlib.pyplot as plt
from pprint import pprint
from eeganalyzer.preprocessing.channel_regex_patterns import get_default_patterns, merge_patterns, get_mne_channel_types
import argparse
import os
import datetime
import logging
import mne

# Configure MNE settings
mne.set_config('MNE_BROWSER_BACKEND', 'qt')
mne.set_log_level(verbose='WARNING')


class EEGPreprocessor:
    """
    Comprehensive EEG preprocessing pipeline for neurophysiological data analysis.

    This class provides methods for loading, inspecting, filtering, resampling,
    and artifact removal from EEG/MEG data using MNE-Python.
    """

    def __init__(self, filepath: str, preload: bool = True, log_level: str = 'INFO'):
        """
        Initialize the EEG preprocessor with enhanced logging.
        
        Parameters
        ----------
        filepath : str
            Path to the EEG file
        preload : bool, default=True
            Whether to preload data into memory
        log_level : str, default='INFO'
            Logging level for MNE and custom logger
        """
        self.filepath = filepath
        self.raw = None
        self.ica = None
        self.channel_categories = {}
        self.preprocessing_history = []
        
        # Setup log file paths
        now = datetime.datetime.now()
        self.log_filename = f"{os.path.splitext(filepath)[0]}_preprocessing__{now:%Y_%m_%d_%H_%M_%S}.log"
        
        # Configure MNE logging - this is the key fix!
        mne.set_log_level(log_level)
        mne.set_log_file(self.log_filename, output_format='%(asctime)s - MNE - %(levelname)s - %(message)s')
        
        # Setup custom logger
        self.logger = logging.getLogger(f'EEGPreprocessor_{os.path.basename(filepath)}')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        self._setup_logging()
        
        self.logger.info(f"Initializing EEG Preprocessor for {filepath}")
        self.logger.info(f"MNE logging configured to write to: {self.log_filename}")
        self.load_data(preload=preload)

    def _setup_logging(self):
        """Setup file and console logging handlers."""
        # Check if handlers already exist to avoid duplicates
        if not self.logger.handlers:
            # File handler - append to the same file that MNE uses
            file_handler = logging.FileHandler(self.log_filename, mode='a')
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def load_data(self, preload: bool = True) -> None:
        """Load EEG data with enhanced logging."""
        self.logger.info(f"Loading data from {self.filepath}")
        
        try:
            # MNE will now automatically log to the file we specified
            if self.filepath.endswith('.edf'):
                self.raw = mne.io.read_raw_edf(self.filepath, preload=preload, verbose=True)
            elif self.filepath.endswith('.fif'):
                self.raw = mne.io.read_raw_fif(self.filepath, preload=preload, verbose=True)
            elif self.filepath.endswith('.set'):
                self.raw = mne.io.read_raw_eeglab(self.filepath, preload=preload, verbose=True)
            elif self.filepath.endswith('.bdf'):
                self.raw = mne.io.read_raw_bdf(self.filepath, preload=preload, verbose=True)
            else:
                self.raw = mne.io.read_raw(self.filepath, preload=preload, verbose=True)
            
            # Log detailed information about loaded data
            self.logger.info(f"Successfully loaded data: {len(self.raw.ch_names)} channels, "
                           f"{self.raw.info['sfreq']} Hz, {self.raw.times[-1]:.2f} seconds")
            self.logger.debug(f"Data info: {self.raw.info}")


            # There is a problem with the annotations in the edf files so we need to update them from custom matlab files
            self._add_custom_annot()
            self.logger.info('Updated annotations from .mat files')

            self.preprocessing_history.append(f"Loaded data from {self.filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {self.filepath}: {str(e)}")
            raise ValueError(f"Failed to load data from {self.filepath}: {str(e)}")

    def _add_custom_annot(self):
        """
        Custom function to deal with the EIRatio annotations to load from the matlab files from my master
        Returns:
            None
        """
        annot_path = self.filepath.replace('original.edf', 'annot-sz.mat')
        update_annotations_suzanne(self.raw, annot_path, self.filepath, method='replace', recompute=False)

    def categorize_channels_legacy(self, mark_unclassified_as_bad = False,
                            patterns=None, merge_with_default=True,
                            save_types_in_info=True) -> Dict[str, List[str]]:
        """
        Automatically categorize channels using regex patterns.
        
        Returns
        -------
        dict
            Dictionary with channel categories as keys and channel lists as values
        """
        # Define regex patterns for different channel types
        default_patterns = get_default_patterns()
        if not patterns:
            patterns = default_patterns
        elif merge_with_default:
            patterns = merge_patterns(patterns)
        else:
            patterns = patterns

        self.channel_categories = {category: [] for category in patterns.keys()}
        unclassified = []
        
        for ch_name in self.raw.ch_names:
            classified = False
            for category, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, ch_name, re.IGNORECASE):
                        self.channel_categories[category].append(ch_name)
                        classified = True
                        break
                if classified:
                    break
            
            if not classified:
                unclassified.append(ch_name)
        
        if unclassified:
            self.channel_categories['UNCLASSIFIED'] = unclassified
        
        self.preprocessing_history.append("Categorized channels")
        self.logger.info("Categorized the channels")

        if save_types_in_info:
            valid_channel_types=get_mne_channel_types()
            ch_types = {}
            for category, channels in self.channel_categories.items():
                if category.lower() in valid_channel_types:
                    for channel in channels:
                        ch_types[channel] = category.lower()
            if ch_types:
                self.raw.set_channel_types(ch_types)
        if mark_unclassified_as_bad:
            self.mark_bad_channels(unclassified)
            self.preprocessing_history.append("Marked unclassified channels as bad")
            self.logger.info('marked unclassified channels as bad')
        return self.channel_categories

    def categorize_channels(self, mark_unclassified_as_bad=False):
        CUSTOM_PATTERNS = {
            'EOG': [
                r'.*Ref-?1.*',  # Matches anything containing 'Ref-0' or 'Ref0'
            ],
            'ECG': [
                r'.*[Ii][Nn].*',  # Matches anything containing 'In' or 'ln' (case insensitive)
            ]
        }
        return self.categorize_channels(patterns=CUSTOM_PATTERNS, merge_with_default=True,
                                                mark_unclassified_as_bad=mark_unclassified_as_bad)

    def print_channel_info(self) -> None:
        """Print detailed information about channels and their categories."""
        print("\n" + "="*60)
        print("CHANNEL INFORMATION")
        print("="*60)
        
        print(f"Total channels: {len(self.raw.ch_names)}")
        print(f"Sampling frequency: {self.raw.info['sfreq']} Hz")
        print(f"Duration: {self.raw.times[-1]:.2f} seconds")
        
        print("\nChannel Categories:")
        print("-" * 30)
        for category, channels in self.channel_categories.items():
            if channels:
                print(f"{category}: {len(channels)} channels")
                if len(channels) <= 10:
                    print(f"  {', '.join(channels)}")
                else:
                    print(f"  {', '.join(channels[:5])} ... {', '.join(channels[-2:])}")
        
        print(f"\nBad channels: {self.raw.info['bads']}")

    def plot_eeg_data(self, duration: float = 20.0, n_channels: int = 20,
                        start: float = 0.0, block: bool = True, remove_dc = False,
                        title: Optional[str] = None,
                        plot_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Visually inspect raw EEG data.
        
        Parameters
        ----------
        duration : float, default=20.0
            Duration of data to display in seconds
        n_channels : int, default=20
            Number of channels to display
        start : float, default=0.0
            Start time for display
        block : bool, default=False
            Whether to block execution until plot is closed
        remove_dc: bool, default=False
            Whether to remove the dc from the channels
        title: str, optional, default=None
            The title of the plot
        plot_kwargs: dict, optional
            kwargs given to the plotting function

            
        Returns
        -------
       None

        """
        try:
            if plot_kwargs:
                self.raw.plot(
                    duration=duration,
                    n_channels=n_channels,
                    start=start,
                    remove_dc=remove_dc,
                    block=block,
                    show_options=True,
                    title=title,
                    bgcolor='white',
                    **plot_kwargs
                )
            else:
                self.raw.plot(
                    duration=duration,
                    n_channels=n_channels,
                    start=start,
                    remove_dc=remove_dc,
                    block=block,
                    show_options=True,
                    title=title,
                    bgcolor='white',
                )
            return None
                
        except Exception as e:
            print(f"Error plotting raw data: {str(e)}")
            return None

    def plot_power_spectral_density(self, picks: Optional[Union[str, List[str]]] = None,
                                    fmin: float = 0.5, fmax: float = 50.0, title: Optional[str] = None,
                                    show=False) -> plt.figure:
        """
        Plot power spectral density of the data.
        
        Parameters
        ----------
        picks : str or list, optional
            Channels to include in PSD plot
        fmin : float, default=0.5
            Minimum frequency to display
        fmax : float, default=50.0
            Maximum frequency to display
        title: str, optional
            title of the plot
        show: bool, default=False
            if the plot should be shown
        Returns
        -------
        None
        """
        if picks is None:
            picks = self.channel_categories.get('EEG', [])
            if not picks:
                picks = 'all'
        
        try:
            psd_fig = self.raw.plot_psd(
                picks=picks,
                fmin=fmin,
                fmax=fmax,
                show=False
            )
            psd_fig.suptitle(title)
            if show:
                psd_fig.show()
            return psd_fig
                
        except Exception as e:
            print(f"Error plotting PSD: {str(e)}")
            return None
    
    def apply_filter(self, l_freq: Optional[float] = 0.5, h_freq: Optional[float] = 40.0,
                    picks: Optional[Union[str, List[str]]] = None) -> None:
        """Apply bandpass filter with detailed logging."""
        filter_info = f"l_freq={l_freq}, h_freq={h_freq}, picks={picks}"
        self.logger.info(f"Applying filter: {filter_info}")
        
        try:
            # MNE will log filter details to the file automatically
            self.raw.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                picks=picks if picks else 'all',
                filter_length='auto',
                l_trans_bandwidth='auto',
                h_trans_bandwidth='auto',
                method='fir',
                phase='zero',
                verbose=True
            )
            
            self.logger.info(f"Filter applied successfully: {filter_info}")
            self.preprocessing_history.append(f"Applied filter: {filter_info}")
            
        except Exception as e:
            self.logger.error(f"Error applying filter: {str(e)}")

    def resample_data(self, sfreq: float) -> None:
        """
        Resample the data to a new sampling frequency.
        
        Parameters
        ----------
        sfreq : float
            New sampling frequency in Hz
        """
        try:
            original_sfreq = self.raw.info['sfreq']
            self.raw.resample(sfreq, verbose=True)
            self.preprocessing_history.append(f"Resampled: {original_sfreq} Hz -> {sfreq} Hz")
            self.logger.info('Applied resampling successfully')
            print(f"Resampled data from {original_sfreq} Hz to {sfreq} Hz")
        except Exception as e:
            self.logger.info('resampling failed')
            print(f"Error resampling data: {str(e)}")

    def find_flat_channels_psd(self, f_ratio_flat = 0.5, l_freq=1, h_freq=40, show=False) -> List:
        """
        Identifies channels with flat frequency power spectrum in EEG data.

        This method processes raw EEG data to identify channels having a flat
        power spectral density (PSD). It applies a bandpass filter to the data,
        computes the PSD, and compares frequency power values for each channel
        against the median power across frequencies. Channels with a sufficiently
        high proportion of frequency bins showing power below a threshold are
        considered flat channels.

        Parameters:
        f_ratio_flat: float
            The fraction of frequency bins that must have power below the specified
            threshold for a channel to be marked as flat. Default is 0.5.

        Returns:
        List
            A list of names of the channels identified as flat.
        """
        temp_raw = self.raw.copy()
        temp_raw = temp_raw.filter(l_freq=l_freq, h_freq=h_freq)
        spectrum = temp_raw.compute_psd(fmax=h_freq+5)
        if show:
            spectrum.plot()
        spectral_data = spectrum.get_data()
        median_power_per_freq = np.median(spectral_data, axis=0)
        flat_freqs = np.zeros_like(spectral_data)
        for i, row in enumerate(spectral_data):
           flat_freqs[i] = row < median_power_per_freq/20
        number_freq_bins = flat_freqs.shape[1]
        mark_channel_names = []
        for i, row in enumerate(flat_freqs):
            if np.sum(row) >= number_freq_bins*f_ratio_flat:
                mark_channel_names.append(spectrum.ch_names[i])
        return mark_channel_names

    def detect_artifacts_automatic(self, ecg_channel: Optional[str]=None,
                                   eog_channels: Optional[Union[str | list[str]]]=None) -> Dict[str, List]:
        """
        Automatically detect ECG and EOG artifacts.

        Returns
        -------
        dict
            Dictionary containing detected artifacts
        """
        artifacts = {'ecg_events': [], 'eog_events': []}

        # Try to find ECG artifacts
        try:
            ecg_epochs = create_ecg_epochs(self.raw, ch_name=ecg_channel, reject=None)
            artifacts['ecg_events'] = ecg_epochs.events
            print(f"Detected {len(ecg_epochs.events)} ECG events")
        except Exception as e:
            print(f"Could not detect ECG artifacts: {str(e)}")

        # Try to find EOG artifacts
        try:
            eog_epochs = create_eog_epochs(self.raw, ch_name=eog_channels, reject=None)
            artifacts['eog_events'] = eog_epochs.events
            print(f"Detected {len(eog_epochs.events)} EOG events")
        except Exception as e:
            print(f"Could not detect EOG artifacts: {str(e)}")
        
        self.preprocessing_history.append("Performed automatic artifact detection")
        return artifacts
    
    def fit_ica(self, n_components: Optional[Union[int, float]] = None, 
                picks: Optional[Union[str, List[str]]] = None,
                t_min: Optional[float] = 0, crop_duration: Optional[float] = None, 
                filter_kwargs: Optional[Dict[str, Any]] = None,
                plot_eeg=False, plot_block=False,
                random_state: int = 42) -> None:
        """
        Fit Independent Component Analysis (ICA) for artifact removal.

        Parameters
        ----------
        n_components : int, optional
            Number of ICA components to compute
        picks : str or list, optional
            Channels to include in ICA, if None all channels not marked as bad are included
        t_min : float, optional
            Where to start the cropped section
        crop_duration : float, optional
            Duration to crop data for ICA fitting (for computational efficiency)
        filter_kwargs: Dict, optional
            arguments for raw.filter that is applied before the ica decomposition but will not be applied to the original raw file
            of the processor. Leaf out for no filtering
        plot_eeg: bool, default=False
            if the eeg used for the fitting should be plotted beforehand
        plot_block: bool, default=False
            If the plot of the eeg used for ica should block the fitting
        random_state : int, default=42
            Random state for reproducibility
        """
        if picks is None:
            picks = 'data'

        self.logger.info(f"Starting ICA fitting with n_components={n_components}, "
                        f"picks={picks}, crop_duration={crop_duration}")

        # Prepare data for ICA
        raw_for_ica = self.raw.copy()
        if crop_duration is not None:
            self.logger.info(f"Cropping data for ICA: {t_min}s to {t_min + crop_duration}s")
            raw_for_ica.crop(tmin=t_min, tmax=t_min + crop_duration)

        raw_for_ica.pick(picks)
        self.logger.info(f"Selected {len(raw_for_ica.ch_names)} channels for ICA")

        if filter_kwargs:
            self.logger.info(f"Applying pre-ICA filtering: {filter_kwargs}")
            raw_for_ica.filter(**filter_kwargs, verbose=True)

        if plot_eeg:
            raw_for_ica.plot(block=plot_block, title='EEG used for ICA fitting')

        # Determine number of components
        if n_components is None:
            n_components = min(len(raw_for_ica.ch_names), 25)
        
        self.logger.info(f"Fitting ICA with {n_components} components")
        
        try:
            self.ica = ICA(
                n_components=n_components,
                max_iter='auto',
                random_state=random_state,
                method='infomax',
                fit_params=dict(extended=True)
            )
            
            # MNE will log ICA fitting progress to the file
            self.ica.fit(raw_for_ica, verbose=True)
            
            # Calculate and log explained variance
            explained_var = self.ica.get_explained_variance_ratio(raw_for_ica)
            
            self.logger.info(f"ICA fitted successfully with {n_components} components")
            for ch_type, variance in explained_var.items():
                self.logger.info(f"  {ch_type}: {variance:.2%} variance explained")
            
            self.preprocessing_history.append(f"Fitted ICA with {n_components} components")
            
        except Exception as e:
            self.logger.error(f"Error fitting ICA: {str(e)}")
            self.ica = None

    def plot_ica_components(self, components: Optional[List[int]] = None,
                           title: Optional[str] = None, show=False) -> plt.figure:
        """
        Plot spatial ICA components for inspection.

        Parameters
        ----------
        components : list, optional
            Specific components to plot
        title: str, optional
            Title for the plot
        show: bool, default=False
            If the plot should be shown
            
        Returns
        -------
        None
        """
        if self.ica is None:
            print("ICA has not been fitted yet. Run fit_ica() first.")
            return None
        
        try:
            if components is not None:
                fig = self.ica.plot_components(picks=components, show=show, title=title)
            else:
                fig = self.ica.plot_components(show=show, title=title)
            return fig
                
        except Exception as e:
            print(f"Error plotting ICA components: {str(e)}")
            return None

    def plot_ica_sources(self, duration: int = 10, start: int = 0, block=True) -> None:
        """
        Plot ICA sources time series.
        
        Parameters
        ----------
        duration : float, default=10.0
            Duration to plot in seconds
        start : float, default=0.0
            Start time in seconds
        block: bool, default=True
            If the plot should block the process

        Returns
        -------
        None
        """
        #TODO: removing the EEG in this plot crashed the programm, investigate why and if that is intended behaviour
        if self.ica is None:
            print("ICA has not been fitted yet. Run fit_ica() first.")
            return None
        
        try:
            self.ica.plot_sources(self.raw, start=start, stop=start+duration, show=True, block=block)
            return None
                
        except Exception as e:
            print(f"Error plotting ICA sources: {str(e)}")
            return None

    def plot_ica_properties(self, component: Optional[int]=None) -> List[plt.figure]:
        """
        Plots the properties of the Independent Component Analysis (ICA) results.

        This method visualizes the aspects of the ICA decomposition performed on the
        raw data. It can include components' spectra, topographies, and more, providing
        insights into the results of the ICA analysis.

        Args:
            self: Object instance context where the `ica` attribute refers to an
                  ICA object and the `raw` attribute refers to raw data involved
                  in ICA decomposition.
            component: Optional, int
                which component should be displayed, by default when left as None the first 5 will be returned

        Returns:
            None
        """
        return self.ica.plot_properties(self.raw, picks=component)

    def print_ica_variance(self):
        pprint(self.ica.get_explained_variance_ratio(self.raw))
    
    def exclude_ica_components(self, components: List[int]) -> None:
        """
        Mark ICA components for exclusion.
        
        Parameters
        ----------
        components : list
            List of component indices to exclude
        """
        if self.ica is None:
            print("ICA has not been fitted yet. Run fit_ica() first.")
            return
        
        self.ica.exclude = components
        print(f"Marked components {components} for exclusion")
        self.preprocessing_history.append(f"Excluded ICA components: {components}")
        self.logger.info(f"Excluded ICA components: {components}")
    
    def apply_ica(self, exclude: Optional[List[int]] = None) -> None:
        """
        Apply ICA to remove artifacts from the data.
        
        Parameters
        ----------
        exclude : list, optional
            Components to exclude (if not already set)
        """
        if self.ica is None:
            print("ICA has not been fitted yet. Run fit_ica() first.")
            return
        
        if exclude is not None:
            self.ica.exclude = exclude
        
        try:
            self.ica.apply(self.raw, verbose=True)
            excluded = self.ica.exclude
            print(f"Applied ICA, excluded components: {excluded}")
            self.preprocessing_history.append(f"Applied ICA, excluded: {excluded}")
        except Exception as e:
            print(f"Error applying ICA: {str(e)}")

    def run_ica_fitting(self):
        # Fit ICA
        print("\n9. Fitting ICA...")
        ica_channels = [self.channel_categories.get(category, []) for category in ['EEG', 'EMG', 'ECG', 'EOG']]
        ica_channels = [channel for sublist in ica_channels for channel in sublist]
        self.fit_ica(n_components=15, crop_duration=60,
                     picks=ica_channels,
                     filter_kwargs={
                         'l_freq': 1,
                         'h_freq': 40,
                        }
                     )

    def run_ica_selection(self, apply=True):
        # Plot ICA components with multiprocessing
        if self.ica is not None:
            self.plot_ica_sources(duration=60)
            # Remove ica components that were excluded
            if apply:
                self.apply_ica()

    def run_full_ica(self):
        # Fit ICA
        print("\n9. Fitting ICA...")
        ica_channels = [self.channel_categories.get(category, []) for category in ['EEG', 'EMG', 'ECG', 'EOG']]
        ica_channels = [channel for sublist in ica_channels for channel in sublist]
        self.fit_ica(n_components=15, crop_duration=60,
                     picks=ica_channels
                     )
        # Plot ICA components with multiprocessing
        if self.ica is not None:
            print("\n10. Plotting ICA components...")
            self.plot_ica_components()

            self.plot_ica_sources(duration=60)

        # Remove ica components that were excluded
        self.apply_ica()

    def mark_bad_channels(self, bad_channels: List[str]) -> None:
        """
        Mark channels as bad.
        
        Parameters
        ----------
        bad_channels : list
            List of channel names to mark as bad
        """
        self.raw.info['bads'].extend([ch for ch in bad_channels if ch not in self.raw.info['bads']])
        self.logger.info(f"Marked channels as bad: {bad_channels}")
        self.logger.info(f"Total bad channels: {self.raw.info['bads']}")
        self.preprocessing_history.append(f"Marked bad channels: {bad_channels}")
    
    def interpolate_bad_channels(self) -> None:
        """Interpolate bad channels using spherical splines."""
        if not self.raw.info['bads']:
            print("No bad channels to interpolate")
            return
        
        try:
            self.raw.interpolate_bads(reset_bads=True)
            print(f"Interpolated bad channels")
            self.preprocessing_history.append("Interpolated bad channels")
        except Exception as e:
            print(f"Error interpolating bad channels: {str(e)}")

    def set_montage(self, montage):
        try:
            montage_obj = mne.channels.make_standard_montage(montage)
            self.raw.set_montage(montage_obj, match_case=False, on_missing='warn')
            self.logger.info(f"Set montage: {montage}")
            self.preprocessing_history.append(f"Set montage: {montage}")
        except Exception as e:
            print(f"Error setting montage: {str(e)}")

    def fit_montage(self, montage: str = 'standard_1020', show_example=False) -> None:
        """
        Set electrode montage for spatial information.
        
        Parameters
        ----------
        montage : str, default='standard_1020'
            Montage to use ('standard_1020', 'standard_1005', etc.)
        show_example: bool, default=False
            If a plot of the default montage should be created
        """
        # Montage
        # print('\n Setting up montage for the eeg, using a standard montage from mne')
        montage_name = montage
        ## Show montage and get object
        if show_example:
            montage = show_example_montage(montage)
        else:
            montage = make_montage(montage_name)
        ## Create electrode mapping
        raw_orig_ch_names = self.raw.ch_names
        montage_ch_names = montage.ch_names
        mapping_dict, unmatched = create_electrode_mapping(montage_ch_names, raw_orig_ch_names)
        self.logger.info('Creating channel name mapping to fit a montage')
        self.logger.info(mapping_dict)
        self.logger.info(f'Could not match channels: {unmatched}')
        ## Rename channel names
        self.rename_channels(mapping_dict)
        ## Apply electrode
        self.set_montage(montage_name)
        ## Recategorize the channels
        self.categorize_channels()

    def save_preprocessed(self, output_path: str, overwrite: bool = False, 
                         create_external_logfile: bool = True) -> None:
        """Save preprocessed data with logging.

        Parameters
        ----------
        output_path : str
            Path for output file
        overwrite : bool, default=False
            Whether to overwrite existing file
        create_external_logfile: bool, default=False
            If the preprocessing summary should also be saved to a textfile
        """
        self.logger.info(f"Saving preprocessed data to {output_path}")
        
        try:
            # Add preprocessing history to info
            self.raw.info['description'] = '; '.join(self.preprocessing_history)
            
            # MNE will log saving details to the file
            self.raw.save(output_path, overwrite=overwrite, verbose=True)
            
            self.logger.info(f"Successfully saved preprocessed data to: {output_path}")
            self.preprocessing_history.append(f"Saved to: {output_path}")
            
            if create_external_logfile:
                output_path_ending = os.path.splitext(output_path)[1]
                logpath = output_path.replace(output_path_ending, '.log')
                self.preprocessing_summary_to_logfile(logpath)
                
        except Exception as e:
            self.logger.error(f"Error saving file: {str(e)}")

    def close_logging(self):
        """Close the MNE log file and clean up handlers."""
        # Close MNE logging
        mne.set_log_file(None)
        
        # Close custom logger handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        self.logger.info("Logging session closed")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.close_logging()
        except:
            pass  # Ignore errors during cleanup

    def get_preprocessing_summary(self) -> str:
        """
        Get a summary of all preprocessing steps performed.
        
        Returns
        -------
        str
            Summary of preprocessing steps
        """
        now = datetime.datetime.now()
        summary = "\n" + "="*60 + "\n"
        summary += "PREPROCESSING SUMMARY\n"
        summary += f'{now:%c}\n'
        summary += "="*60 + "\n"
        summary += f"File: {self.filepath}\n"
        summary += f"Channels: {len(self.raw.ch_names)} total\n"
        summary += f"Sampling rate: {self.raw.info['sfreq']} Hz\n"
        summary += f"Duration: {self.raw.times[-1]:.2f} seconds\n"
        summary += f"Bad channels: {len(self.raw.info['bads'])}\n"
        summary += "\nProcessing steps:\n"
        summary += "-" * 30 + "\n"
        for i, step in enumerate(self.preprocessing_history, 1):
            summary += f"{i:2d}. {step}\n"
        summary += "="*60
        return summary

    def preprocessing_summary_to_logfile(self, filepath):
        """
        Save preprocessing summary to a log file.

        Parameters
        ----------
        filepath : str
            Path to save the log file
        """

        # Check if filepath exists
        if not os.path.exists(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            with open(filepath, 'w') as f:
                f.write(self.get_preprocessing_summary())
            print(f"Saved preprocessing summary to: {filepath}")
            self.preprocessing_history.append(f"Saved preprocessing summary to: {filepath}")
        except Exception as e:
            print(f"Error saving preprocessing summary: {str(e)}")

    def rename_channels(self, mapping: Dict):
        """
        Rename EEG channels using a mapping dictionary.
    
        Parameters
        ----------
        mapping : dict
            Dictionary containing the mapping of old channel names to new channel names
        """
        try:
            self.raw.rename_channels(mapping)
            self.preprocessing_history.append(f"Renamed channels using mapping")
            print(f"Successfully renamed channels according to provided mapping")
        except Exception as e:
            print(f"Error renaming channels: {str(e)}")


def example_preprocessing_pipeline(filepath: str, output_path: Optional[str] = None):
    """
    Example preprocessing pipeline demonstrating the use of EEGPreprocessor.

    Parameters
    ----------
    filepath : str
        Path to input EEG file
    output_path : str, optional
        Path for output file
    """
    # Initialize preprocessor
    print("Initializing EEG Preprocessor...")
    preprocessor = EEGPreprocessor(filepath)
    # Print channel information
    preprocessor.print_channel_info()
    CUSTOM_PATTERNS = {
        'EOG': [
            r'.*Ref-?1.*',  # Matches anything containing 'Ref-0' or 'Ref0'
        ],
        'ECG': [
            r'.*[Ii][Nn].*',  # Matches anything containing 'In' or 'ln' (case insensitive)
        ]
    }
    preprocessor.categorize_channels(patterns=CUSTOM_PATTERNS, merge_with_default=True)
    print(preprocessor.channel_categories.get('EEG'))

    # Inspect raw data
    print("\n0. Inspecting raw data...")
    print('Here we plot the eeg for the first time, allowing us to mark obvious bad channels')
    preprocessor.raw.plot(
        duration=20,
        block=True,
        title='unprocessed raw eeg',
        remove_dc=True,
        theme='light'
    )


    # Montage
    print('\n Setting up montage for the eeg, using a standard montage from mne')
    montage_name= 'standard_1020'
    ## Show montage and get object
    montage = show_example_montage(montage_name)
    ## Create electrode mapping
    raw_orig_ch_names = preprocessor.channel_categories.get('EEG', [])
    montage_ch_names = montage.ch_names
    mapping_dict, unmatched = create_electrode_mapping(montage_ch_names, raw_orig_ch_names)
    ## Rename channel names
    preprocessor.rename_channels(mapping_dict)
    ## Apply electrode
    preprocessor.set_montage(montage_name)
    ## Recategorize the channels
    preprocessor.categorize_channels(patterns=CUSTOM_PATTERNS, merge_with_default=True)

    # Mark unclassified channels as bad
    preprocessor.mark_bad_channels(preprocessor.channel_categories.get('UNCLASSIFIED', []))

    # Mark flat channels as bad
    flat_channels = preprocessor.find_flat_channels_psd()
    preprocessor.mark_bad_channels(flat_channels)

    # Show filtered data
    print("\n1. Initial data inspection filtered within neurologically relevant sections for scalp eeg")
    # preprocessor.plot_eeg_data(
    #     duration=20,
    #     block=False,
    #     title='Initial eeg inspection',
    #     plot_kwargs={
    #         'highpass': 0.5,
    #         'lowpass': 70,
    #     }
    # )

    # Plot power spectral density
    print("\n2. Plotting power spectral density prior preprocessing...")
    preprocessor.plot_power_spectral_density(
        fmin=0.5,
        fmax=70,
        title='PSD of unprocessed EEG',
        picks=preprocessor.channel_categories.get('EEG', None)
    )

    #Apply filtering
    print("\n3. Applying bandpass filter in paper target range")
    preprocessor.apply_filter(l_freq=1, h_freq=40)

    #Resample data
    # print("\n4. Resampling data to match the sampling frequency of the target analysis and unify sampling frequencies across eegs")
    # preprocessor.resample_data(sfreq=1024)

    #Plot power spectral density with multiprocessing
    print("\n5. Plotting power spectral density after processing...")
    preprocessor.plot_power_spectral_density(
        fmin=0.5,
        fmax=50,
        title='PSD of filtered and resampled EEG',
        show=True
    )

    # Detect artifacts automatically
    #TODO: something seems wrong with the artifact detection, as EKG channel is in list but not detected

    # print("\n7. Detecting artifacts...")
    # artifacts = preprocessor.detect_artifacts_automatic(ecg_channel=preprocessor.channel_categories.get('ECG', None),
    #                                                     eog_channels=preprocessor.channel_categories.get('EOG', None))

    #TODO: Implement dead channel detection

    # Fit ICA
    print("\n9. Fitting ICA...")
    ica_channels = [preprocessor.channel_categories.get(category, []) for category in ['EEG', 'EMG', 'ECG', 'EOG']]
    ica_channels = [channel for sublist in ica_channels for channel in sublist]
    preprocessor.fit_ica(n_components=0.99, crop_duration=360,
                         t_min=10,
                         picks=ica_channels
                         )

    # Plot ICA components with multiprocessing
    if preprocessor.ica is not None:
        print("\n10. Plotting ICA components...")
        preprocessor.plot_ica_components()
        # preprocessor.plot_ica_sources_psd()
        preprocessor.print_ica_variance()
        preprocessor.plot_ica_properties(component=0)

        preprocessor.plot_ica_sources(duration=60)

    # Remove ica components that were excluded
    preprocessor.apply_ica()

    # Mark all channels as bad which are not eeg channels
    exclude_channels = [channel for channel in preprocessor.raw.ch_names if channel not in preprocessor.channel_categories.get('EEG', [])]
    preprocessor.mark_bad_channels(exclude_channels)

    # One last check of the final eeg that is saved
    print("\n11. Inspecting the final eeg that will be saved...")
    print('Here we plot the eeg for the last time, controlling the final version of the preprocessed eeg')
    preprocessor.plot_eeg_data(
        duration=20,
        block=True,
        title='final eeg',
    )

    # Save preprocessed data
    if output_path:
        print(f"\n11. Saving preprocessed data to {output_path}...")
        preprocessor.save_preprocessed(output_path, overwrite=True)

    # Print summary
    print(preprocessor.get_preprocessing_summary())
    return preprocessor

def choose_montage():
    print_all_builtin_montages()
    montage = show_example_montage('standard_1020')
    print(montage.ch_names)

    if __name__ == "__main__":

        parser = argparse.ArgumentParser(description='EEG Preprocessing Pipeline')
        parser.add_argument('input_file', help='Path to input EEG file')
        parser.add_argument('-o', '--output', help='Path to output file (optional)')
        args = parser.parse_args()

        input_file = args.input_file
        if args.output:
            output_file = args.output
        else:
            # Generate output filename by replacing the extension with preprocessed-raw.fif
            base_path = os.path.splitext(input_file)[0]
            output_file = f"{base_path}_preprocessed-raw.fif"

        if os.path.exists(input_file):
            preprocessor = example_preprocessing_pipeline(input_file, output_file)
        else:
            print(f"Input file not found: {input_file}")
            print("Please provide a valid input file path.")
