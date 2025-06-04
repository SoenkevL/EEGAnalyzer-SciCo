"""
Advanced EEG Preprocessing Pipeline

This module provides a comprehensive EEG preprocessing pipeline designed to handle
various EEG/MEG data formats and preprocessing steps including filtering, resampling,
artifact removal using ICA, and automatic channel categorization.

Author: EEG Analysis Team
Date: 2025-06-03
"""

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple, Union
import multiprocessing as mp

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs

# Configure MNE settings
mne.set_config('MNE_BROWSER_BACKEND', 'qt')
mne.set_log_level(verbose='WARNING')


class EEGPreprocessor:
    """
    Comprehensive EEG preprocessing pipeline for neurophysiological data analysis.
    
    This class provides methods for loading, inspecting, filtering, resampling,
    and artifact removal from EEG/MEG data using MNE-Python.
    """
    
    def __init__(self, filepath: str, preload: bool = True):
        """
        Initialize the EEG preprocessor.
        
        Parameters
        ----------
        filepath : str
            Path to the EEG file (supports .edf, .fif, .set, .bdf, etc.)
        preload : bool, default=True
            Whether to preload data into memory
        """
        self.filepath = filepath
        self.raw = None
        self.ica = None
        self.channel_categories = {}
        self.preprocessing_history = []
        self.active_processes = {}  # Track active plotting processes
        
        # Load the raw data
        self.load_data(preload=preload)
        
        # Categorize channels automatically
        self.categorize_channels()
    
    def load_data(self, preload: bool = True) -> None:
        """
        Load EEG data from file.
        
        Parameters
        ----------
        preload : bool, default=True
            Whether to preload data into memory
        """
        try:
            # Determine file type and load accordingly
            if self.filepath.endswith('.edf'):
                self.raw = mne.io.read_raw_edf(self.filepath, preload=preload)
            elif self.filepath.endswith('.fif'):
                self.raw = mne.io.read_raw_fif(self.filepath, preload=preload)
            elif self.filepath.endswith('.set'):
                self.raw = mne.io.read_raw_eeglab(self.filepath, preload=preload)
            elif self.filepath.endswith('.bdf'):
                self.raw = mne.io.read_raw_bdf(self.filepath, preload=preload)
            else:
                self.raw = mne.io.read_raw(self.filepath, preload=preload)
                
            self.preprocessing_history.append(f"Loaded data from {self.filepath}")
            print(f"Successfully loaded data: {self.raw.info}")
            
        except Exception as e:
            raise ValueError(f"Failed to load data from {self.filepath}: {str(e)}")
    
    def categorize_channels(self) -> Dict[str, List[str]]:
        """
        Automatically categorize channels using regex patterns.
        
        Returns
        -------
        dict
            Dictionary with channel categories as keys and channel lists as values
        """
        # Define regex patterns for different channel types
        patterns = {
            'EEG': [
                r'^(EEG|E)\s*[A-Z]*[0-9]+',  # EEG channels
                r'^[A-Z]+[0-9]+$',           # Standard electrode names
                r'^(Fp|F|C|P|O|T)[0-9]+$',   # 10-20 system
                r'^(Fz|Cz|Pz|Oz|Fpz)$',     # Midline electrodes
                r'^A[12]$',                   # Reference electrodes
            ],
            'MEG': [
                r'MEG\s*[0-9]+',             # MEG channels
                r'MAG\s*[0-9]+',             # Magnetometers
                r'GRAD\s*[0-9]+',            # Gradiometers
            ],
            'EOG': [
                r'EOG|EYE',                   # Eye movement channels
                r'(VEOG|HEOG)',              # Vertical/Horizontal EOG
                r'(Left|Right).*Eye',        # Eye tracking
            ],
            'ECG': [
                r'(ECG|EKG)',                # Heart channels
                r'CARDIAC',
                r'HEART',
            ],
            'EMG': [
                r'EMG',                      # Muscle activity
                r'MUSCLE',
            ],
            'SPO2': [
                r'SPO2|SAT',                 # Oxygen saturation
                r'PULSE.*OX',
            ],
            'RESP': [
                r'RESP|BREATH',              # Respiration
                r'THORAX|CHEST',
            ],
            'TRIGGER': [
                r'(TRIG|STI)[0-9]*',         # Trigger channels
                r'(EVENT|MARKER)',           # Event markers
                r'Status',                   # Status channel
            ],
            'MISC': [
                r'MISC|OTHER|AUX',           # Miscellaneous
            ]
        }
        
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
        return self.channel_categories
    
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
    
    def _create_plot_process(self, target_func, args, process_name: str) -> mp.Process:
        """
        Create a multiprocessing plot that can run independently.
        
        Parameters
        ----------
        target_func : callable
            Function to run in the process
        args : tuple
            Arguments for the target function
        process_name : str
            Name identifier for the process
            
        Returns
        -------
        mp.Process
            The created process
        """
        # Clean up any finished processes
        self._cleanup_finished_processes()
        
        # Create and start new process
        process = mp.Process(target=target_func, args=args)
        process.start()
        
        # Store process with timestamp for tracking
        self.active_processes[process_name] = process
        
        return process
    
    def _cleanup_finished_processes(self) -> None:
        """Clean up finished processes from the tracking dictionary."""
        finished_processes = []
        for name, process in self.active_processes.items():
            if not process.is_alive():
                finished_processes.append(name)
        
        for name in finished_processes:
            del self.active_processes[name]
    
    def close_all_plots(self) -> None:
        """Close all active plotting processes."""
        for name, process in self.active_processes.items():
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)  # Wait up to 5 seconds
                if process.is_alive():
                    process.kill()  # Force kill if still alive
        self.active_processes.clear()
        print("Closed all active plotting processes")
    
    def list_active_plots(self) -> None:
        """List all currently active plotting processes."""
        self._cleanup_finished_processes()
        if self.active_processes:
            print("Active plotting processes:")
            for name, process in self.active_processes.items():
                status = "running" if process.is_alive() else "finished"
                print(f"  {name}: {status}")
        else:
            print("No active plotting processes")

    def _plot_eeg_data_worker(self, raw_data, duration, n_channels, start, title, bgcolor):
        """Worker function for plotting raw data in a separate process."""
        raw_data.plot(
            duration=duration,
            n_channels=n_channels,
            start=start,
            remove_dc=False,
            block=True,
            show_options=True,
            title=title,
            bgcolor=bgcolor
        )

    def plot_eeg_data(self, duration: float = 20.0, n_channels: int = 20,
                        start: float = 0.0, block: bool = False, 
                        use_multiprocessing: bool = False,
                        title: Optional[str] = None,
                        process_name: Optional[str] = None) -> Optional[mp.Process]:
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
        use_multiprocessing : bool, default=False
            Whether to open plot in a separate process
        title: str, optional, default=None
            The title of the plot
        process_name : str, optional
            Name for the process (auto-generated if not provided)

            
        Returns
        -------
        mp.Process or None
            The process object if use_multiprocessing=True, None otherwise

        """
        try:
            if use_multiprocessing:
                if process_name is None:
                    process_name = f"raw_plot_{len(self.active_processes)}"
                
                # Create a copy of raw data for the process
                raw_copy = self.raw.copy()
                
                process = self._create_plot_process(
                    target_func=self._plot_eeg_data_worker,
                    args=(raw_copy, duration, n_channels, start, title, 'white'),
                    process_name=process_name
                )
                
                print(f"Opened raw data plot in separate process: {process_name}")
                self.preprocessing_history.append(f"Inspected raw data (MP, duration={duration}s)")
                return process
            else:

                current_backend = plt.get_backend()

                self.raw.plot(
                    duration=duration,
                    n_channels=n_channels,
                    start=start,
                    remove_dc=False,
                    block=block,
                    show_options=True,
                    title=title,
                    bgcolor='white'
                )

                # Clean up after blocking plot
                if block:
                    plt.close('all')  # Close all matplotlib figures
                    # Reset the backend if needed
                    if plt.get_backend() != current_backend:
                        plt.switch_backend(current_backend)
                    # Force garbage collection
                    import gc
                    gc.collect()

                self.preprocessing_history.append(f"Inspected raw data (duration={duration}s)")
                return None
                
        except Exception as e:
            print(f"Error plotting raw data: {str(e)}")
            return None

    def plot_power_spectral_density(self, picks: Optional[Union[str, List[str]]] = None,
                                   fmin: float = 0.5, fmax: float = 50.0) -> Optional[mp.Process]:
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
        Returns
        -------
        mp.Process or None
            The process object if use_multiprocessing=True, None otherwise
        """
        if picks is None:
            picks = self.channel_categories.get('EEG', [])
            if not picks:
                picks = 'all'
        
        try:
            self.raw.plot_psd(
                picks=picks,
                fmin=fmin,
                fmax=fmax,
                show=True
            )
            self.preprocessing_history.append(f"Plotted PSD (fmin={fmin}, fmax={fmax})")
            return None
                
        except Exception as e:
            print(f"Error plotting PSD: {str(e)}")
            return None
    
    def apply_filter(self, l_freq: Optional[float] = 0.5, h_freq: Optional[float] = 40.0,
                    picks: Optional[Union[str, List[str]]] = None) -> None:
        """
        Apply bandpass filter to the data.
        
        Parameters
        ----------
        l_freq : float, optional
            Low cut-off frequency in Hz
        h_freq : float, optional
            High cut-off frequency in Hz
        picks : str or list, optional
            Channels to filter
        """
        try:
            self.raw.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                picks=picks,
                filter_length='auto',
                l_trans_bandwidth='auto',
                h_trans_bandwidth='auto',
                method='fir',
                phase='zero'
            )
            filter_info = f"l_freq={l_freq}, h_freq={h_freq}"
            self.preprocessing_history.append(f"Applied filter: {filter_info}")
            print(f"Applied bandpass filter: {filter_info}")
        except Exception as e:
            print(f"Error applying filter: {str(e)}")
    
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
            self.raw.resample(sfreq)
            self.preprocessing_history.append(f"Resampled: {original_sfreq} Hz -> {sfreq} Hz")
            print(f"Resampled data from {original_sfreq} Hz to {sfreq} Hz")
        except Exception as e:
            print(f"Error resampling data: {str(e)}")
    
    def detect_artifacts_automatic(self) -> Dict[str, List]:
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
            ecg_epochs = create_ecg_epochs(self.raw, ch_name=None, reject=None)
            artifacts['ecg_events'] = ecg_epochs.events
            print(f"Detected {len(ecg_epochs.events)} ECG events")
        except Exception as e:
            print(f"Could not detect ECG artifacts: {str(e)}")
        
        # Try to find EOG artifacts
        try:
            eog_epochs = create_eog_epochs(self.raw, ch_name=None, reject=None)
            artifacts['eog_events'] = eog_epochs.events
            print(f"Detected {len(eog_epochs.events)} EOG events")
        except Exception as e:
            print(f"Could not detect EOG artifacts: {str(e)}")
        
        self.preprocessing_history.append("Performed automatic artifact detection")
        return artifacts
    
    def fit_ica(self, n_components: Optional[int] = None, picks: Optional[Union[str, List[str]]] = None,
               crop_duration: Optional[float] = None, random_state: int = 42) -> None:
        """
        Fit Independent Component Analysis (ICA) for artifact removal.
        
        Parameters
        ----------
        n_components : int, optional
            Number of ICA components to compute
        picks : str or list, optional
            Channels to include in ICA
        crop_duration : float, optional
            Duration to crop data for ICA fitting (for computational efficiency)
        random_state : int, default=42
            Random state for reproducibility
        """
        if picks is None:
            picks = self.channel_categories.get('EEG', [])
            if not picks:
                picks = 'eeg'
        
        # Prepare data for ICA
        raw_for_ica = self.raw.copy()
        if crop_duration is not None:
            raw_for_ica.crop(tmax=crop_duration)
        
        if isinstance(picks, list) and picks:
            raw_for_ica.pick(picks)
        elif isinstance(picks, str):
            raw_for_ica.pick_types(**{picks: True})
        
        # Determine number of components
        if n_components is None:
            n_components = min(len(raw_for_ica.ch_names), 25)
        
        try:
            self.ica = ICA(
                n_components=n_components,
                max_iter='auto',
                random_state=random_state,
                method='infomax',
                fit_params=dict(extended=True)
            )
            
            self.ica.fit(raw_for_ica)
            
            # Calculate explained variance
            explained_var = self.ica.get_explained_variance_ratio(raw_for_ica)
            
            print(f"ICA fitted with {n_components} components")
            for ch_type, variance in explained_var.items():
                print(f"  {ch_type}: {variance:.2%} variance explained")
            
            self.preprocessing_history.append(f"Fitted ICA with {n_components} components")
            
        except Exception as e:
            print(f"Error fitting ICA: {str(e)}")
            self.ica = None

    def _plot_ica_components_worker(self, ica, components, title=None):
        """Worker function for plotting ICA components in a separate process."""
        if components is not None:
            ica.plot_components(picks=components, show=True, title=title)
        else:
            ica.plot_components(show=True, title=title)

    def plot_ica_components(self, components: Optional[List[int]] = None,
                           use_multiprocessing: bool = False,
                           title: Optional[str] = None,
                           process_name: Optional[str] = None) -> Optional[mp.Process]:
        """
        Plot ICA components for inspection.

        Parameters
        ----------
        components : list, optional
            Specific components to plot
        use_multiprocessing : bool, default=False
            Whether to open plot in a separate process
        process_name : str, optional
            Name for the process (auto-generated if not provided)
            
        Returns
        -------
        mp.Process or None
            The process object if use_multiprocessing=True, None otherwise
        """
        if self.ica is None:
            print("ICA has not been fitted yet. Run fit_ica() first.")
            return None
        
        try:
            if use_multiprocessing:
                if process_name is None:
                    process_name = f"ica_components_{len(self.active_processes)}"
                
                process = self._create_plot_process(
                    target_func=self._plot_ica_components_worker,
                    args=(self.ica, components, title),
                    process_name=process_name
                )
                
                print(f"Opened ICA components plot in separate process: {process_name}")
                return process
            else:
                if components is not None:
                    self.ica.plot_components(picks=components, show=True, title=title)
                else:
                    self.ica.plot_components(show=True, title=title)
                return None
                
        except Exception as e:
            print(f"Error plotting ICA components: {str(e)}")
            return None

    def _plot_ica_sources_worker(self, ica, raw_data):
        """Worker function for plotting ICA sources in a separate process."""
        ica.plot_sources(raw_data, show=True, block=True)

    def plot_ica_sources(self, duration: float = 10.0, start: float = 0.0,
                        use_multiprocessing: bool = False,
                        process_name: Optional[str] = None) -> Optional[mp.Process]:
        """
        Plot ICA sources time series.
        
        Parameters
        ----------
        duration : float, default=10.0
            Duration to plot in seconds
        start : float, default=0.0
            Start time in seconds
        use_multiprocessing : bool, default=False
            Whether to open plot in a separate process
        process_name : str, optional
            Name for the process (auto-generated if not provided)
            
        Returns
        -------
        mp.Process or None
            The process object if use_multiprocessing=True, None otherwise
        """
        if self.ica is None:
            print("ICA has not been fitted yet. Run fit_ica() first.")
            return None
        
        try:
            raw_crop = self.raw.copy().crop(tmin=start, tmax=start + duration)
            
            if use_multiprocessing:
                if process_name is None:
                    process_name = f"ica_sources_{len(self.active_processes)}"
                
                process = self._create_plot_process(
                    target_func=self._plot_ica_sources_worker,
                    args=(self.ica, raw_crop),
                    process_name=process_name
                )
                
                print(f"Opened ICA sources plot in separate process: {process_name}")
                return process
            else:
                self.ica.plot_sources(raw_crop, show=True, block=False)
                return None
                
        except Exception as e:
            print(f"Error plotting ICA sources: {str(e)}")
            return None
    
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
            self.ica.apply(self.raw)
            excluded = self.ica.exclude
            print(f"Applied ICA, excluded components: {excluded}")
            self.preprocessing_history.append(f"Applied ICA, excluded: {excluded}")
        except Exception as e:
            print(f"Error applying ICA: {str(e)}")
    
    def mark_bad_channels(self, bad_channels: List[str]) -> None:
        """
        Mark channels as bad.
        
        Parameters
        ----------
        bad_channels : list
            List of channel names to mark as bad
        """
        self.raw.info['bads'].extend([ch for ch in bad_channels if ch not in self.raw.info['bads']])
        print(f"Marked channels as bad: {bad_channels}")
        print(f"Total bad channels: {self.raw.info['bads']}")
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
    
    def set_montage(self, montage: str = 'standard_1020') -> None:
        """
        Set electrode montage for spatial information.
        
        Parameters
        ----------
        montage : str, default='standard_1020'
            Montage to use ('standard_1020', 'standard_1005', etc.)
        """
        try:
            montage_obj = mne.channels.make_standard_montage(montage)
            self.raw.set_montage(montage_obj, match_case=False, on_missing='warn')
            print(f"Set montage: {montage}")
            self.preprocessing_history.append(f"Set montage: {montage}")
        except Exception as e:
            print(f"Error setting montage: {str(e)}")
    
    def save_preprocessed(self, output_path: str, overwrite: bool = False) -> None:
        """
        Save preprocessed data to file.
        
        Parameters
        ----------
        output_path : str
            Path for output file
        overwrite : bool, default=False
            Whether to overwrite existing file
        """
        try:
            # Add preprocessing history to info
            self.raw.info['description'] = '; '.join(self.preprocessing_history)
            
            self.raw.save(output_path, overwrite=overwrite)
            print(f"Saved preprocessed data to: {output_path}")
            self.preprocessing_history.append(f"Saved to: {output_path}")
        except Exception as e:
            print(f"Error saving file: {str(e)}")
    
    def get_preprocessing_summary(self) -> str:
        """
        Get a summary of all preprocessing steps performed.
        
        Returns
        -------
        str
            Summary of preprocessing steps
        """
        summary = "\n" + "="*60 + "\n"
        summary += "PREPROCESSING SUMMARY\n"
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
    
    # Inspect raw data with multiprocessing
    print("\n1. Inspecting raw data...")
    raw_plot_process = preprocessor.plot_eeg_data(
        duration=20,
        # use_multiprocessing=True,
        block=True,
        title='unprocessed raw eeg',
        process_name="initial_raw_inspection"
    )
    
    # Plot power spectral density with multiprocessing
    print("\n2. Plotting power spectral density...")
    psd_plot_process = preprocessor.plot_power_spectral_density(
        fmin=0.5, 
        fmax=50,
    )
    
    # Apply filtering
    print("\n3. Applying bandpass filter...")
    preprocessor.apply_filter(l_freq=0.5, h_freq=40.0)
    
    # Show filtered data
    print("\n4. Inspecting filtered data...")
    filtered_plot_process = preprocessor.plot_eeg_data(
        duration=20, 
        # use_multiprocessing=True,
        block=True,
        title='eeg after filtering',
        process_name="filtered_raw_inspection"
    )
    
    # Resample data
    print("\n5. Resampling data...")
    preprocessor.resample_data(sfreq=256.0)
    
    # Detect artifacts automatically
    print("\n6. Detecting artifacts...")
    artifacts = preprocessor.detect_artifacts_automatic()

    # Set montage
    print("\n7. Setting electrode montage...")
    preprocessor.set_montage('standard_1020')

    # Fit ICA
    print("\n8. Fitting ICA...")
    preprocessor.fit_ica(n_components=15, crop_duration=60)

    # Plot ICA components with multiprocessing
    if preprocessor.ica is not None:
        print("\n9. Plotting ICA components...")
        ica_comp_process = preprocessor.plot_ica_components(
            use_multiprocessing=False,
            process_name="ica_components"
        )
        
        ica_sources_process = preprocessor.plot_ica_sources(
            duration=10,
            use_multiprocessing=False,
            process_name="ica_sources"
        )
    

    # List active plots
    print("\n10. Listing active plots...")
    preprocessor.list_active_plots()
    
    # Save preprocessed data
    if output_path:
        print(f"\n11. Saving preprocessed data to {output_path}...")
        preprocessor.save_preprocessed(output_path, overwrite=True)
    
    # Print summary
    print(preprocessor.get_preprocessing_summary())
    
    print("\nNote: Multiple plots are now open in separate processes.")
    print("Use preprocessor.list_active_plots() to see active plots")
    print("Use preprocessor.close_all_plots() to close all plots")
    
    return preprocessor


if __name__ == "__main__":
    # Example usage demonstrating multiprocessing capabilities
    sample_file = "../example/eeg/PN001-original.edf"
    output_file = "../example/eeg/PN001-preprocessed.fif"
    
    if os.path.exists(sample_file):
        preprocessor = example_preprocessing_pipeline(sample_file, output_file)

    else:
        print(f"Sample file not found: {sample_file}")
        print("Please update the filepath or provide your own EEG file.")