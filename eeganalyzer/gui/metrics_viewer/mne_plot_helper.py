#!/usr/bin/env python3
"""
Copyright (C) <2025>  <Soenke van Loh>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

Helper script to display MNE plots in a separate process.

This script is called by the main application to display MNE plots without blocking
the main application thread.
"""

import sys
import os
import mne
import argparse


def main():
    """Parse arguments and display the MNE plot."""
    parser = argparse.ArgumentParser(description='Display MNE plot in a separate process')
    parser.add_argument('--filepath', type=str, required=True, help='Path to the EEG file')
    parser.add_argument('--start-time', type=float, help='Start time in seconds')
    parser.add_argument('--end-time', type=float, help='End time in seconds')
    parser.add_argument('--title', type=str, help='Window title')

    args = parser.parse_args()

    # Check if the file exists
    if not os.path.exists(args.filepath):
        print(f"Error: File not found at {args.filepath}", file=sys.stderr)
        return 1

    try:
        # Load the EEG file based on file extension
        file_ext = os.path.splitext(args.filepath)[1].lower()

        if file_ext == '.edf':
            raw = mne.io.read_raw_edf(args.filepath, preload=True)
        elif file_ext == '.bdf':
            raw = mne.io.read_raw_bdf(args.filepath, preload=True)
        elif file_ext == '.gdf':
            raw = mne.io.read_raw_gdf(args.filepath, preload=True)
        elif file_ext in ['.vhdr', '.vmrk', '.eeg']:
            raw = mne.io.read_raw_brainvision(args.filepath, preload=True)
        elif file_ext == '.cnt':
            raw = mne.io.read_raw_cnt(args.filepath, preload=True)
        elif file_ext == '.set':
            raw = mne.io.read_raw_eeglab(args.filepath, preload=True)
        else:
            # Try the generic reader as a fallback
            try:
                raw = mne.io.read_raw(args.filepath, preload=True)
            except Exception as e:
                print(f"Unsupported file format: {file_ext}. Error: {str(e)}", file=sys.stderr)
                return 1

        # Set the time window if specified
        if args.start_time is not None and args.end_time is not None:
            # Convert from seconds to points if needed
            start_idx = max(0, int(args.start_time * raw.info['sfreq']))
            end_idx = min(len(raw.times), int(args.end_time * raw.info['sfreq']))
            duration = (end_idx - start_idx) / raw.info['sfreq']

            # Create a plot with the specified time window
            fig = raw.plot(start=args.start_time, duration=duration,
                          scalings='auto', block=True, show=True)
        else:
            # Just show the whole EEG
            fig = raw.plot(scalings='auto', block=True, show=True)

        # Set the window title if provided
        if args.title and hasattr(fig, 'canvas') and hasattr(fig.canvas, 'manager'):
            fig.canvas.manager.set_window_title(args.title)

        return 0

    except Exception as e:
        print(f"Error displaying MNE plot: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())