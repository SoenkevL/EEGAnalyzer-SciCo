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

Utility functions and constants for the EEG Metrics Viewer.
"""

from typing import Dict, List, Tuple
import customtkinter as ctk

# Common metadata columns that are not EEG channels
METADATA_COLUMNS = ['eeg_id', 'label', 'startDataRecord', 'duration', 'metric']

# Aggregation colors and styles
AGGREGATION_COLORS: Dict[str, str] = {
    'mean': 'red',
    'std': 'purple',
    'median': 'green'
}

AGGREGATION_STYLES: Dict[str, str] = {
    'mean': '-',
    'std': '--',
    'median': '-.'
}

# Common 10-20 system channels (in alphabetical order)
COMMON_CHANNELS: List[str] = [
    'C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz',
    'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'Pz', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'
]

# Map of alternative channel names (old -> new and new -> old)
ALTERNATIVE_CHANNEL_NAMES: Dict[str, str] = {
    'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'
}

# Font configurations
def get_label_font() -> ctk.CTkFont:
    """Return a font for labels."""
    return ctk.CTkFont(size=12)

def get_dropdown_font() -> ctk.CTkFont:
    """Return a font for dropdowns and smaller text."""
    return ctk.CTkFont(size=11)

def get_button_font() -> ctk.CTkFont:
    """Return a font for buttons."""
    return ctk.CTkFont(size=14, weight="bold")