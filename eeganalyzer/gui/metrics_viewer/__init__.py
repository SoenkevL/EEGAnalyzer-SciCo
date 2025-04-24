"""
EEG Metrics Viewer Package

This package provides a GUI for visualizing metrics computed from EEG data
and stored in the EEGAnalyzer.sqlite database.
"""

from .app import App
from .database_handler import DatabaseHandler
from .plot_frame import MetricsPlotFrame
from .selection_frame import SelectionFrame

__all__ = ['App', 'DatabaseHandler', 'MetricsPlotFrame', 'SelectionFrame']