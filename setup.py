"""
Setup script for EEGAnalyzer package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eeganalyzer",
    version="0.1.0",
    author="SoenkevL",
    author_email="",
    description="A comprehensive tool for analyzing and visualizing EEG data with focus on customizable metric analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SoenkevL/EEGAnalyzer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "edgeofpy==0.0.1",
        "icecream==2.1.3",
        "customtkinter==5.2.2",
        "mat4py==0.6.0",
        "matplotlib==3.9.2",
        "mne==1.8.0",
        "multiprocesspandas==1.1.5",
        "neurokit2==0.2.10",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "PyYAML==6.0.2",
        "scipy==1.14.1",
        "SQLAlchemy>=2.0.40",
    ],
    entry_points={
        "console_scripts": [
            "eeganalyzer=eeganalyzer.cli.cli:main",
            "eegviewer=eeganalyzer.gui.run_metrics_viewer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "eeganalyzer": ["example/*.yaml"],
    },
)