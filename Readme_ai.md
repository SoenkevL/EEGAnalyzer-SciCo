# EEGAnalyzer - Scientific EEG Analysis Tool
# TODO: Check the content of this file carefully and edit it as it is ai generated and not complete nor completely right

A comprehensive Python package for analyzing EEG data with customizable metrics and visualization capabilities. This tool provides a complete pipeline from raw EEG data processing to advanced analysis and visualization.

## Overview

EEGAnalyzer is designed to bring together powerful Python libraries for biomedical timeseries analysis, including MNE for EEG processing, pandas for data handling, and various signal processing libraries. The tool offers a modular architecture that makes it easy to incorporate custom analysis functions and extend functionality.

## Features

### Core Analysis Capabilities
- **Signal Processing**: Comprehensive EEG preprocessing with filtering, artifact removal, and channel interpolation
- **Metric Analysis**: Customizable metrics including chaotic dynamics, complexity assessment, fractal properties, and entropy calculations
- **Flexible Data Support**: Works with various EEG file formats (EDF, BDF, GDF, BrainVision, CNT, EEGLAB)
- **Batch Processing**: Process entire folder structures with subfolder support

### Modular Architecture
- **Extensible Design**: Easy integration of custom analysis functions
- **Configuration-Based**: YAML configuration files for flexible parameter control
- **Database Integration**: SQLite database for efficient metric storage and comparison
- **Command-Line Interface**: Simple CLI for automated processing workflows

### Visualization & Analysis
- **GUI Viewer**: Interactive visualization of computed metrics and original EEG data
- **MNE Integration**: Seamless plotting capabilities with MNE-Python
- **Export Options**: Standardized CSV output format for further analysis

## Installation

### Prerequisites
- Python 3.8+
- Additional system dependencies for visualization:
    - **QT**: For GUI functionality
        - Ubuntu: Follow [Stanford's QT installation guide](https://web.stanford.edu/dept/cs_edu/resources/qt/install-linux)
        - Arch: `sudo pacman -S qtcreator`
    - **Tk**: For additional GUI components
        - Ubuntu: `sudo apt-get -y install tk`
        - Arch: `sudo pacman -S tk`

### Installation Options

#### Option 1: From PyPI (Recommended)
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package
pip install eeganalyzer-scico
```

#### Option 2: From Source
``` bash
# Clone repository
git clone https://github.com/SoenkevL/EEGAnalyzer.git
cd EEGAnalyzer

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .
```
## Usage
### Command-Line Interface
#### EEG Analysis
``` bash
eeganalyzer --yaml_config <config_file> --logfile_path <log_file>
```
**Arguments:**
- : Path to YAML configuration file (required) `--yaml_config`
- : Path to log file (optional) `--logfile_path`

#### Visualization
``` bash
eegviewer --sql_path <database_path>
```
**Arguments:**
- : Path to SQLite database file (required) `--sql_path`

### Python API
``` python
from src.eeganalyzer.core.processor import process_experiment
from src.eeganalyzer.utils.config import load_yaml_file

# Load configuration
config = load_yaml_file('config.yaml')

# Process experiments
process_experiment(config, 'results/analysis.log')
```
## Project Structure
``` 
EEGAnalyzer/
├── src/
│   ├── eeganalyzer/           # Main package
│   │   ├── cli/              # Command-line interface
│   │   ├── core/             # Core processing logic
│   │   ├── preprocessing/    # EEG preprocessing modules
│   │   └── utils/            # Utility functions
│   └── gui/                  # GUI components
├── example/                  # Example configurations and data
│   ├── eeg/                 # Example EEG data directory
│   ├── metrics/             # Output metrics directory
│   ├── example_config_eeg.yaml
│   ├── metrics.py
│   └── metric_analysis_example.py
├── tests/                   # Test suites
├── requirements.txt
├── pyproject.toml
└── README.md

```
## Configuration
The tool uses YAML configuration files to control processing parameters:
- **Signal Processing**: Filtering, sampling rates, montage settings
- **Epoching**: Start/stop times, duration, window overlap
- **File Processing**: Inclusion criteria, output directories
- **Metrics**: Selection of analysis functions to apply

## Quick Start Example
1. **Install the package** following the installation instructions above
2. **Prepare data structure**:
``` bash
   mkdir -p example/eeg
```
1. **Download example data**: Get an EEG file (e.g., from Kaggle's SEINA dataset) and place it in the directory `example/eeg/`
2. **Run analysis**:
``` bash
   eeganalyzer --yaml_config example/example_config_eeg.yaml --logfile_path example/test.log
```
1. **Visualize results**:
``` bash
   eegviewer example/EEGAnalyzer.sqlite
```
## Contributing
This project is open for contributions. The modular design makes it easy to add new analysis functions, file format support, or visualization features.
## License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.
## Support
For issues, questions, or contributions, please visit the project repository or contact the maintainers.
_EEGAnalyzer was originally developed for Masters thesis research on chaotic dynamics analysis of EEG signals. It has evolved into a comprehensive tool for the broader EEG analysis community._
