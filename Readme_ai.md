# EEGAnalyzer - Scientific EEG Analysis Tool

A comprehensive Python package for analyzing EEG data with customizable metrics and visualization capabilities. This
tool provides a complete pipeline from raw EEG data processing to advanced analysis and visualization.

## Overview

EEGAnalyzer is designed to bring together powerful Python libraries for biomedical timeseries analysis, including MNE
for EEG processing, pandas, sqlite, and SQLAlchemy for data handling, and various signal processing libraries. The tool offers a modular
architecture that makes it easy to incorporate custom analysis functions and extend functionality.

## Features

### Core Analysis Capabilities

- **Signal Processing**: Comprehensive EEG preprocessing with filtering, artifact removal, and channel interpolation
- **Metric Analysis**: Customizable metrics including chaotic dynamics, complexity assessment, fractal properties, and
  entropy calculations
- **Flexible Data Support**: Works with various EEG file formats (EDF, BDF, GDF, BrainVision, CNT, EEGLAB)
- **Batch Processing**: Process entire folder structures with subfolder support

### Modular Architecture

- **Extensible Design**: Easy integration of custom analysis functions
- **Configuration-Based**: YAML configuration files for flexible parameter control
- **Database Integration**: SQLite database for efficient metric storage and comparison
- **Command-Line Interface**: Simple CLI for automated processing workflows

### Visualization & Analysis

- **CLI**: Command-line interface for Metric analysis and preprocessing (preprocessing still limited)
- **GUI Viewer**: Interactive visualization of computed metrics and original EEG data
- **MNE Integration**: Seamless plotting capabilities with MNE-Python
- **Export Options**: Standardized CSV output format for further analysis

## Installation

### Prerequisites

- Python 3.8+
- Additional system dependencies for visualization:
    - **QT**: For GUI functionality
        - Ubuntu:
          Follow [Stanford's QT installation guide](https://web.stanford.edu/dept/cs_edu/resources/qt/install-linux)
        - Arch: `sudo pacman -S qtcreator`
    - **Tk**: For additional GUI components
        - Ubuntu: `sudo apt-get -y install tk`
        - Arch: `sudo pacman -S tk`

### Installation Options

#### Option 1: From Source 
Recommended for research as it is more adaptable and kept up to date.

I highly recommend forking the project beforehand to ensure the pipeline or functions dont change during your research. 
When you create your fork, exchange the path to the forked repository in the command below. Remember to use SSH instead of HTTPS if you want to sync with your fork using an SSHKey.

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

#### Option 2: From PyPI
I try to keep it updated whenever a version changes. 
This is most useful to get first experience with the tool or use it in its default configuration.

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package
pip install eeganalyzer-scico
```

## Usage

### Command-Line Interface

#### EEG Analysis

``` bash
eeganalyzer --yaml_config <config_file> --logfile_path <log_file>
```

**Arguments:**

- `--yaml_config`: Path to YAML configuration file (required)
- `--logfile_path`: Path to log file (optional)

#### Visualization

``` bash
eegviewer --sql_path <database_path>
```

**Arguments:**

- `--sql_path`: Path to SQLite database file (required)

#### EEG Preprocessing

For interactive EEG preprocessing with visual feedback:

``` bash
python run_preprocessing_viewer.py
```

Launch the interactive preprocessing GUI that provides:
- Visual, step-by-step data processing
- Real-time visualization of preprocessing effects
- Interactive ICA component selection and artifact removal
- Support for multiple EEG file formats (EDF, BDF, GDF, BrainVision, CNT, EEGLAB, FIF)
- Save preprocessed data in various formats

### Python API

#### Standard Analysis

``` python
from src.eeganalyzer.core.processor import process_experiment
from src.eeganalyzer.utils.config import load_yaml_file

# Load configuration
config = load_yaml_file('config.yaml')

# Process experiments
process_experiment(config, 'results/analysis.log')
```

#### Programmatic Preprocessing

``` python
from eeganalyzer.preprocessing.eeg_preprocessing_pipeline import EEGPreprocessor

# Initialize and load data
preprocessor = EEGPreprocessor('path/to/eeg_file.edf')
preprocessor.categorize_channels()

# Apply preprocessing chain
preprocessor.apply_filter(l_freq=1.0, h_freq=40.0)  # Bandpass filter
preprocessor.resample_data(sfreq=250)               # Downsample
preprocessor.fit_ica()                              # ICA fitting
preprocessor.exclude_ica_components([0, 1, 2])     # Remove artifacts
preprocessor.interpolate_bad_channels()             # Fix bad channels

# Save results
preprocessor.save_preprocessed('clean_eeg.fif')
```

### Preprocessing Workflow

The preprocessing functionality offers both interactive and programmatic approaches:

**Key Preprocessing Operations:**
- **Filtering**: Highpass, lowpass, and bandpass filters
- **Resampling**: Adjust sampling frequency for analysis requirements
- **Channel Management**: Automatic bad channel detection and interpolation
- **ICA Analysis**: Independent Component Analysis for artifact removal
- **Montage Fitting**: Electrode positioning and coordinate systems
- **Artifact Detection**: Automated identification of flat channels and noise

The preprocessing module integrates seamlessly with the main analysis pipeline, allowing you to preprocess data interactively and then proceed with automated metric computation.

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
│   ├── example_config_eeg.yaml
│   ├── metrics.py
│   └── metric_analysis_example.py
├── tests/                   # Test suites (not implemented yet)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Configuration

The tool uses YAML configuration files to control processing parameters for the Analysis functionallity:

- **Signal Processing**: Filtering, sampling rates, montage settings
- **Epoching**: Start/stop times, duration, window overlap
- **File Processing**: Inclusion criteria, output directories
- **Metrics**: Selection of analysis functions to apply

## Quick Start Example

1. **Install the package** following the installation instructions above
2. **Prepare data structure**:
    1. Create a directory for the analysis
        ``` bash
           mkdir -p example/eeg
        ```
    2. **Download example data**: Get an EEG file (e.g., from Kaggle's SEINA dataset) and place it in the directory
       `example/eeg/`
3. **Preprocess the file**:
   1. using the gui
        ``` bash
           python run_preprocessing_viewer.py
        ```
      Here select the eeg you have just added to your eeg example folder using the File section at the top right.
4. **Run analysis**:
   1.   ``` bash
          eeganalyzer --yaml_config example/example_config_eeg.yaml --logfile_path example/test.log
        ```

5. **Visualize results**:
    1. ``` bash
           eegviewer example/EEGAnalyzer.sqlite
        ```

## Contributing

This project is open for contributions. The modular design makes it easy to add new analysis functions, file format
support, or visualization features.

## License

This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.

## Support

For issues, questions, or contributions, please visit the project repository or contact the maintainers.
_EEGAnalyzer was originally developed for Masters thesis research on chaotic dynamics analysis of EEG signals. It has
evolved into a comprehensive tool for the broader EEG analysis community._