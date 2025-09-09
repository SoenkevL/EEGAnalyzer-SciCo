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
        - mac: `brew install qt`
    - **Tk**: For additional GUI components
        - Ubuntu: `sudo apt-get -y install tk`
        - Arch: `sudo pacman -S tk`
        - mac: `brew install python-tk`

### Installation
I highly recommend forking the project beforehand to ensure the pipeline or functions dont change during your research. 
When you create your fork, exchange the path to the forked repository in the command below. Remember to use SSH instead of HTTPS if you want to sync with your fork using an SSHKey.

``` bash
# Clone repository
git clone https://github.com/SoenkevL/EEGAnalyzer.git
cd EEGAnalyzer

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the pipeline itself in development mode
pip install -e .
```

## Usage

### Command-Line Interface

#### EEG Preprocessing and visualization

For interactive EEG preprocessing with visual feedback:
(not recommended for server use)

``` bash
preprocessor
```

Launch the interactive preprocessing GUI that provides:
- Visual, step-by-step data processing
- Real-time visualization of preprocessing effects
- Interactive ICA component selection and artifact removal
- Support for multiple EEG file formats (EDF, BDF, GDF, BrainVision, CNT, EEGLAB, FIF)
- Save preprocessed data in various formats

**src/custom_files**

This folder contains custom files that are used by the pipeline.
1. channel_ident_patterns.py: Is used by the preprocessing module to identify channel types

#### EEG Analysis
The eeganalysis is orchestrated at three main entry points for the user
**.env**

The pipeline makes use of environment variables for configuration. You can set these variables in a `.env` file
in the root directory of the project.
Here one should specify the following:
CONFIG_PATH: Path to the configuration file
MAX_PROCESSORS: Maximum number of parallel processes (will be set to number of available cores by default and as max value)
LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

**src/custom_files**

This folder contains custom files that are used by the pipeline.
1. metrics.py: Contains the metrics that are used by the pipeline.
2. pipeline_preprocessing.py: Contains the preprocessing pipeline that is used by the pipeline.

**config.yaml**
A config file controls hyperparameters for things like preprocessing, file paths and metric calculation. An example can be found in the example folder

Once everything is set you can just run 
``` bash
eeganalyzer
```
from the commmand line. Make sure that if you use relative filepaths you exectue this from the right directory

#### Visualization
This is still very rudimentary
``` bash
metricviewer --sql_path <database_path>
```

**Arguments:**

- `--sql_path`: Path to SQLite database file (required)

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
│   ├── custom_files/         # Custom files for the pipeline intended for alteration 
│   ├── eeganalyzer/          # Main package
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
