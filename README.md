# Attenuatorium
Dealing with reflection files of (xray) diffraction data for merging when multiple runs of similar data have been performed using various attenuation levels.

## Overview
Attenuatorium is a tool for analyzing and merging X-ray diffraction data from HKL files. It allows combining data from different files by selecting portions based on intensity ranges, helping to create a more complete dataset by combining strong and weak reflection data.

## Features
- Load and visualize reflection data from HKL files
- Compare intensity and I/sigma distributions across multiple datasets
- Selectively merge data based on intensity thresholds
- High intensity data is taken from weak datasets (less attenuated)
- Low intensity data is taken from strong datasets (more attenuated)
- Medium intensity data can be statistically averaged
- Custom threshold controls for fine-tuning the merge process
- Export merged data to a new HKL file

## Installation
1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
### Command-line Interface
Run the original script:
```
python attenuatorium.py
```

### Graphical User Interface
Run the GUI application:
```
python attenuatorium_gui.py
```

1. Click "Load HKL Files" to select your input files
2. Use the threshold sliders to set intensity ranges for merging
3. Click "Merge Data" to create a combined dataset
4. Click "Save Merged Data" to export to a new HKL file

## License
See LICENSE file for details.
