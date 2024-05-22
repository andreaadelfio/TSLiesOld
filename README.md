# Anomaly Detection in the Fermi Anti-Coincidence Detector with Machine Learning

This project contains a series of scripts used for data preprocessing, data augmentation, model training, and model evaluation. The scripts are organized in a modular way for easy importation and usage in the main script.

## Scripts

1. **config.py**: This script is used to define the configuration parameters for the project. The configuration parameters are defined as a series of constants.

2. **makecountrates.py**: This script is used to calculate the count rates for the catalog data.

3. **catalogreader.py**: This script is used to read the catalog data and convert it into a pandas dataframe.

4. **spacecraft.py**: This script is used to retrieve the spacecraft data and convert it into a pandas dataframe.

5. **sunmonitor.py**: This script is used to retrieve the sun monitor data from GOES and convert it into a pandas dataframe.

## Usage

To use these scripts, import the required script into your main script and call the necessary functions. Ensure that the configuration parameters in `config.py` are set correctly for your project and folders (see the `DIR` path).

## Contributing

Contributions are welcome. Please open an issue to discuss your idea or submit a pull request with your changes.
