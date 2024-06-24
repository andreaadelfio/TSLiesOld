# Anomaly Detection in the Fermi Anti-Coincidence Detector with Machine Learning techinques and Poisson-FOCuS triggering algorithm

This repository contains a series of modules and scripts used for a series of tasks:
- data preprocessing;
- data augmentation;
- model training;
- model evaluation;
- triggering.

The codes are organized in a modular way for easy importation and usage in the main scripts.

## Modules

1. [**config.py**](/modules/config.py): This module is used to define the configuration parameters for the project. The configuration parameters are defined as a series of constants.

2. [**makecountrates.py**](/modules/makecountrates.py): This module is used to get the count rates for the ACD data in `root` format.

3. [**catalogreader.py**](/modules/catalogreader.py): This module is used to read the ACD data and convert it into a `pandas.DataFrame`.

4. [**spacecraft.py**](/modules/spacecraft.py): This module is used to retrieve the spacecraft data and convert it into a `pandas.DataFrame`.

5. [**sunmonitor.py**](/modules/sunmonitor.py): This module is used to retrieve the sun monitor data from GOES and convert it into a `pandas.DataFrame`.

6. [**nn.py**](/modules/nn.py): This modules is used to create, train and validate a Machine Learning model.

7. [**trigger.py**](/modules/trigger.py): This module is used to find anomalies in the ACD data given the background model, with a triggering algorithm.

8. [**plotter.py**](/modules/plotter.py): This module contains some plotting functions adapted to the specific cases.

9. [**utils.py**](/modules/utils.py): This module continas some utility functions.

## Usage

To use these modules, import the required modules into your main script and call the necessary functions. You can find examples in the Scripts folder. Ensure that the configuration parameters in `config.py` are set correctly for your project and folders (see the `DIR` path).

## Scripts

This folder contains the main scripts that manage dataset preparation ([main_dataset.py](/scripts/main_dataset.py)), machine learning training ([main_ml.py](/scripts/main_ml.py)) and triggering algorithm separately ([main_trigger.py](/scripts/main_trigger.py)).

## Contributing

Contributions are welcome. Please open an issue to discuss your idea or submit a pull request with your changes.
