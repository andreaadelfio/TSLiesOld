# ACDAnomalies

## What is it?
Anomaly Detection in the Fermi Anti-Coincidence Detector (ACD) with Machine Learning techinques and the Poisson-FOCuS triggering algorithm.

## Table of contents
- [Main Features](#main-features)
- [Installation and Dependencies](#installation-and-dependencies)
- [Modules](#modules)
- [Scripts](#scripts)
- [Usage](#usage)
- [Contibution](#contributing)
- [Contact](#contact)

## Main Features

This repository contains a series of modules and scripts used for a series of tasks:
- data preprocessing;
- data augmentation;
- model training;
- model evaluation;
- triggering.

## Installation and Dependencies

To install this repository, clone it from [ACDAnomalies](https://github.com/andreaadelfio/ACDAnomalies) and install the required packages:
```
git clone https://github.com/andreaadelfio/ACDAnomalies.git
cd ACDAnomalies
pip install -r requirements.txt
```
A ROOT installation is required to use the [ROOT python library](https://root.cern/manual/python/). Check the [website](https://root.cern/install/).

## Modules

The codes are organized in a modular way for easy importation and usage in the main scripts.
All modules can be found in the [**/modules**](/modules) folder.

1. [**config.py**](/modules/config.py): This module is used to define the configuration parameters for the project. The configuration parameters are defined as a series of constants.

2. [**makecountrates.py**](/modules/makecountrates.py): This module is used to get the count rates for the ACD data in [`root`](https://root.cern.ch/root/html600/notes/release-notes.html#ttreereader) format.

3. [**catalogreader.py**](/modules/catalogreader.py): This module is used to read the ACD data and convert it into a [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

4. [**spacecraft.py**](/modules/spacecraft.py): This module is used to retrieve the spacecraft data and convert it into a [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

5. [**sunmonitor.py**](/modules/sunmonitor.py): This module is used to retrieve the sun monitor data from GOES and convert it into a [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

6. [**nn.py**](/modules/nn.py): This modules is used to create, train and validate a Machine Learning model.

7. [**trigger.py**](/modules/trigger.py): This module is used to find anomalies in the ACD data given the background model, with a triggering algorithm.

8. [**plotter.py**](/modules/plotter.py): This module contains some plotting functions adapted to the specific cases.

9. [**utils.py**](/modules/utils.py): This module continas some utility functions.

## Scripts

This folder contains the main scripts that manage dataset preparation ([**main_dataset.py**](/scripts/main_dataset.py)), machine learning training ([**main_ml.py**](/scripts/main_ml.py)) and triggering algorithm separately ([**main_trigger.py**](/scripts/main_trigger.py)).

## Usage

To use these modules, import the required modules into your main script and call the necessary functions. You can find examples in the [**/scripts**](/scripts) folder. 

> [!TIP]
> Start from the scripts found in the [**/scripts**](/scripts) folder.

The **config.py** expects a certain directory structure, such as:
``` bash
ACDAnomalies
├── data
│   ├── anomalies
│   ├── inputs_outputs
│   ├── LAT_ACD
│   ├── model_nn
│   ├── solar
│   └── spacecraft
├── logs
├── modules
└── scripts
```

> [!IMPORTANT]
> Ensure that the configuration parameters in [**config.py**](/modules/config.py) are set correctly for your project and folders (see the [`DIR`](/modules/config.py#L9) variable describing the path to your ACDAnomalies, e.g. `DIR = /home/ACDAnomalies`).
> 

## Contributing

Contributions are welcome. Please open an issue to discuss your idea or submit a pull request with your changes.

### TO-DO:
```
- check cuFFT, cuDNN, cuBLAS
- check the articles in lectures
- check if __init__.py are actually necessary (https://sentry.io/answers/what-is-init-py-for-in-python/, https://martinxpn.medium.com/what-are-packages-in-python-and-what-is-the-role-of-init-py-files-82-100-days-of-python-325a992b2b13)
- improve the trigger plotting
- improve get_feature_importance
- separate get_feature_importance based on SHAP / lime use
- add catalog compilation
```

## Contact

If you need help on the project or you want to discuss about it, you can contact me at the following e-mails:
- <and.adelfio@gmail.com>
- <andrea.adelfio@pg.infn.it>

<hr>

[Go to Top](#table-of-contents)
