'''This package contains all the scripts that are used in the project.
   The scripts are used to perform various tasks like data preprocessing,
   data augmentation, model training, model evaluation, etc. 
   The scripts are organized in a modular way so that they can be
   easily imported and used in the main script. 

   The scripts are organized in the following way, here presented in order of
   utilization:

1. :mod:`config.py`:
This script is used to define the configuration parameters
for the project. The configuration parameters are defined as a series
of constants. 
      
2. :mod:`makecountrates.py`:
This script is used to calculate the count rates for the catalog data.

3. :mod:`catalogreader.py`:
This script is used to read the catalog data and convert it into a pandas dataframe. 

4. :mod:`spacecraft.py`: 
This script is used to retrieve the spacecraft data and 
convert it into a pandas dataframe.

5. :mod:`sunmonitor.py`:
This script is used to retrieve the sun monitor data from GOES and
convert it into a pandas dataframe.

6. :mod:`utils.py`:
This script is used to define utility functions that are used
in the project.

7. :mod:`nn.py`:
This script is used to define the neural network and knn models.

8. :mod:`plotter.py`:
This script is used to plot the results of the model.
'''