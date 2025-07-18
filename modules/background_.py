'''
DEPRECATED MODULE - DO NOT USE

This module has been deprecated. All classes have been moved to individual files
in the modules/background/ directory:

- MedianKNeighborsRegressor -> modules/background/knnpredictors.py
- MultiMedianKNeighborsRegressor -> modules/background/knnpredictors.py
- MultiMeanKNeighborsRegressor -> modules/background/knnpredictors.py

Please update your imports to use the new module structure:
    from modules.background.knnpredictors import MedianKNeighborsRegressor
    
This file will be removed in a future version.
'''
import warnings

warnings.warn(
    "modules.background_ is deprecated. Use modules.background.knnpredictors instead.",
    DeprecationWarning,
    stacklevel=2
)

# Legacy imports for backward compatibility (will be removed)
try:
    from modules.background.knnpredictors import (
        MedianKNeighborsRegressor,
        MultiMedianKNeighborsRegressor, 
        MultiMeanKNeighborsRegressor
    )
except ImportError:
    pass
