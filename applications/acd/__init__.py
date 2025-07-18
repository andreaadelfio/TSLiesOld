"""
ACD-specific applications and utilities for TSLies Framework

This module contains ACD (Anti-Coincidence Detector) specific implementations:
- solar: GOES solar activity monitoring (SunMonitor)
- spacecraft: Fermi spacecraft data handling (SpacecraftOpener)
- catalogs: Astronomical event catalog management
- main_*: Example scripts demonstrating TSLies usage for ACD data

These are domain-specific applications built on top of the generic TSLies framework.
"""

try:
    from .solar import SunMonitor
except ImportError:
    SunMonitor = None

try:
    from .spacecraft import SpacecraftOpener  
except ImportError:
    SpacecraftOpener = None

try:
    from .catalogs import *  # Import what's available in catalogs
except ImportError:
    pass

# Main scripts are available as modules but not auto-imported
# They can be run as: python -m applications.acd.main_ml

__version__ = '1.0.0'
__author__ = 'Andrea Adelfio'
