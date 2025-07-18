"""
Domain-specific applications for the Time Series Anomaly Detection Framework

This package contains domain-specific implementations built on top of the 
generic time series framework:

- acd: ACD (Anti-Coincidence Detector) specific applications for Fermi mission data
"""

# Import ACD applications
try:
    from .acd import SunMonitor, SpacecraftOpener
except ImportError:
    # Graceful fallback if dependencies are missing
    SunMonitor = None
    SpacecraftOpener = None

__version__ = '1.0.0'
__author__ = 'Andrea Adelfio'
