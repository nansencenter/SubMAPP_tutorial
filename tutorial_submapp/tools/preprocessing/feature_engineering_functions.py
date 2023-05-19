import numpy as np

"""
This module groups all the functions needed to compute features using
other variables.

All of these functions needs to take as input the raw data e.g. a
netCDF4.Dataset object and should return a np.ndarray.
"""

def compute_ws10(raw_data):
    """
    Compute wind speed using wind speed on x and y
    
    Arguments:
        raw_data (netCDF4._netCDF4.Dataset) -- netCDF4 file
    
    Returns:
        (np.ndarray) -- wind speed
    """
    u10 = raw_data.variables['u10'][:,:,:].squeeze()
    v10 = raw_data.variables['v10'][:,:,:].squeeze()
    return np.sqrt(np.square(u10)+np.square(v10))


def compute_chl_a(raw_data):
    """
    Computes chlorophyll-a concentration using large and small phytoplankton chlorophyll-a
    
    Arguments:
        raw_data (netCDF4._netCDF4.Dataset) -- netCDF4 file
    Returns:
        (np.ndarray) -- chlorophyll-a vector
    """
    dia_chl = raw_data.variables['ECO_diachl'][:,:,:].squeeze()
    fla_chl = raw_data.variables['ECO_flachl'][:,:,:].squeeze()
    return np.array(dia_chl + fla_chl)