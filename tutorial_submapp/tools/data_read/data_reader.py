import numpy as np
from netCDF4 import Dataset
from .tools_read_data import unix_time_seconds
import matplotlib.dates as dates
# from .tools_read_data import mean_depth
from ..preprocessing.feature_engineering_functions import *
from ..preprocessing.feature_engineering_instructions import computation_instructions

def read_data(path: str,
              train_on: str,
              variable_list: list,
              starting_date: str,
              ending_date: str,
              depth_min: int = -100,
              depth_max: int = 0) -> np.ndarray:
    """
    Return a np.ndarray containing the raw data to be processed
    
    Arguments:
        path (str) -- path to the data file
        train_on (str) -- 'surface', 'depth', 'both' data you retrieve from the file
        variable_list (List[str]) -- variables to retrieve from the dataset
        starting_date (str): -- Starting date for the time series, need to be in the 
        following format : YYYY-MM-DD
        ending_date (str): -- Ending date of the time series, need to be in the
        following format : YYYY-MM-DD
        depth_min (int) -- used to filter the data, only values over this minimum depth
        will be taken. Default to -100m.
    """
    # Parse the starting date passed in argument to get the time origin
    splitted_starting_date = starting_date.split('-')
    initial_year = int(splitted_starting_date[0])
    initial_month = int(splitted_starting_date[1])
    initial_day = int(splitted_starting_date[2])

    sse0 = unix_time_seconds(initial_year,initial_month,initial_day)       # gotm time origin

    # Parse the ending date
    splitted_end_date = ending_date.split('-')
    ending_year = int(splitted_end_date[0])
    ending_month = int(splitted_end_date[1])
    ending_day = int(splitted_end_date[2])

    ending_date = unix_time_seconds(ending_year,ending_month,ending_day)

    # Number of years between ending and starting date
    nb_years = ending_year - initial_year

    #################### Read netcdf file
    if path.endswith('.nc') or path.endswith('.cdf'):
        # Read datafile
        nc_fid = Dataset(path, mode='r') 
        time = nc_fid.variables['time'][:]
        time_epoch = time + sse0                                  # numpy array of epoch time
        date_axis  = dates.epoch2num(time_epoch)
        variables_arranged_by_year_list = None
        if(train_on=='surface'):
            variables_arranged_by_year_list=[
                [np.empty([]) for y in range(nb_years)] for p in range(len(variable_list))
            ]
            #extract data
            variable_dict = {}
            for variable in variable_list:
                # We check if the variable is in the raw data
                if variable in nc_fid.variables:
                    variable_dict[variable] = nc_fid.variables[variable][:,:,:].squeeze()
                else:
                    variable_dict[variable] = computation_instructions[variable](nc_fid)
            nc_fid.close()
            for y in range(nb_years):

                #--- extract date axis and data for the specified date range
                current_starting_year = initial_year+y
                current_ending_year = current_starting_year+1

                mpdate1 = dates.epoch2num(unix_time_seconds(current_starting_year,1,1)) # Matplotlib dates
                if current_ending_year == ending_year:
                    mpdate2 = dates.epoch2num(unix_time_seconds(ending_year,ending_month,ending_day)) # Matplotlib dates
                else:
                    mpdate2 = dates.epoch2num(unix_time_seconds(current_ending_year,1,1)) # Matplotlib dates

                idx_range = np.where((date_axis >= mpdate1) & (date_axis < mpdate2)) # index search
                for i,variable_name in enumerate(variable_list):
                    variables_arranged_by_year_list[i][y] = variable_dict[variable_name][idx_range].squeeze()
        
        elif(train_on=='profile'):
            depth = nc_fid.variables['z'][1,:,:,:].squeeze()
            p_data = depth.size
            depth = depth[depth > depth_min]
            p = depth.size
            #extract data, we have one more dimension here compared to surface data
            variable_dict = {}
            variable_list = variable_list.copy()
            # variable_list += 'z'
            
            for variable in variable_list:
                # We check if the variable is in the raw data
                if variable in nc_fid.variables:
                    shape = nc_fid.variables[variable].shape
                    if(shape[1]!=1):
                        variable_dict[variable] = nc_fid.variables[variable][:,:,:,:].squeeze()
                    else:
                        variable_dict[variable] = nc_fid.variables[variable][:,:,:].squeeze()
                else:
                    variable_dict[variable] = computation_instructions[variable](nc_fid)
            nc_fid.close()
            variables_arranged_by_year_list=[
                [np.empty([]) for y in range(nb_years)] for p in range(len(variable_list))
            ]
            for y in range(nb_years):
                #--- extract date axis and data for the specified date range
                current_starting_year = initial_year+y
                current_ending_year = current_starting_year+1

                mpdate1 = dates.epoch2num(unix_time_seconds(current_starting_year,1,1)) # Matplotlib dates
                if current_ending_year == ending_year:
                    mpdate2 = dates.epoch2num(unix_time_seconds(ending_year,ending_month,ending_day)) # Matplotlib dates
                else:
                    mpdate2 = dates.epoch2num(unix_time_seconds(current_ending_year,1,1)) # Matplotlib dates

                idx_range = np.where((date_axis >= mpdate1) & (date_axis < mpdate2)) # index search
                for i,variable_name in enumerate(variable_list):
                    variables_arranged_by_year_list[i][y] = variable_dict[variable_name][idx_range].squeeze()[:,p_data-p:]
        #TODO adapt previous code for this case
        elif(train_on=='both'):
            raise NotImplementedError
        
        array = np.array(variables_arranged_by_year_list,dtype=object)
        return array
    
    else:
        print('This type of data file is not readable yet by this function')
        raise NotImplementedError
