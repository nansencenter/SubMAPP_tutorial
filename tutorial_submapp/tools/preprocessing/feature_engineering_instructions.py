from .feature_engineering_functions import *

# This dictionnary maps a variable name to the function 
# that computes its value in the feature_engineering.py file

computation_instructions = {
    'ws10': compute_ws10,
    'chl-a': compute_chl_a
}