# Preprocessing

The class `Preprocessor` will do basics preprocessing work on your data to get datasets ready for training.

## Feature engineering

When you need to compute a new feature using raw variables, you need to have a look on the feature_engineering_functions.py and feature_engineering_instructions.py files.

### feature_engineering_functions

The goal of this file is to build gather functions useful for the creation of new features.
These functions are usually applied when we read the data (data_read/data_reader.py), because for memory optimization you just just feed the preprocessor with the needed variables for the training and not the entire dataset.

### feature_engineering_instructions

This file is basically a dictionnary that will map a variable name to the function called to compute it.

## Example :

Let's assume we want to study chlorophyll-a concentration and we have a dataset containing both large and small phytoplankton concentrations.
The new feature `chl-a` is the sum of the two variables `small_phytoplankton_concentration` and `large_phytoplankton_concentration`.

To integrate this feature to our training, we can write the function `compute_chl_a` in the file `feature_engineering_functions` as the following :

```
def compute_chl_a(raw_data):
    large_phytoplankton_concentration = raw_data.variables['large_phytoplankton_concentration'][:,:,:].squeeze()
    small_phytoplankton_concentration = raw_data.variables['small_phytoplankton_concentration'][:,:,:].squeeze()
    return np.array(large_phytoplankton_concentration + small_phytoplankton_concentration)
```

Then in the `feature_engineering_instructions` file we can import this function and update the mapping dictionnary call `computation_instructions`

```
computation_instructions = {
    'chl-a': compute_chl_a,
    some_other_mapping...,
}
```

With that done, anytime you will add `chl-a` in the `variables` field in a config file, this function will be called and you will have the chlorophyll-a concentration resulting of the sum of `small_phytoplankton_concentration` and `large_phytoplankton_concentration`