from math import ceil
import numpy as np
from .utils import standardize, remove_random_data

def normalization(n_years, data, n_var, new_data=False, data_stdev=None, data_mean=None):
    """ Temporary function just to normalize outside preprocessor object for user tutorial purpose. """
    # --- Standardization: step 1: find standardization coeff (on the training data set only!)
    preprocessed_data = data.copy()
    data_train=data.copy()
    T_y=[]
    for y in range(n_years):
        T = len(preprocessed_data[0][y])
        T_y = np.concatenate([T_y, [T]])
    T_y = T_y.astype(int)
    data_max = np.zeros(n_var)
    data_min = np.zeros(n_var)
    if not new_data:
        data_train_tot = [np.concatenate(data) for data in preprocessed_data]
        data_mean = np.zeros(n_var)
        data_stdev = np.zeros(n_var)
        for i in range(n_var):
            data_max[i] = np.max(data_train_tot[i])
            data_min[i] = np.min(data_train_tot[i])
            (_ ,data_mean[i], data_stdev[i]) = standardize(data_train_tot[i])

    # --- standardization: step 2: standardize
    data_train_norm = [np.empty([]) for y in range(n_years)]
    
    
    for y in range(n_years):
        if len(preprocessed_data[0][0].shape) == 1:
            p_tmp=1
        else:
            (_,p_tmp) = preprocessed_data[0][0].shape
        data_train_norm[y] = np.concatenate([
            np.reshape(standardize(data_train[i][y], 
                    data_mean[i], 
                    data_stdev[i]),(T_y[y],p_tmp) ) for i in range(n_var)], axis=1)

    data_train = np.array(data_train_norm,dtype=object)
    if np.shape(data_train)[0]==1:
        data_train = np.array(data_train_norm)
        data_train = data_train[0,:,:]
    if not new_data:
        return data_train, data_mean, data_stdev, data_max, data_min
    if new_data:
        return data_train

def normalize_new_data(data, data_stdev, data_mean, n_years, n_var):
    preprocessed_data = data.copy()
    data_train = data.copy()
    T_y=[]
    for y in range(n_years):
        T = len(preprocessed_data[0][y])
        T_y = np.concatenate([T_y, [T]])
    T_y = T_y.astype(int)

    data_train_norm = [np.empty([]) for y in range(n_years)]
    for y in range(n_years):
        if len(preprocessed_data[0][0].shape) == 1:
            p_tmp=1
        else:
            (_,p_tmp) = preprocessed_data[0][0].shape
        data_train_norm[y] = np.concatenate([
            np.reshape(standardize(data_train[i][y],
                    data_mean[i],
                    data_stdev[i]),(T_y[y],p_tmp) ) for i in range(n_var)], axis=1)

    data_train = np.array(data_train_norm,dtype=object)

    if np.shape(data_train)[0]==1:
        data_train = np.array(data_train_norm)
        data_train = data_train[0,:,:]

    return data_train


class Preprocessor:
    """
    This class will do all the preprocessing needed to get datasets ready for the training of a model.
    """
    def __init__(self,raw_data, var, years, n_years_train, test_year=0, shuffle: bool = False):
        self.raw_data = raw_data
        self.years =  years

        # Reshaping data if neededcd ..
        if len(self.raw_data[0][0].shape)==1:
            self._reshape()
            # self.T = raw_data[0][0].shape[0]
            # self.p = 1
        self.p = raw_data[0][0].shape[1]
        self.n_years = raw_data.shape[1]
        self.n_var = len(var)
        self.n_depth_levels = raw_data[0][0][0].shape[0] 
        self.has_splitted = False
        self.var = var
        self._shuffle(shuffle)
        self.test_year = test_year
        self.n_years_train = n_years_train
        

    def _standardize(self):
        # --- Standardization: step 1: find standardization coeff (on the training data set only!)
        T_y=[]
        for y in range(self.n_years):
            T = len(self.preprocessed_data[0][y])
            T_y = np.concatenate([T_y, [T]])
        T_y = T_y.astype(int)
        data_train_tot = [np.concatenate(data[:self.n_years_train]) for data in self.preprocessed_data]
        data_mean = np.zeros(self.n_var)
        data_stdev = np.zeros(self.n_var)
        data_max = np.zeros(self.n_var)
        data_min = np.zeros(self.n_var)
        for i in range(self.n_var):
            data_max[i] = np.max(data_train_tot[i])
            data_min[i] = np.min(data_train_tot[i])
            (_ ,data_mean[i], data_stdev[i]) = standardize(data_train_tot[i])
        self.data_mean = data_mean
        self.data_stdev = data_stdev
        self.data_max = data_max
        self.data_min = data_min
        # --- standardization: step 2: standardize
        data_train_norm = [np.empty([]) for y in range(self.n_years)]
        data_map_norm = [np.empty([]) for y in range(self.n_years)]
        
        
        for y in range(self.n_years):
            if len(self.preprocessed_data[0][0].shape) == 1:
                p_tmp=1
            else:
                (_,p_tmp) = self.preprocessed_data[0][0].shape
            data_train_norm[y] = np.concatenate([
                np.reshape(standardize(self.data_train[i][y], 
                        data_mean[i], 
                        data_stdev[i]),(T_y[y],p_tmp) ) for i in range(self.n_var)], axis=1)
            data_map_norm[y] = np.concatenate([
                np.reshape(standardize(self.data_map[i][y], 
                        data_mean[i], 
                        data_stdev[i]),(T_y[y],p_tmp) ) for i in range(self.n_var)], axis=1)
        
            
        print("------------------------------------------------")
        print("#################### SUMMARY ###################")
        print("------------------------------------------------")
        for i in range(self.n_var):
            print("DATA TYPE: ", 
                self.var[i],
                " MEAN: ", round(data_mean[i],2),
                " STDEV: ", round(data_stdev[i],2),
                " MIN: ", round(data_min[i],2),
                " MAX: ", round(data_max[i],2))
        self.data_train = np.array(data_train_norm,dtype=object)
        self.data_map = np.array(data_map_norm,dtype=object)
        

    def _shuffle(self,shuffle):
        """
        Shuffle the raw data on the year axis
        
        Arguments:
            shuffle {boolean} -- indicates whether or not you want to shuffle the data,
            default to false
        """
        if(shuffle):
            # We generate permutation according to number of years we have in the data
            perm = np.random.permutation(self.n_years)
            # We shuffle the raw_data
            self.raw_data = np.take(self.raw_data,perm, axis=1)
            # We keep track of the year permutation
            self.years = np.take(self.years, perm)

    def chooseyear(self,test_year):
        index = np.where((self.years==test_year))[0][0]
        self.chooseindex(index)

    def chooseindex(self, index):
        self.years = np.concatenate((np.delete(self.years,index,axis=0),self.years[index]))

        self.raw_data = self.raw_data.T
        self.raw_data = np.concatenate(
            (np.delete(self.raw_data,index,axis=0), self.raw_data[index, :]))
        self.raw_data = self.raw_data.T


    def train_test_split(self, standardize: bool = True):
        """
        Proceed to the train test split of the data        
        
        Returns:
            (input_train, input_test, map_train, map_test)
        """
        if standardize:
            self._standardize()

        if(not self.has_splitted):
            # We do not need to shuffle since it's done in the __init__
            self.input_test = self.data_train[self.n_years_train:]
            self.input_train = self.data_train[:self.n_years_train]
            self.map_test = self.data_map[self.n_years_train:]
            self.map_train = self.data_map[:self.n_years_train]
            # self.train = np.reshape(train,-1)
            # self.test = np.reshape(test,-1)
            self.has_splitted = True

        if(np.shape(self.input_train)[0]==1):
            return (np.reshape(self.input_train,(1,-1)).T,\
                    np.reshape(self.input_test,(1,-1)).T, \
                    np.reshape(self.map_train,(1,-1)).T,\
                    np.reshape(self.map_test,(1,-1)).T)
        
        return self.input_train.T, self.input_test.T, self.map_train.T, self.map_test.T

    def average_over_timestep(self, 
                              step_size) -> None:
        """
        Compute the average over a duration given by step_size and
        store it in the preprocessed_data attribute of the class.

        Example :
        ```
        x = [1,2,3,4,5,6]
        step_size = 3
        self.preprocessed_data <- [2,5]
        ```
        Keyword Arguments:
            step_size {int} -- step on which you compute the average,
            needs to be a strictly positive int.
        """
        if(isinstance(step_size,int)):
            if (step_size <=0):
                raise ValueError('The step size must be a strictly positive int')
            # We do nothing to prevent useless computing
            if step_size==1:
                self.preprocessed_data = self.raw_data
            
            if step_size>=2:
                new_data = []      
                for var in range(self.n_var):
                    time_series_var = []
                    for y in range(self.n_years):
                        T = self.raw_data[var][y].shape[0]
                        T_new = ceil(T/step_size)
                        averaged = np.zeros((T_new,self.p))
                        for t in range(0, T_new-1):
                            averaged[t] = np.nanmean(self.raw_data[var][y][t*step_size:(t+1)*step_size],axis=0)
                        averaged[-1] = np.nanmean(self.raw_data[var][y][(T_new-1)*step_size:], axis=0)
                        time_series_var.append(averaged)
                    new_data.append(time_series_var)
                self.preprocessed_data = np.array(new_data,dtype=object)
        else:
            raise TypeError('The step size must be a strictly positive int')

    def sparsify(self, p:list, map:bool, input:bool):
        """ Remove random data from the data to check out
        the robustness of the model on handling missing values.
        
        :param p: Probability list where each element i corresponds to
        the proportion of having non missing values for the variable at index i.
        :type p: list
        :param map: [description]
        :type map: bool
        :param input: [description]
        :type input: bool
        """

        sparse_data = [[]]*self.n_var
        for dtype in range(self.n_var):
            # We sparsify or not the training set e.g. the first n_years_train
            if input:
                sparse_data[dtype] = [remove_random_data(data, prob=p[dtype]) for data in self.preprocessed_data[dtype][:self.n_years]]
                self.data_train = sparse_data.copy()
            else:
                self.data_train = self.preprocessed_data
            # Same for the test set
            if map:
                sparse_data[dtype] = [remove_random_data(data, prob=p[dtype]) for data in self.preprocessed_data[dtype][self.n_years:]]
                self.data_map = sparse_data.copy()
            else:
                self.data_map = self.preprocessed_data
        
    def _reshape(self):
        (T,) = self.raw_data[0][0].shape
        self.raw_data[0][0] = np.reshape(self.raw_data[0][0],(T,1))

    def get_preprocessed_data(self):
        """
        Returns preprocessed data as an np.ndarray
        """
        return self.preprocessed_data
