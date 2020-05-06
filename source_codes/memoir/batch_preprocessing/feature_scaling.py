import json
import os

import numpy as np
from numpy import ndarray


class Normalize:

    '''
    Class to normalize the dataset using the values stored in `data_statistics.json` or entered by the user.

    Args:
        v_type (str): Type of the videos from which the frames are taken for training.
        color_space (str): Color space to be used for training.

    Keyword Args:
        min (list, ndarray): List or numpy array of minimum values of the three channels.
        max (list, ndarray): List or numpy array of maximum values of the three channels.
    '''

    def __init__(
        self,
        v_type: str = "Animated",
        color_space: str = "gray",
        **kwargs
    ):

        assert isinstance(
            v_type, str
        ), "'v_type' can only take string values."
        
        assert isinstance(
            color_space, str
        ), "'color_space' can only take string values."

        file_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(file_path)

        variables = ["min", "max"]

        for var in variables:

            if var in kwargs.keys():

                if isinstance(kwargs[var], list):

                    value = np.array(kwargs[var])
                    value = np.reshape(value, newshape=(1, 1, 1, 3))

                    setattr(self, var, value)

                if isinstance(kwargs[var], ndarray):

                    var_shape = kwargs[var].shape

                    if var_shape == (3,) or var_shape == (3, 1):

                        value = np.reshape(kwargs[var], newshape=(1, 1, 1, 3))

                        setattr(self, var, value)

                    elif var_shape == (1, 1, 1, 3):

                        setattr(self, var, kwargs[var])

                    else:

                        raise ValueError("Inconsistent shape of the {} keyword argument".format(var))
                        exit(0)

            else:

                with open("../data/data_statistics.json") as file:
                    data_stats = json.load(file)

                value = np.array(data_stats[v_type][color_space][var])
                value = np.reshape(value, newshape=(1, 1, 1, 3))

                setattr(self, var, value)

    def fit(self, batch: ndarray) -> ndarray:

        '''
        Fits the `Normalize` object to the entered batch of image.

        Args:
            batch (ndarray): Batch of image to normalize.

        Returns:
            Normalized batch of images.
        '''

        normalized_batch = (batch - self.min) / (self.max - self.min)

        return normalized_batch


class Standardize:

    '''
    Class to standardize the dataset using the values stored in `data_statistics.json` or entered by the user.

    Args:
        v_type (str): Type of the videos from which the frames are taken for training.
        color_space (str): Color space to be used for training.

    Keyword Args:
        mean (list, ndarray): List or numpy array of means of the three channels.
        std (list, ndarray): List or numpy array of standard deviations of the three channels.
    '''

    def __init__(
        self,
        v_type: str = "Animated",
        color_space: str = "gray",
        **kwargs
    ):

        assert isinstance(
            v_type, str
        ), "'v_type' can only take string values."
        
        assert isinstance(
            color_space, str
        ), "'color_space' can only take string values."

        file_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(file_path)

        variables = ["mean", "std"]

        for var in variables:

            if var in kwargs.keys():

                if isinstance(kwargs[var], list):

                    value = np.array(kwargs[var])
                    value = np.reshape(value, newshape=(1, 1, 1, 3))

                    setattr(self, var, value)

                if isinstance(kwargs[var], ndarray):

                    var_shape = kwargs[var].shape

                    if var_shape == (3,) or var_shape == (3, 1):

                        value = np.reshape(kwargs[var], newshape=(1, 1, 1, 3))

                        setattr(self, var, value)

                    elif var_shape == (1, 1, 1, 3):

                        setattr(self, var, kwargs[var])

                    else:

                        raise ValueError("Inconsistent shape of the {} keyword argument".format(var))
                        exit(0)

    def fit(self, batch: ndarray) -> ndarray:

        '''
        Fits the `Standardize` object to the entered batch of image.

        Args:
            batch (ndarray): Batch of image to standardize.

        Returns:
            Standardized batch of images.
        '''

        standardized_batch = (batch - self.mean) / self.std

        return standardized_batch


class Mean_Normalize:

    '''
    Class to mean-normalize the dataset using the values stored in `data_statistics.json` or entered by the user.

    Args:
        v_type (str): Type of the videos from which the frames are taken for training.
        color_space (str): Color space to be used for training.

    Keyword Args:
        mean (list, ndarray): List or numpy array of means of the three channels.
        min (list, ndarray): List or numpy array of minimum values of the three channels.
        max (list, ndarray): List or numpy array of maximum values of the three channels.
    '''

    def __init__(
        self,
        v_type: str = "Animated",
        color_space: str = "gray",
        **kwargs
    ):

        assert isinstance(
            v_type, str
        ), "'v_type' can only take string values."
        
        assert isinstance(
            color_space, str
        ), "'color_space' can only take string values."

        file_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(file_path)

        variables = ["mean", "min", "max"]

        for var in variables:

            if var in kwargs.keys():

                if isinstance(kwargs[var], list):

                    value = np.array(kwargs[var])
                    value = np.reshape(value, newshape=(1, 1, 1, 3))

                    setattr(self, var, value)

                if isinstance(kwargs[var], ndarray):

                    var_shape = kwargs[var].shape

                    if var_shape == (3,) or var_shape == (3, 1):

                        value = np.reshape(kwargs[var], newshape=(1, 1, 1, 3))

                        setattr(self, var, value)

                    elif var_shape == (1, 1, 1, 3):

                        setattr(self, var, kwargs[var])

                    else:

                        raise ValueError("Inconsistent shape of the {} keyword argument".format(var))
                        exit(0)

    def fit(self, batch: ndarray) -> ndarray:

        '''
        Fits the `Mean-Normalize` object to the entered batch of image.

        Args:
            batch (ndarray): Batch of image to mean-normalize.

        Returns:
            Mean-Normalized batch of images.
        '''

        mean_normalized_batch = (batch - self.mean) / (self.max - self.min)

        return mean_normalized_batch


class Unit_Vector:

    '''
    Class to unit-vectorize the dataset using the values stored in `data_statistics.json` or entered by the user. Each image channel will be converted to a feature vector of unit length.
    '''

    def __init__(self):
        pass

    def fit(self, batch: ndarray, factor: int = 1) -> ndarray:

        '''
        Fits the `Unit_Vector` object to the entered batch of image.

        Args:
            batch (ndarray): Batch of image to unit-vectorize.
            factor (int, float): If not unit, the length of the feature vector of a channel. Default: 1

        Returns:
            Unit-vectorized batch of images.
        '''

        assert isinstance(
            factor, int
        ), "'factor' can take only integer values."
        
        assert isinstance(
            batch, ndarray
        ), "'batch' must be a numpy array."
        
        assert len(batch.shape) == 4, "Inconsistent batch shape. Must be 4 dimensional."

        squared = np.square(batch, dtype=np.uint8)
        squared_sum = np.sqrt(np.sum(squared, axis=(1, 2, 3), keepdims=True))

        unit_batch = (batch / squared_sum) * factor

        return unit_batch
