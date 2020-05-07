import json
import os
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm

from memoir.batch_preprocessing import channels


class Std:

    '''
    Class to create an object with the methods to calculate standard deviation of the dataset.

    Args:
        v_type (str): Type of the videos to calculate the standard deviation for. Default: 'Animated'
        image_size (tuple): Size of the images to be used for the calculation of standard deviation. Default: (320, 240)
        color_space (str): Color space of the images to calculate the standard deviation for. Default: 'gray'

    Raises:
        ValueError: When the entered value of `v_type` or `color_space` is not present.
        ValueError: When the input value for `image_size` has inconsistent shape.
        TypeError: When the data type of the entered value of `v_type` or `color_space` does not match with the allowed ones.
    '''

    def __init__(self, v_type='Animated', image_size=(320, 240), color_space='gray'):
        self.v_type = v_type
        self.image_size = image_size
        self.color_space = color_space

        paths = self.file_reader(return_paths=True)

        if not isinstance(v_type, str):
            raise TypeError("Invalid data type of the entered value for v_type")
            exit(0)

        if v_type not in os.listdir(paths["frame_dir"]):
            raise ValueError("Entered value of v_type is not present in the dataset")
            exit(0)

        if not isinstance(color_space, str):
            raise TypeError("Invalid data type of the entered value for color_space")
            exit(0)

        if color_space not in channels.available_channels():
            raise ValueError("Entered value of color_space is not present in available color spaces.")
            exit(0)

        if len(self.image_size) != 2:
            raise ValueError("Image size must be a tuple of two entries for width and height.")
            exit(0)

    def standard_deviation(self, means=None, return_variance=False):

        '''
        Class method to calculate the standard deviation of the dataset.

        Args:
            means (tuple): Tuple of means of the three channels of the dataset images.
            return_variance (bool): Whether to return the variance of the dataset.

        Returns:
            A tuple with the standard deviaitons for the three channels of the dataset images.

        Raises:
            ValueError: When the entered value for `means` has inconsistent shape.
            TypeError: When the data type of `means` or `return_variance` does not match with the allowed ones.
        '''

        if len(means) != 3:
            raise ValueError("Entered value for \'means\' must be a tuple with three entries of mean for the three respective channels.")
            exit(0)

        if not isinstance(means, Union[list, tuple]):
            raise TypeError("Data type of the entered value for `means` does not match with either of the allowed data types i.e. `list` or `tuple`.")
            exit(0)

        if not isinstance(return_variance, bool):
            raise TypeError("Data type of the entered value for `return_variance` does not match with the allowed data type i.e. `bool`.")
            exit(0)

        channel_1_mean = means[0]
        channel_2_mean = means[1]
        channel_3_mean = means[2]

        samples = self.file_reader()
        
        N = len(samples)

        channel_1_Sum_std = channel_2_Sum_std = channel_3_Sum_std = 0.0

        print(
            "\nCalculating standard deviation of the \'{}\' dataset in \'{}\' color space...".format(
                self.v_type,
                self.color_space
            )
        )

        for ind in tqdm(iterable=range(N), desc="SD"):

            image = cv2.imread(samples[ind])
            image = cv2.resize(image, (self.image_size[0], self.image_size[1]))

            if self.color_space != "bgr":
                image = channels.change_channel(image, channel=self.color_space)

            if self.color_space == "gray":
                image = channels.change_channel(image, channel="bgr")

            channel_1 = np.reshape(image[:, :, 0], -1)
            channel_2 = np.reshape(image[:, :, 1], -1)
            channel_3 = np.reshape(image[:, :, 2], -1)

            channel_1_diffs = channel_1 - channel_1_mean
            channel_1_SumOfSquares = np.sum(channel_1_diffs ** 2)

            channel_2_diffs = channel_2 - channel_2_mean
            channel_2_SumOfSquares = np.sum(channel_2_diffs ** 2)

            channel_3_diffs = channel_3 - channel_3_mean
            channel_3_SumOfSquares = np.sum(channel_3_diffs ** 2)

            channel_1_Sum_std += (
                1 / (N * self.image_size[0] * self.image_size[1])
            ) * channel_1_SumOfSquares
            channel_2_Sum_std += (
                1 / (N * self.image_size[0] * self.image_size[1])
            ) * channel_2_SumOfSquares
            channel_3_Sum_std += (
                1 / (N * self.image_size[0] * self.image_size[1])
            ) * channel_3_SumOfSquares

        variance_channel_1 = channel_1_Sum_std
        variance_channel_2 = channel_2_Sum_std
        variance_channel_3 = channel_3_Sum_std

        channel_1_std = np.sqrt(channel_1_Sum_std)
        channel_2_std = np.sqrt(channel_2_Sum_std)
        channel_3_std = np.sqrt(channel_3_Sum_std)

        variances = tuple((variance_channel_1, variance_channel_2, variance_channel_3))
        stds = tuple((channel_1_std, channel_2_std, channel_3_std))

        if return_variance:

            return stds, variances

        else:

            return stds

    def file_reader(self, return_paths=False):
    
        file_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(file_path)

        with open("../../../paths.json", "r") as file:
            paths = json.load(file)

        if return_paths:
            return paths

        series = os.listdir(os.path.join(paths["frame_dir"], self.v_type))
        samples = []

        for srs in series:
            list_ = os.listdir(os.path.join(paths["frame_dir"], self.v_type, srs, "Frames"))
            samples += [
                os.path.join(paths["frame_dir"], self.v_type, srs, "Frames", i) for i in list_
            ]

        return samples
