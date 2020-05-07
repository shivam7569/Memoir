import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from memoir.batch_preprocessing import channels


class Mean():

    '''
    Class to create an object with the methods to calculate arithmetic mean, geometric mean and harmonic mean of the dataset.

    Args:
        v_type (str): Type of the videos to calculate the mean for. Default: 'Animated'
        image_size (tuple): Size of the images to be used for the calculation of mean. Default: (320, 240)
        color_space (str): Color space of the images to calculate the mean for. Default: 'gray'

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
            raise ValueError("Entered value of color_space is not present in available color spaces")
            exit(0)

        if len(self.image_size) != 2:
            raise ValueError("Image size must be a tuple of two entries for width and height.")
            exit(0)

    def arithmetic_mean(self) -> tuple:

        '''
        Class method to calculate arithmetic mean of the dataset. 

        Returns:
            Arithmetic means for the three channels of the dataset images as a tuple.
        '''

        samples = self.file_reader()
        
        N = len(samples)

        channel_1_am = channel_2_am = channel_3_am = 0.0

        print("\nCalculating arithmetic mean of the \'{}\' dataset in \'{}\' color space...".format(
                self.v_type,
                self.color_space
            )
        )

        for ind in tqdm(iterable=range(N),
                        desc="AM"
        ):

            image = cv2.imread(samples[ind])
            image = cv2.resize(image, (self.image_size[0], self.image_size[1]))

            if self.color_space != "bgr":
                image = channels.change_channel(image, channel=self.color_space)

            if self.color_space == "gray":
                image = channels.change_channel(image, channel="bgr")

            channel_1 = np.reshape(image[:, :, 0], -1)
            channel_2 = np.reshape(image[:, :, 1], -1)
            channel_3 = np.reshape(image[:, :, 2], -1)

            channel_1_mean = channel_1.mean()
            channel_2_mean = channel_2.mean()
            channel_3_mean = channel_3.mean()

            channel_1_am += channel_1_mean
            channel_2_am += channel_2_mean
            channel_3_am += channel_3_mean

        channel_1_mean = channel_1_am / N
        channel_2_mean = channel_2_am / N
        channel_3_mean = channel_3_am / N

        arith_mean = tuple((channel_1_mean, channel_2_mean, channel_3_mean))

        return arith_mean

    def geometric_mean(self) -> tuple:

        '''
        Class method to calculate geometric mean of the dataset. 

        Returns:
            Geometric means for the three channels of the dataset images as a tuple.
        '''

        samples = self.file_reader()
        
        N = len(samples)

        channel_1_gm = channel_2_gm = channel_3_gm = 0.0

        print("\nCalculating geometric mean of the \'{}\' dataset in \'{}\' color space...".format(
                self.v_type,
                self.color_space
            )
        )

        for ind in tqdm(iterable=range(N),
                        desc="GM"
        ):

            image = cv2.imread(samples[ind])
            image = cv2.resize(image, (self.image_size[0], self.image_size[1]))

            if self.color_space != "bgr":
                image = channels.change_channel(image, channel=self.color_space)

            if self.color_space == "gray":
                image = channels.change_channel(image, channel="bgr")

            channel_1 = np.reshape(image[:, :, 0], -1)
            channel_2 = np.reshape(image[:, :, 1], -1)
            channel_3 = np.reshape(image[:, :, 2], -1)

            channel_1 = np.array(channel_1, dtype=np.float)
            channel_2 = np.array(channel_2, dtype=np.float)
            channel_3 = np.array(channel_3, dtype=np.float)

            channel_1_mean = (np.sum(np.log(channel_1 + 1))) / channel_1.shape[0]
            channel_2_mean = (np.sum(np.log(channel_2 + 1))) / channel_2.shape[0]
            channel_3_mean = (np.sum(np.log(channel_3 + 1))) / channel_3.shape[0]

            channel_1_gm += channel_1_mean
            channel_2_gm += channel_2_mean
            channel_3_gm += channel_3_mean

        channel_1_mean = np.exp(channel_1_gm / N)
        channel_2_mean = np.exp(channel_2_gm / N)
        channel_3_mean = np.exp(channel_3_gm / N)

        geo_mean = tuple((channel_1_mean, channel_2_mean, channel_3_mean))

        return geo_mean

    def harmonic_mean(self) -> tuple:

        '''
        Class method to calculate harmonic mean of the dataset. 
    
        Returns:
            Harmonic means for the three channels of the dataset images as a tuple.
        '''

        samples = self.file_reader()
        
        N = len(samples)

        channel_1_hm = channel_2_hm = channel_3_hm = 0.0

        print("\nCalculating harmonic mean of the \'{}\' dataset in \'{}\' color space...".format(
                self.v_type,
                self.color_space
            )
        )

        for ind in tqdm(iterable=range(N),
                        desc="HM"
        ):

            image = cv2.imread(samples[ind])
            image = cv2.resize(image, (self.image_size[0], self.image_size[1]))

            if self.color_space != "bgr":
                image = channels.change_channel(image, channel=self.color_space)

            if self.color_space == "gray":
                image = channels.change_channel(image, channel="bgr")

            channel_1 = np.reshape(image[:, :, 0], -1)
            channel_2 = np.reshape(image[:, :, 1], -1)
            channel_3 = np.reshape(image[:, :, 2], -1)

            channel_1 = np.array(channel_1, dtype=np.float)
            channel_2 = np.array(channel_2, dtype=np.float)
            channel_3 = np.array(channel_3, dtype=np.float)

            channel_1_mean = (np.sum(np.reciprocal(channel_1 + 1))) / channel_1.shape[0]
            channel_2_mean = (np.sum(np.reciprocal(channel_2 + 1))) / channel_2.shape[0]
            channel_3_mean = (np.sum(np.reciprocal(channel_3 + 1))) / channel_3.shape[0]

            channel_1_hm += channel_1_mean
            channel_2_hm += channel_2_mean
            channel_3_hm += channel_3_mean

        channel_1_mean = N / channel_1_hm
        channel_2_mean = N / channel_2_hm
        channel_3_mean = N / channel_3_hm

        harmo_mean = tuple((channel_1_mean, channel_2_mean, channel_3_mean))

        return harmo_mean

    def am_gm_hm(self):

        '''
        Class method to calculate all the three means (arithmetic, geometric and harmonic) of the dataset in a single run. 
        
        Returns:
            Three tuples for the arithmetic, geometric and harmonic means for the three channels of the dataset images.
        '''

        samples = self.file_reader()
        
        N = len(samples)

        channel_1_am = channel_2_am = channel_3_am = 0.0
        channel_1_gm = channel_2_gm = channel_3_gm = 0.0
        channel_1_hm = channel_2_hm = channel_3_hm = 0.0

        print("\nCalculating arithmetic, geometric, and harmonic means of the \'{}\' dataset in \'{}\' color space...".format(
                self.v_type,
                self.color_space
            )
        )

        for ind in tqdm(iterable=range(N),
                        desc="AM_GM_HM"
        ):

            image = cv2.imread(samples[ind])
            image = cv2.resize(image, (self.image_size[0], self.image_size[1]))

            if self.color_space != "bgr":
                image = channels.change_channel(image, channel=self.color_space)

            if self.color_space == "gray":
                image = channels.change_channel(image, channel="bgr")

            channel_1 = np.reshape(image[:, :, 0], -1)
            channel_2 = np.reshape(image[:, :, 1], -1)
            channel_3 = np.reshape(image[:, :, 2], -1)

            channel_1 = np.array(channel_1, dtype=np.float)
            channel_2 = np.array(channel_2, dtype=np.float)
            channel_3 = np.array(channel_3, dtype=np.float)

            ##### AM #####
            channel_1_mean_am = channel_1.mean()
            channel_2_mean_am = channel_2.mean()
            channel_3_mean_am = channel_3.mean()

            channel_1_am += channel_1_mean_am
            channel_2_am += channel_2_mean_am
            channel_3_am += channel_3_mean_am

            ##### GM #####
            channel_1_mean_gm = (np.sum(np.log(channel_1 + 1))) / channel_1.shape[0]
            channel_2_mean_gm = (np.sum(np.log(channel_2 + 1))) / channel_2.shape[0]
            channel_3_mean_gm = (np.sum(np.log(channel_3 + 1))) / channel_3.shape[0]

            channel_1_gm += channel_1_mean_gm
            channel_2_gm += channel_2_mean_gm
            channel_3_gm += channel_3_mean_gm

            ##### HM #####
            channel_1_mean_hm = (np.sum(np.reciprocal(channel_1 + 1))) / channel_1.shape[0]
            channel_2_mean_hm = (np.sum(np.reciprocal(channel_2 + 1))) / channel_2.shape[0]
            channel_3_mean_hm = (np.sum(np.reciprocal(channel_3 + 1))) / channel_3.shape[0]

            channel_1_hm += channel_1_mean_hm
            channel_2_hm += channel_2_mean_hm
            channel_3_hm += channel_3_mean_hm

        channel_1_AM = channel_1_am / N
        channel_2_AM = channel_2_am / N
        channel_3_AM = channel_3_am / N

        channel_1_GM = np.exp(channel_1_gm / N)
        channel_2_GM = np.exp(channel_2_gm / N)
        channel_3_GM = np.exp(channel_3_gm / N)

        channel_1_HM = N / channel_1_hm
        channel_2_HM = N / channel_2_hm
        channel_3_HM = N / channel_3_hm

        arith_mean = tuple((channel_1_AM, channel_2_AM, channel_3_AM))
        geome_mean = tuple((channel_1_GM, channel_2_GM, channel_3_GM))
        harmo_mean = tuple((channel_1_HM, channel_2_HM, channel_3_HM))

        return arith_mean, geome_mean, harmo_mean

    def file_reader(self, return_paths=False) -> list:
    
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
