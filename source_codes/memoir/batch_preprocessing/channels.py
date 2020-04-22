import cv2
import numpy as np 
from numpy import ndarray

def bgr2gray(batch: ndarray):

    '''
    Function to convert the color space of a batch of images from "BGR" to "GRAY"

    Args:
        batch: A batch of images in "BGR" colorspace. 
    '''

    new_batch = []
    for img in batch:
        _img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_batch.append(_img)

    return np.array(new_batch)

def bgr2hsv(batch: ndarray):

    '''
    Function to convert the color space of a batch of images from "BGR" to "HSV"

    Args:
        batch: A batch of images in "BGR" colorspace. 
    '''

    new_batch = []
    for img in batch:
        _img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        new_batch.append(_img)

    return np.array(new_batch)

def bgr2lab(batch: ndarray):

    '''
    Function to convert the color space of a batch of images from "BGR" to "LAB"

    Args:
        batch: A batch of images in "BGR" colorspace. 
    '''

    new_batch = []
    for img in batch:
        _img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        new_batch.append(_img)

    return np.array(new_batch)

def bgr2luv(batch: ndarray):

    '''
    Function to convert the color space of a batch of images from "BGR" to "LUV"

    Args:
        batch: A batch of images in "BGR" colorspace. 
    '''

    new_batch = []
    for img in batch:
        _img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        new_batch.append(_img)

    return np.array(new_batch)

def bgr2hls(batch: ndarray):

    '''
    Function to convert the color space of a batch of images from "BGR" to "HLS"

    Args:
        batch: A batch of images in "BGR" colorspace. 
    '''

    new_batch = []
    for img in batch:
        _img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        new_batch.append(_img)

    return np.array(new_batch)

def bgr2xyz(batch: ndarray):

    '''
    Function to convert the color space of a batch of images from "BGR" to "XYZ"

    Args:
        batch: A batch of images in "BGR" colorspace. 
    '''

    new_batch = []
    for img in batch:
        _img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
        new_batch.append(_img)

    return np.array(new_batch)

def bgr2ycr_cb(batch: ndarray):

    '''
    Function to convert the color space of a batch of images from "BGR" to "YCR_CB"

    Args:
        batch: A batch of images in "BGR" colorspace. 
    '''

    new_batch = []
    for img in batch:
        _img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        new_batch.append(_img)

    return np.array(new_batch)

def bgr2yuv(batch: ndarray):

    '''
    Function to convert the color space of a batch of images from "BGR" to "YUV"

    Args:
        batch: A batch of images in "BGR" colorspace. 
    '''

    new_batch = []
    for img in batch:
        _img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        new_batch.append(_img)

    return np.array(new_batch)

def change_channel(batch: ndarray, channel='gray'):

    '''
    Function to convert the color space of a batch of images from "BGR" to the user specified color space.

    Args:
        batch(ndarray): A batch of images in "BGR" colorspace. 
        channel(str): The desired color space of images.

    Returns:
        A batch of images with converted color space.

    '''

    channel_dict = {
        'gray': bgr2gray,
        'hsv': bgr2hsv,
        'lab': bgr2lab,
        'luv': bgr2luv,
        'hls': bgr2hls,
        'xyz': bgr2xyz,
        'ycr_cb': bgr2ycr_cb,
        'yuv': bgr2yuv
    }

    return channel_dict[channel](batch)

def available_channels(_return_=False):

    '''
    Function to display available color space to use.

    Args:
        _return_(bool): Whether to return a list of available color space to use,
    '''

    avail_channels = ['gray', 'hsv', 'lab', 'luv', 'hls', 'xyz', 'ycr_cb', 'yuv']

    print('\n')
    for channel in avail_channels:
        print(channel)

    if _return_:
        return avail_channels