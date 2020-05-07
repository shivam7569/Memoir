import json
import os

import cv2
import numpy as np


def d_tree():
    vid_types = os.listdir(
        paths['frame_dir']
        # To get the names of the types of videos in the Data_Memoir directory.
    )
    
    global data_tree

    data_tree = {}  # To store the structure of Data_Memoir directory.

    for v_t in vid_types:

        # Looping over each video type to get the structure upto Series level in 'data_tree' variable.

        data_tree[v_t] = os.listdir(
            os.path.join(
                paths['frame_dir'], v_t
                # Taking the video type as key and listing its content as the corresponding value.
            ))


def find_key(sers):
    '''
    sers: The name of the series to find video type for.
    '''

    # To find the video type for a series, used in determing the path of an image.

    for key in data_tree.keys():
        # Looping over the keys (video types in the Data_Memoir directory) to find the one for a given series.
        if sers in data_tree[key]:
            return key


def path_maker(v_type, srs):
    '''
    v_type: Name of the type of the series.
    srs: Name of the series.
    '''

    # Function to return the path of the frames for a given series.

    path = os.path.join(paths['frame_dir'], v_type, srs, 'Frames')
    return path


def batch_generator(image_names, batch_size=64, image_size=(320, 240)):

    '''
    Generates a batch of images. These images are the ones filtered out by `memoir.data.batch_generator.image_names_generator`.  
    
    Args:
        image_names (list): List of names of images for a given video_type and series, as returned by `image_names_generator`.
        batch_size (int): Size of the batch of images.
        image_size (tuple): Size of the images in the batch.    
    
    Returns:
        Returns a numpy array of images.
    '''

    # Function to create a batch of images from the list of names of images searched according to user query in batch_generator function

    batch = []  # list to store images of batch
    random_indices = np.random.randint(
        low=0, high=len(image_names), size=batch_size)  # Generating random indices to pick images
    for ind in random_indices:

        # Reading and storing the images according to the random indices generated, after resizing them.

        image_name = image_names[ind]

        img = cv2.imread(image_name)  # Reading an image.
        img = cv2.resize(img, image_size)  # Resizing the image.

        batch.append(img)  # Appending the image to the batch (list).

    batch = np.array(batch)
    return batch


def image_names_generator(v_type='Animated', series='All'):
    
    '''
    This function is to find the names of the frames according to user query. These names will be used to generate a batch.
    This approach helps in preventing memory overloading by not loading the frames but only their names in the memory. 
    
    Args:
        v_type (str): Type of the series (For now, 'Animated'? or 'Real'?) to take frames for the batch from. Default: 'Animated'
        series (str): The search can be made more specific if frames are needed only from one particular series. Default: 'All'
    
    Returns:
        Returns a list of names of frames according to user query, to be used to generate batches.
    
    Raises:
        KeyError: When the `series` entered is not present in `v_type`.
    '''

    file_path = os.path.dirname(os.path.realpath(__file__)) 
    os.chdir(file_path)

    with open('../../../paths.json', 'r') as file:
        global paths
        paths = json.load(file)

    d_tree()

    if not isinstance(series, str) or series != 'All':

        # Checking if entered name of the series is valid string.

        if series not in data_tree[v_type]:

            # Checking whether entered name of the series belong to the entered type of video.

            raise KeyError('\nEntered series is either not present in the data or does not belong to entered video type. Terminating...')
            return None

    all_images = []

    if series == 'All':

        # To be executed if the batch to be made of all the frames for a given video type.

        for srs in data_tree[v_type]:

            # Looping over all the series of the given video type.

            # Getting the video type of the series using find_key function.
            v_type = find_key(srs)
            # Getting the path to frames of the series using path_maker function.
            path = path_maker(v_type, srs)
            all_images += [os.path.join(path, fram_name)
                           for fram_name in os.listdir(path)]  # Storing the names of the images for the given query.

        return all_images

    else:

        # To be executed if the batch to be made of all the frames for a given series.

        v_type = find_key(series)
        path = path_maker(v_type, series)
        all_images += [os.path.join(path, fram_name)
                       for fram_name in os.listdir(path)]

        return all_images
