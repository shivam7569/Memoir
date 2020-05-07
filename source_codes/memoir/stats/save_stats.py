import copy
import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from memoir.batch_preprocessing import channels
from memoir.stats.mean import Mean
from memoir.stats.std import Std


def calculate_stats(
    video_type: str = "All",
    color_space: str = "gray",
    image_size: tuple = (320, 240),
    save_stats: bool = True,
    overwrite: bool = False,
    _return_: bool = False,
) -> dict:

    '''
    Function to calculate and save (in a json file) the basis statistic measures, like mean (arithmetic, geometric and harmonic), variance, and standard deviation.
    Since image datasets are often large, this function calculates the stats using running algorithm. It takes about an hour, for 250000 images and one color space,
    to calculate one stat. Arithmetic, Geometric and Harmonic means are calculated together, while standard deviation uses the mean to execute.

    Args:
        video_type (str): Type of the video to calculate the stats for. Default: 'All'
        color_space (str, list): The color space of the images to calculate the stats for. Either a single string value or a list of string of color spaces. Use 'All' for all the available color spaces. Default: 'gray'
        image_size (tuple): Size of the images to be used. Default: (320, 240)
        save_stats (bool): Whether to save the stats in `data_statistics.json`. Default: True
        overwrite (bool): If there is an existing `data_statistics.json` file, whether to overwrite it or not. Default: False
        _return_ (bool): Whether to return the calculated stats. Useful when needed for further processing, if any. Default: False

    Returns:
        A dictionary of the calculated statistics when `_return_` parameter is set to `True`.

    Raises:
        Warning: when there is an existing `data_statistics.json` file.
        ValueError: when the value of `video_type' or `color_space` is not found in the dataset.
        TypeError: when the data type of `video_type` or `color_space` does not match the allowed ones.
    '''

    paths = file_reader()

    if video_type == "All":

        video_types = os.listdir(paths["frame_dir"])

    elif not isinstance(v_type, str):

        raise TypeError("Invalid data type of video_type parameter. Can only accept \'str\' values.")
        exit(0)

    elif v_type not in os.listdir(paths["frame_dir"]):

        raise ValueError("Invalid entry for video_type. Entered value not found in the dataset")
        exit(0)

    else:

        video_types = [video_type]

    if color_space == "All":

        avail_c = channels.available_channels(_return_=True, _print_=False)

    elif isinstance(color_space, str):

        if color_space in channels.available_channels(_print_=False, _return_=True):

            avail_c = [color_space]

        else:

            raise ValueError("Invalid entry for color_space. Value not found in the available color spaces.")
            exit(0)

    elif isinstance(color_space, list):

        avail_c = color_space

    else:

        raise TypeError("Invalid data type of color_space parameter. Can only accept \'str\' values.")
        exit(0)

    data_sts = {}
    stats_keys = {}
    color_spc = {}

    for v_d in video_types:

        for key in avail_c:

            _m_ = Mean(v_type=v_d, image_size=image_size, color_space=key)
            _s_ = Std(v_type=v_d, image_size=image_size, color_space=key)

            am, gm, hm = _m_.am_gm_hm()
            std, var = _s_.standard_deviation(means=am, return_variance=True)

            stats_keys["arithmetic_mean"] = am
            stats_keys["geometric_mean"] = gm
            stats_keys["harmonic_mean"] = hm

            stats_keys["std"] = std
            stats_keys["var"] = var

            stats_keys["min"] = (0, 0, 0)
            stats_keys["max"] = (255, 255, 255)

            color_spc[key] = copy.deepcopy(stats_keys)
            data_sts[v_d] = copy.deepcopy(color_spc)

    if save_stats:

        stat_file = Path("./data_statistics.json")

        try:

            abs_path = stat_file.resolve(strict=True)
            warnings.warn("Stats file already exists.")

            if overwrite:

                print("Overwriting the existing file...")

                with open("./data_statistics.json", "w") as file:
                    json.dump(data_sts, file)

            else:

                print("Not overwriting the existing file.")

        except:

            with open("./data_statistics.json", "w") as file:
                json.dump(data_sts, file)

    if _return_:
        return data_sts


def file_reader() -> dict:

    file_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_path)

    with open("../../../paths.json", "r") as file:
        paths = json.load(file)

    return paths
