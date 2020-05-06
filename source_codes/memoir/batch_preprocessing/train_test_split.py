from random import shuffle
from typing import Union

import cv2


def train_test_val(
    image_names: list,
    test_fraction: Union[float, int] = 0.25,
    val_data: bool = True,
    val_fraction: float = 0.1,
):

    '''
    Function to split the dataset into train, test, and validation sets.

    Args:
        image_names (list): Dataset to be used for training.
        test_fraction (int, float): Size of the test set. Default: 0.25
            If **float**, takes it as the fraction of the whole dataset.
            If **integer**, takes only that many images in the test set from the whole dataset.
        val_data (bool): Whether to make validation set or not.
        val_fraction (float, int): Size of the validation set. Default: 0.1
            If **float**, takes it as the fraction of the train dataset.
            If **integer**, takes only that many images in the validation set from the train set.

    Returns:
        Train, test and validation (if asked) datasets.
    '''

    total_samples = len(image_names)
    shuffle(image_names)
    test = image_names[: int(test_fraction * total_samples)]
    train = image_names[int(test_fraction * total_samples) :]

    if val_data:

        total_train = len(train)
        shuffle(train)
        validation = train[: int(val_fraction * total_train)]
        train = train[int(val_fraction * total_train) :]

        return train, test, validation

    else:

        return train, test
