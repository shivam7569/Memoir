Generating frames from videos for dataset
=========================================

.. code-block:: python
    :linenos:

    import memoir.data.video_to_frames as vf

    vf.convert_vid_to_frames(
                        vid_type='Animated',
                        series='All',
                        threshold=50000
                        )

Generating a batch of images for training, testing, and validation
==================================================================

.. code-block:: python
    :linenos:

    # Make sure you generate frames before proceeding further.

    from memoir.data import batch_generator as bg
    from memoir.batch_preprocessing import train_test_split as tts

    img_names = bg.image_names_generator(
                                    v_type='Animated',
                                    series='All'
                                    )
    train_images, test_images, val_images = tts.train_test_val(
                                                            image_names=img_names,
                                                            test_fraction=0.25,
                                                            val_data=True,
                                                            val_fraction=0.1
                                                            )

Getting the names of available transformation in data data_augmentation
=======================================================================

.. code-block:: python
    :linenos:

    from memoir.batch_preprocessing import data_augmentation as da

    da.available_transformations()

Data Augmentation of a batch of images
======================================

.. code-block:: python
    :linenos:

    from memoir.data import batch_generator as bg
    from memoir.batch_preprocessing import data_augmentation as da

    batch = bg.batch_generator(
                            image_names=train_images,
                            batch_size=64,
                            image_size=(320, 240)
                            )

    augmented_batch = da.data_aug(
                                batch,
                                size_of_aug=0.3,
                                techniques='All',
                                num_of_trans=5,
                                repeat=True
                                )

Getting the names of available color spaces for color space transformation
==========================================================================

.. code-block:: python
    :linenos:

    from memoir.batch_preprocessing import channels

    channels.available_channels(_return_=False, _print_=True)

Change the color space of an image or a batch
=============================================

.. code-block:: python
    :linenos:

    import cv2
    from memoir.data import batch_generator as bg
    from memoir.batch_preprocessing import channels

    batch = bg.batch_generator(
                            image_names=train_images,
                            batch_size=64,
                            image_size=(320, 240)
                            )

    image = cv2.imread("Read an image")

    image_after_conversion = channels.change_channel(image, '<desired_channel>')
    batch_after_conversion = channels.change_channel(batch, '<desired_channel>')

Feature Scaling of a batch of images
====================================

.. code-block:: python
    :linenos:

    # Make sure you generate frames before proceeding further.

    from memoir.batch_preprocessing import feature_scaling as fs
    from memoir.data import batch_generator as bg

    batch = bg.batch_generator(
                            image_names=train_images,
                            batch_size=64,
                            image_size=(320, 240)
                            )

    ##### Normalization #####

    norm_scaler = fs.Normalize(
                            v_type='Animated',
                            color_space='gray',
                            **kwargs[minimum, maximum]
                            )
    normalized_batch = norm_scaler.fit(batch)

    ##### Standardization #####

    std_scaler = fs.Standardize(
                            v_type='Animated',
                            color_space='gray',
                            **kwargs[mean, std]
                            )
    standardized_batch = std_scaler.fit(batch)

    ##### Mean-Normalization #####

    m_norm_scaler = fs.Mean_Normalize(
                            v_type='Animated',
                            color_space='gray',
                            **kwargs[mean, minimum, maximum]
                            )
    mean_normalized_batch = m_norm_scaler.fit(batch)

    ##### Unit_Vectorization #####

    unit_scaler = fs.Unit_Vector()
    unit_vectorized_batch = unit_scaler.fit(batch, factor=20)