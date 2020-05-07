Feature Scaling of a batch of images
====================================

.. automodule:: None
   :members:
   :undoc-members:
   :show-inheritance:
   
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