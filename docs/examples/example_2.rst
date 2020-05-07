Generating a batch of images for training, testing, and validation
==================================================================

.. automodule:: None
   :members:
   :undoc-members:
   :show-inheritance:
   
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