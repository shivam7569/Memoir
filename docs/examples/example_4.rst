Data Augmentation of a batch of images
======================================

.. automodule:: None
   :members:
   :undoc-members:
   :show-inheritance:
   
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