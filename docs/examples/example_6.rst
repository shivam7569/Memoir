Change the color space of an image or a batch
=============================================

.. automodule:: None
   :members:
   :undoc-members:
   :show-inheritance:
   
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