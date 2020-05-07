channels
========

.. automodule:: memoir.batch_preprocessing.channels
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: bgr2gray, bgr2hls, bgr2hsv, bgr2lab, bgr2luv, bgr2xyz, bgr2ycr_cb, bgr3yuv, gray2bgr, bgr2yuv

.. code-block:: python
   :linenos:

   from memoir.batch_preprocessing import channels
   import cv2

   channels.available_channels(
                           _return_=False,
                           _print_=True
                           )

   image = cv2.imread("Read an image")

   image_after_conversion = channels.change_channel(
                                                batch=image,
                                                channel='lab'
                                                )