data\_augmentation
==================

.. automodule:: memoir.batch_preprocessing.data_augmentation
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: avail_transf, geoT, horizontal_flip, vertical_flip, pad, scaling, vc_affine, vc_brightness, vc_contrast, vc_crop, vc_noise, vc_rotation

.. code-block:: python
   :linenos:

   from memoir.batch_preprocessing import data_augmentation as da

   img_batch = <batch_of_images>

   augmented_batch = da.data_aug(
                              batch=img_batch,
                              size_of_aug=0.3,
                              techniques='All',
                              num_of_trans=5,
                              repeat=True
                              )