train\_test\_split
==================

.. automodule:: memoir.batch_preprocessing.train_test_split
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python
   :linenos:

   from memoir.batch_preprocessing import train_test_split as tts

   img_names = <list_of_names_of_images_in_the_concerned_dataset>

   train_images, test_images, val_images = tts.train_test_val(
                                                         image_names=img_names,
                                                         test_fraction=0.25,
                                                         val_data=True,
                                                         val_fraction=0.1
                                                         )