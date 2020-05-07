batch\_generator
================

.. automodule:: memoir.data.batch_generator
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: d_tree, find_key, path_maker

.. code-block:: python
   :linenos:

   from memoir.data import batch_generator as bg

   img_names = bg.image_names_generator(
                                       v_type='Animated',
                                       series='All'
                                       )
   batch = bg.batch_generator(
                           image_names=img_names,
                           batch_size=64,
                           image_size=(320, 240)
                           )