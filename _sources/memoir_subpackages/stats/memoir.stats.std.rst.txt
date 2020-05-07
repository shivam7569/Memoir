std
===

.. automodule:: memoir.stats.std
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: file_reader

.. code-block:: python
   :linenos:

   from memoir.stats.std import Std

   std_object = Std(
                 v_type='Animated',
                 image_size=(320, 240),
                 color_space='gray'
                 )

   mean_of_data = <get_the_mean_of_the_data>

   standard_deviation, variance = std_object.standard_deviation(
                                                            means=mean_of_data,
                                                            return_variance=True
                                                            )