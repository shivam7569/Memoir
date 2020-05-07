mean
====

.. automodule:: memoir.stats.mean
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: file_reader

.. code-block:: python
   :linenos:

   from memoir.stats.mean import Mean

   mean_object = Mean(
                     v_type='Animated',
                     image_size=(320, 240),
                     color_space='gray'
                     )

   ##### Arithmetic Mean #####

   AM = mean_object.arithmetic_mean()

   ##### Geometric Mean #####

   GM = mean_object.geometric_mean()

   ##### Harmonic Mean #####

   HM = mean_object.harmonic_mean()

   ##### All means in single run #####

   AM_GM_HM = mean_object.am_gm_hm()