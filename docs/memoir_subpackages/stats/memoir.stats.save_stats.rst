save\_stats
===========

.. automodule:: memoir.stats.save_stats
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: file_reader

.. code-block:: python
   :linenos:

   from memoir.stats.save_stats import calculate_stats

   stats = calculate_stats(
                     video_type='All',
                     color_space='gray',
                     image_size=(320, 240),
                     save_stats=True,
                     overwrite=True,
                     _return_=True
                     )