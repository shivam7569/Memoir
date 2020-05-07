feature\_scaling
================

.. automodule:: memoir.batch_preprocessing.feature_scaling
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python
   :linenos:

   from memoir.batch_preprocessing import feature_scaling as fs

   img_batch = <batch_of_images>

   ##### Normalization #####

   norm_scaler = fs.Normalize(
                           v_type='Animated',
                           color_space='gray',
                           **kwargs[minimum, maximum]
                           )
   normalized_batch = norm_scaler.fit(img_batch)

   ##### Standardization #####

   std_scaler = fs.Standardize(
                           v_type='Animated',
                           color_space='gray',
                           **kwargs[mean, std]
                           )
   standardized_batch = std_scaler.fit(img_batch)

   ##### Mean-Normalization #####

   m_norm_scaler = fs.Mean_Normalize(
                           v_type='Animated',
                           color_space='gray',
                           **kwargs[mean, minimum, maximum]
                           )
   mean_normalized_batch = m_norm_scaler.fit(img_batch)

   ##### Unit_Vectorization #####

   unit_scaler = fs.Unit_Vector()
   unit_vectorized_batch = unit_scaler.fit(img_batch, factor=20)