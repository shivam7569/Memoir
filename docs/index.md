# Documentation

## Pre-requisites

* Fork the repository.
* Clone the forked repository. 

## Generate Data for Training

### `videos_to_frames.py`

***Generates the data for training from the videos.***
    
    * First checks for the Data_Memoir(videos) folder in Data directory.
    * If found, calls the function `process_call` to start the process.
    * If not found, raises an OSError.
    * Asks for user input for the type of video to generate data for. We use only 'Animated' and 'Real' for now. Default: 'All'.
    * Also asks for the number of frames to generate for one series. Default: 50000

* Download the videos from this [link](https://drive.google.com/drive/folders/1JgnDgGsDxWffh41VOcvSjVLrFJVxZdCp?usp=sharing).
* Put this folder in the **Data** directory of the repo.
* Make sure the name of the folder is **Memoir_Videos**.
* Now run the code `videos_to_frames.py`
* It will create a folder **Data_Memoir** in the repository.

## Generating Batch of Images for Training

### `batch_processor.py`

***Returns a batch of image***

    image_names_generator(v_type='Animated', series='All')
    
            * v_type: Type of the video to consider. Default: 'Animated'
            * series: Name of the series from the given v_type to create batch from. Default: 'All'

        - returns the names of the images for the given values of v_type and series.

    batch_generator(images, batch_size=64, image_size=(576, 384))

            * images: Names of the images returned by image_name_generator
            * batch_size: Size of the batch. Default: 64
            * image_size: Size of the final batch images. Default: (576, 384)

        - returns a batch of images

* Import the functions from python file `batch_processor.py`.
* First call the function **image_names_generator**.
* Now call the function **batch_generator** on the returned value from **image_names_generator** function to get the batch of images.
* This is done to save processing time by executing the task done by **image_names_generator** only once, and not everytime a batch is generated.