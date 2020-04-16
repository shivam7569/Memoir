![](./images/Logo.jpg)

## Dataset

The videos used for generating the dataset are available [here](https://drive.google.com/open?id=1JgnDgGsDxWffh41VOcvSjVLrFJVxZdCp).

## Documentation

The documentation is available [here](https://andy6975.github.io/Memoir/).

## Installation

### Fork the repository.

### Clone the forked repository.
```
$ git clone <link_to_forked_repository>
```

### Change the directory to Memoir.
```
$ cd /path/to/Memoir/
```

### Install the dependencies.

**For Python (virtual) environment:**
```
$ pip3 install numpy
$ pip3 install scikit-image
$ pip3 install opencv-python
$ pip3 install tensorflow==1.14 (cpu)
$ pip3 install tensorflow-gpu==1.14 (gpu)**
```
** You will need CUDA and cuDNN libraries to run tensorflow on GPU.

[Building Tensorflow 1.14 with GPU Support and TensorRT on Ubuntu 16.04 LTS](https://medium.com/analytics-vidhya/building-tensorflow-1-14-with-gpu-support-and-tensorrt-on-ubuntu-16-04-84bbd356e03)

[Building Tensorflow 2.0 with GPU support and TensorRT on Ubuntu 18.04 LTS](https://medium.com/analytics-vidhya/building-tensorflow-2-0-with-gpu-support-and-tensorrt-on-ubuntu-18-04-lts-part-1-e04ce41f885c)

**For Conda environment:**

Either use `anaconda-navigator` to install the dependencies, including tensorflow-gpu==1.14. It will automatically take care of CUDA and cuDNN. Or you could just duplicate the developer's environment using:
```
conda env create -f requirements.yml
```
### Install the package Memoir:

**For both the environments.**
```
(Memoir)$ pip install .
```
to install in editable (developer) mode:
```
(Memoir)$ pip install -e .
```

### To uninstall Memoir,
```
$ pip uninstall -y Memoir
```
