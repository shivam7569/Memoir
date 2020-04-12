import os
import random
from scipy import ndarray

# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
from skimage import data, exposure
from skimage.util import crop

#rotation

def vc_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)


#noise

def vc_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

#brightness

def vc_brightness(image_array: ndarray):

    gamma=1
    gain=1

    if is_random:
        gamma = np.random.uniform(1-gamma, 1+gamma)
    return exposure.adjust_gamma(image_array, gamma, gain)
   

#crop

def vc_crop(image_array: ndarray):
    """Pad image by n_pixels on each size, then take random crop of same
    original size.
    """
    n_pixels=4
    pad_mode='edge'
    assert len(image_array.shape) == 3
    h, w, nc = image_array.shape

    # First pad image by n_pixels on each side
    padded = pad(image_array, [(n_pixels, n_pixels) for _ in range(2)] + [(0,0)],
        mode=pad_mode)

    # Then take a random crop of the original size
    crops = [(c, 2*n_pixels-c) for c in np.random.randint(0, 2*n_pixels+1, [2])]
    # For channel dimension don't do any cropping
    crops += [(0,0)]

    return crop(padded, crops, copy=True) 
#without adding padded pixels
#crop
# crop_width{sequence, int}: Number of values to remove from the edges of each axis. 
# ((before_1, after_1), … (before_N, after_N)) specifies unique crop widths at the 
# start and end of each axis. ((before, after),) specifies a fixed start and end 
# crop for every axis. (n,) or n for integer n is a shortcut for before = after = n 
# for all axes.
def vc_crop2(image_array:ndarray):
    
    
    return crop(A, ((50, 100), (50, 50), (0,0)), copy=False)

#affine transformation

def vc_affine(image_array: ndarray):

    return skimage.transform.AffineTransform(image_array)

def vc_contrast(image_array: ndarray):


    v_min, v_max = np.percentile(image_array, (0.2, 99.8))
    return exposure.rescale_intensity(image_array, in_range=(v_min, v_max))


# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': vc_rotation,
    'noise': vc_noise,
    'brightness': vc_multi,
    'affine': vc_affine,
    'crop': vc_crop,
    'contrast': vc_contrast
}


folder_path = '/Desktop/videocolor'

#no. of images we want 
num_files_desired = 10 

# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 0
while num_generated_files <= num_files_desired:
    # random image from the folder
    image_path = random.choice(images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    # random num of transformation to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1

new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, num_generated_files)

# write image to the disk
io.imsave(new_file_path, transformed_image)
num_generated_files += 1
