import os
import random

import cv2 as cv
import numpy as np
from skimage import data, exposure, io
from skimage import transform as sk_tf
from skimage.util import crop, pad


def vc_affine(image_array: np.ndarray):

    # Function to apply affine transformation on a image.

    '''
    image_array: The image to apply transformation on.
    '''

    ran_num = random.randint(0, 1) # To randomly apply rotation too.
    ran_num = 0

    if ran_num == 0:

        # Only image translation will take place.

        srcTri = np.array([[0, 0],
                        [image_array.shape[1] - 1, 0],
                        [0, image_array.shape[0] - 1]]).astype(np.float32)
        
        dstTri = np.array([[0, image_array.shape[1] * 0.2],
                        [image_array.shape[1] * 0.9, image_array.shape[0] * 0.3],
                        [image_array.shape[1] * 0.15, image_array.shape[0] * 0.7]]).astype(np.float32)
        
        warp_mat = cv.getAffineTransform(srcTri, dstTri)
        warp_img = cv.warpAffine(image_array, warp_mat,
                                (image_array.shape[1], image_array.shape[0]))
        warp_img = cv.resize(warp_img, image_size)

        return warp_img

    if ran_num == 1:

        # Both rotation and translation will take place.

        srcTri = np.array([[0, 0], [image_array.shape[1] - 1, 0],
                           [0, image_array.shape[0] - 1]]).astype(np.float32)
        dstTri = np.array([[0, image_array.shape[1]*0.33], [image_array.shape[1]*0.85, image_array.shape[0]
                                                    * 0.25], [image_array.shape[1]*0.15, image_array.shape[0]*0.7]]).astype(np.float32)
        warp_mat = cv.getAffineTransform(srcTri, dstTri)
        warp_img = cv.warpAffine(image_array, warp_mat, (image_array.shape[1], image_array.shape[0]))
        warp_img = cv.resize(warp_img, image_size)
        
        # Rotating the image after Warp
        
        center = (warp_img.shape[1]//2, warp_img.shape[0]//2)
        angle = random.uniform(-90, 90)
        scale = random.uniform(0, 0.5)
        rot_mat = cv.getRotationMatrix2D(center, angle, scale)
        warp_rotate_img = cv.warpAffine(warp_img, rot_mat,
                                        (warp_img.shape[1], warp_img.shape[0]))

        return warp_rotate_img

def vc_rotation(image_array: np.ndarray, radius=None):

    # Function to apply rotational transformation to an image.

    '''
    image_array: The image to apply transformation on.
    radius: The radius of the circle (centered at the middly point of image) to pick the centre of rotation from.
    '''

    if radius == None:
        radius = random.randint(0, 30)

    x_centre = random.randint(
        image_size[0]//2 - radius, image_size[0]//2 - radius)

    y_centre = random.randint(
        image_size[1]//2 - radius, image_size[1]//2 + radius)

    angle = round(random.uniform(-180, 180), 2)

    center = tuple(np.array([x_centre, y_centre]))
    rot_img = cv.getRotationMatrix2D(center, angle, 1.0)
    rot_img = cv.warpAffine(image_array, rot_img, image_size)
    rot_img = cv.resize(rot_img, image_size)
    
    return rot_img

def vc_noise(image_array: np.ndarray, noise_typ=None):

    # Function to add noise to an image.

    '''
    image_array: The image to apply transformation on.
    noise_type: Type of the noise to apply - gauss, salt&pepper, poisson, speckle.
    '''

    if noise_typ == None:

        noise_typ = random.choice(['gauss', 's&p', 'poisson', 'speckle'])

    if noise_typ == "gauss":

        row, col, ch = image_array.shape
        mean = random.uniform(0, 1)
        var = random.uniform(0, 0.1)
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image_array + gauss
        noisy_img = cv.resize(noisy, image_size)
        
        return noisy_img
    
    elif noise_typ == "s&p":
        
        row, col, ch = image_array.shape
        s_vs_p = random.uniform(0, 1)
        amount = random.uniform(0, 0.05)
        out = np.copy(image_array)
        
        # Salt mode
        num_salt = np.ceil(amount * image_array.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image_array.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image_array.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image_array.shape]
        out[coords] = 0
        noisy_img = cv.resize(out, image_size)
        
        return noisy_img
    
    elif noise_typ == "poisson":
        
        vals = len(np.unique(image_array))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image_array * vals) / float(vals)
        noisy_img = cv.resize(noisy, image_size)
        
        return noisy_img
    
    elif noise_typ == "speckle":
        
        factor = random.uniform(0, 0.4)
        row, col, ch = image_array.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image_array + image_array * gauss * factor
        noisy_img = cv.resize(noisy, image_size)
        
        return noisy_img

def vc_brightness(image_array: np.ndarray):

    # Function to alter the brightness of an image.

    '''
    image_array: The image to apply transformation on.
    '''

    value = random.randint(0, 40)

    hsv = cv.cvtColor(image_array, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    bright_img = cv.resize(img, image_size)
    
    return bright_img

def vc_crop(image_array: np.ndarray):

    # Function to crop an image.
    # Pad image by n_pixels on each size, then take random crop of same original size.
    
    '''
    image_array: The image to apply transformation on.
    '''

    n_pixels = 40
    
    pad_mode = random.choice(['edge', 'constant', 'linear_ramp', 'maximum', 'mean', 'median',
                              'minimum', 'reflect', 'symmetric', 'wrap', 'empty'])
    
    assert len(image_array.shape) == 3
    
    w, h, nc = image_array.shape

    padded = pad(image_array, [(n_pixels, n_pixels) for _ in range(2)] + [(0, 0)],
                 mode=pad_mode)

    crops = [(c, 2*n_pixels-c)
             for c in np.random.randint(0, 2*n_pixels+1, [2])]
    
    crops += [(0, 0)]
    crop_ = crop(padded, crops, copy=True)
    crop_out = cv.resize(crop_, image_size)
    
    return crop_out

def vc_contrast(image_array: np.ndarray):

    # Function to alter the contrast of an image.

    '''
    image_array: The image to apply transformation on.
    '''

    r_min, r_max = random.uniform(0, 40), random.uniform(60, 100)
    
    v_min, v_max = np.percentile(image_array, (r_min, r_max))
    exp_img = exposure.rescale_intensity(image_array, in_range=(v_min, v_max))
    exp_img = cv.resize(exp_img, image_size)
    
    return exp_img


def horizontal_flip(image_array: np.ndarray):

    # Function to horizontally flip an image.

    '''
    image_array: The image to apply transformation on.
    '''
    
    return image_array[:, ::-1]


def vertical_flip(image_array: np.ndarray):

    # Function to vertically flip an image.

    '''
    image_array: The image to apply transformation on.
    '''

    return image_array[::-1, :]


def scaling(image_array: np.ndarray):
    
    # Function to scale an image.

    '''
    image_array: The image to apply transformation on.
    '''

    bool_ = random.choice([True, False])
    
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    
    scaled_r = sk_tf.rescale(r, 0.25, anti_aliasing=bool_)
    scaled_g = sk_tf.rescale(g, 0.25, anti_aliasing=bool_)
    scaled_b = sk_tf.rescale(b, 0.25, anti_aliasing=bool_)
    scaled_img = np.dstack((scaled_r, scaled_g, scaled_b))
    scaled_img = cv.resize(scaled_img, image_size)
    
    return scaled_img


def geoT(image_array: np.ndarray):

    # Function to apply geometrical transformation on an image.

    '''
    image_array: The image to apply transformation on.
    '''

    src = np.array([[0, 0], [0, 50], [300, 50], [300, 0]])
    dst = np.array([[155, 15], [65, 40], [260, 130], [360, 95]])

    tform3 = sk_tf.ProjectiveTransform()
    tform3.estimate(src, dst)
    geoT_image = sk_tf.warp(image_array, tform3, output_shape=(50, 300))
    geoT_image = cv.resize(geoT_image, image_size)
    return geoT_image


def available_transformations():

    dict_ = avail_transf()
    print('\n')
    for key in dict_.keys():
        print(key)
    print('\n')

# dictionary of the transformations we defined earlier


def avail_transf():
    avail_transformations = {
        'Rotation': vc_rotation,
        'Affine': vc_affine,
        'Noise': vc_noise,
        'Brightness': vc_brightness,
        'Crop': vc_crop,
        'Contrast': vc_contrast,
        'Horizontal_flip': horizontal_flip,
        'Vertical_flip': vertical_flip,
        'Scaling': scaling,
        'Geometrical': geoT
    }
    return avail_transformations

def data_aug(batch, size_of_aug=None, techniques='All'):

    # Function to perform data augmentation.

    '''
    batch: The batch of images to perform augmentation on.
    size_of_aug: The size of the augmented part of the returned batch after transformation:
                if float: Take it as the fraction of the batch to augment.
                if integer: Augment only that many images.
                if None: Take 30% of the batch for augmentation.
    '''
    
    if size_of_aug == None:
        size_of_aug = int(batch.shape[0] * 0.4)

    if isinstance(size_of_aug, float):
        size_of_aug = int(batch.shape[0] * size_of_aug)

    avail_transformations = avail_transf()

    if techniques != 'All':
        # print(keys)
        for tech in list(avail_transformations):

            try:
                if tech in techniques:
                    continue
                else:
                    del avail_transformations[tech]
            except KeyError as ex:
                print("No such key: '%s'" % ex.message)

    global image_size
    image_size = batch[0].shape[1::-1]

    num_generated_files = 1
    while num_generated_files <= size_of_aug:
        index = random.choice(range(batch.shape[0]))
        image = batch[index]
        num_transformations_to_apply = random.randint(
            1, len(avail_transformations))

        num_transformations = 0
        while num_transformations <= num_transformations_to_apply:
            key = random.choice(list(avail_transformations))
            transformed_image = avail_transformations[key](image)
            num_transformations += 1

        batch[index] = transformed_image
        num_generated_files += 1

    return batch
