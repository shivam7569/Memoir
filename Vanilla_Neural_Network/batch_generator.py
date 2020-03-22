import os
import cv2
import random
import numpy as np

os.chdir('./Data/')
anim_chars = os.listdir()

frames_dict = {}
all_images = []
batch_size = 10
shape = (720, 480)

def rgb2lab(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return lab_image

def rgb2hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def rgb2gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def file_path(name):
    folder = name.split('_')[0]
    path = './' + folder + '/Frames/' + name
    return path

def batch_gen():
    global frames_dict, all_images, batch_size

    batch = []

    for i in range(len(anim_chars)):
        frames_dict[i+1] = sorted(os.listdir('./' + anim_chars[i] + '/Frames/'))
        all_images += frames_dict[i+1]

    batch_indices = [random.randrange(0, 100000, 1) for i in range(batch_size)]

    for ind in batch_indices:
        name = all_images[ind]
        path = file_path(name)
        image = cv2.imread(path)
        image = cv2.resize(image, dsize=shape)
        batch.append(image)



        lab_img = rgb2lab(image)
        hsv_img = rgb2hsv(image)
        gra_img = rgb2gray(image)
        cv2.imshow('Real', image)
        cv2.imshow('Lab', lab_img)
        cv2.imshow('HSV', hsv_img)
        cv2.imshow('Gray', gra_img)
        cv2.waitKey(0)

    batch = np.array(batch)
    return batch