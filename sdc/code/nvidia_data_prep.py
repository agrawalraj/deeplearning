
# Author: Raj Agrawal 

# The following script reads in all images stored in a root directory and converts 
# the images to an array.

from __future__ import division 

import os
import glob
import numpy as np
import pandas as pd 
import scipy.misc
import random

from scipy.misc import imread 
from scipy.ndimage.interpolation import zoom
from ..load_comma_data import * 

def readImage(path, reduction_factor=.4):
    """
    Overview: 
        Reduces photo size by a factor of 1/reduction_factor 
        and converts to greyscale  
    ----------
    path: string   
        Path where the image is is located. 
    
    reduction_factor: float
        Defaults to .15. reduction_factor > 1 enlarges the photo and 
        reduction_factor < 1 shrinks the photo. 
    Returns
    -------
    reduced_image: numpy array 
        Grayscaled and resized image   
    """
    grey_image = imread(path, flatten=True)
    return zoom(grey_image, reduction_factor)

def to3DMatrix(paths, imgsize=(102, 182), reduction_factor=.4):
    """
    Overview: 
        Reduces images size by a factor of 1/reduction_factor 
        and converts to greyscale and stores in an array of
        shape (len(paths), length, width).  
    ----------
    paths: list   
        List consisting of paths where the images are located. See  
    imgsize: tuple
        The (length,width) of image and defaults to (81, 144). 
    
    reduction_factor: float
        Defaults to .15. reduction_factor > 1 enlarges the photo and 
        reduction_factor < 1 shrinks the photo. 
    Returns
    -------
    images_by_time: numpy array 
        Grayscaled and resized images of shape 
        (len(paths), length, width) 
    """
    num_images = len(paths)
    images_by_time = np.zeros(shape=(num_images, imgsize[0], imgsize[1])) 
    for i, path in enumerate(paths):
        image = readImage(path, reduction_factor)
        images_by_time[i, :, :] = image
        print('Finished Processing image ' + str(i))
    return images_by_time

def toMatrix(paths, num_frames, imgsize=(102, 182), multiple=4, reduction_factor=.4):
    """
    Overview: 
        Reduces images size by a factor of 1/reduction_factor 
        and converts to greyscale. Each 'sample' is of shape 
        num_frames x imgsize. This aggregates all samples into 
        a numpy array. 
    ----------
    paths: list   
        List consisting of paths where the images are located. See 
        ** Note ** below 
    num_frames: int 
        The number of images in one sample
    imgsize: tuple
        The (length,width) of image and defaults to (81, 144). 
    
    reduction_factor: float
        Defaults to .15. reduction_factor > 1 enlarges the photo and 
        reduction_factor < 1 shrinks the photo. 
    Returns
    -------
    images_by_time: numpy array 
        Grayscaled and resized images of shape 
        (num_samples, num_frames, length, width)
    Note: 
        This ASSUMES that the paths are in the right order. In other words, 
        if paths = [im1, im2, im3, im4] and num_frames = 2 this means sample
        1 is [im1, im2] and sample 2 is [im3, im4].   
    """
    num_images = len(paths)
    num_samples = int(num_images / num_frames)
    effective_num_frames = int(num_frames / multiple) + 1 
    images_by_time = np.zeros(shape=(num_samples, effective_num_frames, imgsize[0], imgsize[1])) 
    for sample_index in range(num_samples):
        index_in_array = sample_index * num_frames
        sample_paths = paths[index_in_array:(num_frames + index_in_array)]
        sample_paths = sample_paths[::multiple] 
        sample = to3DMatrix(sample_paths, imgsize, reduction_factor)
        images_by_time[sample_index, :, :, :] = sample 
        print('Finished Processing Sample ' + str(sample_index))
    return images_by_time

def makeOrderedPaths(folder_root, num_pics):
    """
    Returns the paths of all files in the folder_root but keeping frames in 
    temporal order 
    """
    paths = [folder_root + '/' + str(i) + '.jpg' for i in range(1, num_pics + 1)]
    return paths

if __name__ == '__main__':

    path_to_images = './driving_dataset'
    path_to_lables = './driving_dataset/data.txt'
    number_frames = 10

    # Read in video data/labels 
    num_pics = 45400
    paths = makeOrderedPaths(path_to_images, num_pics)
    images_by_time = toMatrix(paths=paths, num_frames=number_frames, imgsize=(102, 182), 
                              multiple=4, reduction_factor=.4)
    labels = []
    with open("driving_dataset/data.txt") as f:
        for i, line in enumerate(f):
            if i % number_frames == 0 and i > 0:
                #the paper by Nvidia uses the inverse of the turning radius,
                #but steering wheel angle is proportional to the inverse of turning radius
                #so the steering wheel angle in radians is used as the output
                labels.append(float(line.split()[1]) * scipy.pi / 180)

    # Shuffle data
    # Check to make sure this matches images_by_time --> might need to pad ends w/ extra labels 
    # num_samples = images_by_time.shape[0]
    # indcs = np.arange(num_samples)
    # np.random.shuffle(indcs)
    # images_by_time = images_by_time[indcs]
    # labels = labels[indcs]

    # Save in data folder 
    np.save('./images_by_time_mat', images_by_time)
    np.save('./labels', labels)
