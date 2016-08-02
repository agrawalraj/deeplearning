# Author: Raj Agrawal 

# Helper functions to randomly rotate multiple frames (or just 2d images) 
# in the minibatch samples  

# References: Taken mostly from https://jessesw.com/Deep-Learning/

# Note: TODO LATER - RIGHT NOW ONLY SUPPORTS GREYSCALE IMAGES 

from scipy.ndimage import convolve, rotate
import numpy as np

def random_image_generator(image_stack):
    """
    Overview: 
        This function randomly translates and rotates multiple 
        image frames (all in same direction), producing a new, 
        altered version as output. 
    ----------
    image_stack: numpy array    
        A numpy array of shape (1, num_frames, length, width) 

    Returns
    -------
    new_image: numpy array  
        A numpy array of shape (1, num_frames, length, width) 
        that is the randomly rotated version of 'image_stack'
    """
    num_frames = image_stack.shape[1]
    length = image_stack.shape[2]
    width = image_stack.shape[3]
    # Create our movement vectors for translation first. 
        
    move_up = [[0, 1, 0],
               [0, 0, 0],
               [0, 0, 0]]
        
    move_left = [[0, 0, 0],
                 [1, 0, 0],
                 [0, 0, 0]]
        
    move_right = [[0, 0, 0],
                  [0, 0, 1],
                  [0, 0, 0]]
                                   
    move_down = [[0, 0, 0],
                 [0, 0, 0],
                 [0, 1, 0]]
        
    # Create a dict to store these directions in.
        
    dir_dict = {1:move_up, 2:move_left, 3:move_right, 4:move_down}
        
    # Pick a random direction to move.
        
    direction = dir_dict[np.random.randint(1,5)]
        
    # Pick a random angle to rotate (30 degrees clockwise to 30 degrees counter-clockwise).
        
    angle = np.random.randint(-30,31)
        
    # Move the random direction and change the pixel data back to a 2D shape.
    new_image = np.zeros(shape=(1, num_frames,length, width))
    for i, image in enumerate(image_stack[0, :, :, :]):
        moved = convolve(image.reshape(length,width), direction, mode = 'constant')
        # Rotate the image
        rotated = rotate(moved, angle, reshape = False)
        new_image[0, i, :, :] = rotated
    return new_image

def random_2D_image_generator(image):
    """
    Overview: 
        This function randomly rotates an image  
    ----------
    image: numpy array    
        A numpy array of shape (length, width) 

    Returns
    -------
    new_image: numpy array  
        A numpy array of shape (1, length, width) 
        that is the randomly rotated version of 'image' 
    """
    length = image.shape[1]
    width = image.shape[2]
    # Create our movement vectors for translation first. 
        
    move_up = [[0, 1, 0],
               [0, 0, 0],
               [0, 0, 0]]
        
    move_left = [[0, 0, 0],
                 [1, 0, 0],
                 [0, 0, 0]]
        
    move_right = [[0, 0, 0],
                  [0, 0, 1],
                  [0, 0, 0]]
                                   
    move_down = [[0, 0, 0],
                 [0, 0, 0],
                 [0, 1, 0]]
        
    # Create a dict to store these directions in.
        
    dir_dict = {1:move_up, 2:move_left, 3:move_right, 4:move_down}
        
    # Pick a random direction to move.
        
    direction = dir_dict[np.random.randint(1,5)]
        
    # Pick a random angle to rotate (30 degrees clockwise to 30 degrees counter-clockwise).
        
    angle = np.random.randint(-30,31)
        
    # Move the random direction and change the pixel data back to a 2D shape.
    moved = convolve(image.reshape(length,width), direction, mode = 'constant')

    # Rotate the image
    new_image = rotate(moved, angle, reshape = False)
    return new_image
