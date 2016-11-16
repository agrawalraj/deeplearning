
from __future__ import division 
import numpy as np 

# Utility functions to help train neural networks 

def iterate_minibatches2d(inputs, targets, batchsize, shuffle=False):
    """
    Overview: 
        An iterator that randomly rotates or flips 3/4 of the 
        samples in the minibatch. The remaining 1/4 of the samples
        are left the same. This is used for (minibatch) stochastic gradient 
        decent updating.
    ----------
    inputs: numpy array  
        This should be the training data of shape 
        (num_train, num_channels, length, width)
    
    targets: numpy array 
        This should be the corresponding labels of shape
        (num_train, )
    
    batchsize: int
        The number of samples in each minibatch 
    
    shuffle: 
        Defaults to false. If true, the training data is
        shuffled.
    Returns
    -------
    batch_sample_input: numpy array
        An array consisting of the minibatch data with some samples
        possibly flipped or randomly rotated. 
    
    batch_sample_target: numpy array
        The corresponding labels for the batch_sample_input
    """
    inputs = inputs.astype(np.float32) #Do conversion in batches - o/w runs out of memory 
    inputs /= 255
    inputs = inputs.astype(np.float32)
    num_samps = inputs.shape[0]
    indcs = np.arange(num_samps)
    if shuffle:
        np.random.shuffle(indcs)
    for i in range(0, num_samps - batchsize + 1, batchsize): 
        batch_indcs = indcs[i:(i + batchsize)]
        batch_sample_input = inputs[batch_indcs]
        batch_sample_target = targets[batch_indcs]

        # This handles random orientation logic
        num_changes = int(batchsize * .75) # Prop of samples we distort
        distorts_per_cat = int(num_changes / 2) # Of those we distort, flip half, rotate other half

        swap_indcs = np.random.choice(batchsize, num_changes, replace=False)
        flip_indcs = swap_indcs[0:distorts_per_cat]
        rotate_indcs = swap_indcs[distorts_per_cat:(2 * distorts_per_cat)]
        batch_sample_input[flip_indcs] = batch_sample_input[flip_indcs, :, :, ::-1] 
        for i in rotate_indcs:
            batch_sample_input[i, :, :, :] = random_2D_image_generator(batch_sample_input[i, :, :, :]) # !
        yield batch_sample_input, batch_sample_target

def stop_early(curr_val_acc, val_acc_list, patience=200):
    """
    Overview: 
        This implements the early stopping logic for training.  
    ----------
    curr_val_acc: float    
        The accuracy for the current epoch 
    val_acc_list: array    
        List of accuracies for past epochs
    patience: int   
        How many epochs to look back in order to compare 
        'curr_val_acc'  
    Returns
    -------
    boolean: True or False 
        If true this means that the network should halt. Otherwise, 
        the network should continue training.   
    """
    num_epochs = len(val_acc_list)
    if num_epochs < patience:
        return False 
    else:
        prev_acc = val_acc_list[num_epochs - patience]
        if prev_acc > curr_val_acc:
            print('Early Stopping')
            return True 
        else:
            return False

def save_network_weights(path, network):
    """
    Overview: 
        This saves the weights for each layer in the network
        in a .npz file  
    ----------
    path: string   
        Location of where to store the weights 
    network: Lasagne object
        The network from which the weights will be extracted from   
    Returns
    -------
    None  
    """
    np.savez(path, *lasagne.layers.get_all_param_values(network))

def save_weights(network, epoch, curr_val_acc, val_acc_list, name='2dcnn', multiple=100):
    """
    Overview: 
        This saves the weights for each layer in the network
        in a .npz file at multiples of 'multiple' or if the 
        curr_val_acc is the best accuracy so far. The weights are saved 
        in '../data/weights/train/NameEPOCH.npz' 
    ----------
    network: string   
        The network from which the weights will be extracted from 
    epoch: int
        The current epoch of training 
    curr_val_acc: 
        The accuracy for the current epoch
    
    val_acc_list:
        List of accuracies for past epochs
    multiple:
        Defaults at 100. Specifies the cycle time for 
        saving wieghts. 
    Returns
    -------
    None:
        Prints if it saves weights and specifies the epoch  
    """
    # Save weights every 20 epochs to server (transport to s3 eventually)
    if epoch % multiple == 0 or curr_val_acc > np.max(val_acc_list):
        weight_path = '../data/weights/train/' + name + str(epoch)
        save_network_weights(weight_path, network)
        print('Saved Weights for ' + str(epoch))
