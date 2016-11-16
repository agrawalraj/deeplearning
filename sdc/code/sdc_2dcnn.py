# Author: Raj Agrawal 

# Builds a 2D spatio-temporal convolutional neural network to predict 
# steering angle from a video stream  

# Quick Architectural Overview:
# - 3 convolutional layers (ReLu, Dropout, MaxPooling), 4 dense layers
# - Squared Loss   
# - Nesterov-Momentum update  
# - Early Stopping

# References: See 'Papers' folder. 
# Some code taken from http://lasagne.readthedocs.io/en/latest/user/tutorial.html 

from __future__ import division 

import sys
import os
import lasagne
import theano
import numpy as np
import cPickle as pickle
import h5py 

import lasagne
import theano
import theano.tensor as T
from theano.tensor import *

from lasagne.nonlinearities import rectify
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer, BatchNormLayer
from lasagne.updates import adam, nesterov_momentum
from lasagne import layers

from random_image_generator import * 
from load_comma_data import *
from training_helper_fns import * 

def build_cnn(input_var, dim1, dim2):
    """
    Overview:
        Builds 2D CNN model
    ----------
    input_var: Theano Tensor 
        For our architecture this should be set to  
        TensorType('float32', (False,)*4).
    
    Returns
    -------
    dict
        A dictionary containing the network layers, where the output layer 
        is at key 'output'
    """
    net = {}
    net['input'] = InputLayer((None, 3, dim1, dim2), input_var=input_var)
    net['norm'] = BatchNormLayer(net['input'])

    # ----------- Conv layer group ---------------
    net['conv1a'] = Conv2DDNNLayer(net['norm'], 24, (5,5), stride=(2, 2), nonlinearity=rectify,flip_filters=False)
    
    net['conv2a'] = Conv2DDNNLayer(net['conv1a'], 36, (5,5), stride=(2, 2), nonlinearity=rectify)

    net['conv3a'] = Conv2DDNNLayer(net['conv2a'], 48, (3,3), stride=(2, 2), nonlinearity=rectify)

    net['conv4a'] = Conv2DDNNLayer(net['conv3a'], 64, (3,3), stride=(2, 2), nonlinearity=rectify)

    net['conv5a'] = Conv2DDNNLayer(net['conv4a'], 64, (3,3), nonlinearity=rectify)
    
    # ----------------- Dense Layers -----------------
    net['fc1']  = DenseLayer(net['conv5a'], num_units=500, nonlinearity=rectify)
    net['drop1'] = DropoutLayer(net['fc1'], p=.5)
    net['fc2']  = DenseLayer(net['drop1'], num_units=100, nonlinearity=rectify)
    net['drop2'] = DropoutLayer(net['fc2'], p=.5)
    net['fc3']  = DenseLayer(net['drop2'], num_units=50, nonlinearity=rectify)
    net['fc4']  = DenseLayer(net['fc3'], num_units=10, nonlinearity=rectify)
    net['output']  = DenseLayer(net['fc4'], num_units=1, nonlinearity=None)
    return net

if __name__ == '__main__':

	# TODO LOOK AT DATA BEFORE SPLITTING 

    all_paths_train = ['../data/camera/2016-01-30--11-24-51.h5', '../data/camera/2016-02-08--14-56-28.h5',  
    '../data/camera/2016-05-12--22-20-00.h5','../data/camera/2016-01-30--13-46-00.h5',  
    '../data/camera/2016-02-11--21-32-47.h5','../data/camera/2016-06-02--21-39-29.h5',
    '../data/camera/2016-01-31--19-19-25.h5', '../data/camera/2016-03-29--10-50-20.h5',  
    '../data/camera/2016-06-08--11-46-01.h5']

    paths_val = '../data/camera/2016-02-02--10-16-58.h5'
    paths_test = '../data/camera/2016-04-21--14-48-08.h5'

    # Might need to increase Python's recursion limit (I didn't need to)
    # sys.setrecursionlimit(10000)

    X_val, y_val, speed_val = load_data_label(paths_val)
    # X_val = X_val /= 255
    # X_val = X_val.astype(np.float32) #runs out of memory 
    y_val = y_val.astype(np.float32)

    # Fit model 
    dtensor5 = TensorType('float32', (False,)*4)
    input_var = dtensor5('inputs')
    target_var = T.fvector('targets')
    network = build_cnn(input_var, 160, 320)['output']

        # Create loss function
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

        # Create parameter update expressions (later I will make rates adaptive)
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = nesterov_momentum(loss, params, learning_rate=0.01,
    #                                         momentum=0.9)
    updates = adam(loss, params)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()
    #Delete this - same 
    test_acc = T.mean(lasagne.objectives.squared_error(test_prediction, target_var),
                  dtype=theano.config.floatX)

    # Compile training function that updates parameters and returns training loss
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    num_epochs = 8000 # Will probably not do this many b/c of early stopping 
    best_network_weights_epoch = 0 
    epoch_accuracies = [] 
    # Train network 
    # In each epoch, we do a full pass over the training data:
    for epoch in range(num_epochs):
    	for train_data_path in all_paths_train:
    		X_train, y_train, speed = load_data_label(train_data_path) #Y is angle 
    		y_train = y_train.astype(np.float32)

	        train_err = 0
	        train_batches = 0
	        for batch in iterate_minibatches2d(X_train, y_train, 16, shuffle=True):
	            inputs, targets = batch
	            train_err += train_fn(inputs, targets)
	            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches2d(X_val, y_val, 16, shuffle=False):#TODO FIX - ROTATIING VAL SET 
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Print the results for this epoch:
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f}".format(
            val_acc / val_batches * 100))
        print("Current Epoch = " + str(epoch))
        
        # Check if we are starting to overfit  
        if stop_early(val_acc, epoch_accuracies): 
            # Save best weights in models directory
            best_weight_path = '../data/weights/train/' + '2dcnn' + str(best_network_weights_epoch) + '.npz'
            os.rename(best_weight_path, '../data/weights/best' + '2dcnn' + str(best_network_weights_epoch) + '.npz')
            break   

        epoch_accuracies.append(val_acc)

        # Save weights every 100 epochs or if best weights.  
        save_weights(network, epoch, val_acc, epoch_accuracies) 

        # Update best weights  
        if val_acc >= np.max(epoch_accuracies):
            best_network_weights_epoch = epoch # This epoch is best so far  

    # Save Model (Not doing anymore - just use 'load_3dcnn_model' instead)
    # with open('../model/network.pickle', 'wb') as f:
    #     pickle.dump(network, f, -1)