# Author: Raj Agrawal 

# Builds a 3D spatio-temporal convolutional neural network to detect texting and 
# driving from a video stream 

# Quick Architectural Overview:
# - 3 convolutional layers (ReLu, Dropout, MaxPooling), 2 dense layers
# - Binary Cross-entropy 
# - Adam update  
# - Early Stopping

# Lots of code taken from http://lasagne.readthedocs.io/en/latest/user/tutorial.html 

from __future__ import division 

import sys
import lasagne
import theano
import numpy as np
import cPickle as pickle

import lasagne
import theano
import theano.tensor as T
from theano.tensor import *

from lasagne.nonlinearities import softmax, rectify, sigmoid
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from lasagne.objectives import binary_crossentropy, binary_hinge_loss
from lasagne.updates import adam
from lasagne import layers

from random_image_generator import * 

def build_cnn(input_var):
    """
    Builds 3D spatio-temporal CNN model
    Returns
    -------
    dict
        A dictionary containing the network layers, where the output layer is at key 'output'
    """
    net = {}
    net['input'] = InputLayer((None, 1, 10, 81, 144), input_var=input_var)

    # ----------- 1st Conv layer group ---------------
    net['conv1a'] = Conv3DDNNLayer(net['input'], 8, (3,3,3), nonlinearity=rectify,flip_filters=False)
    net['pool1']  = MaxPool3DDNNLayer(net['conv1a'],pool_size=(1,2,2))
    net['dropout1'] = DropoutLayer(net['pool1'], p=.1)

    # ------------- 2nd Conv layer group --------------
    net['conv2a'] = Conv3DDNNLayer(net['dropout1'], 16, (3,3,3), nonlinearity=rectify)
    net['pool2']  = MaxPool3DDNNLayer(net['conv2a'],pool_size=(2,2,2))
    net['dropout2'] = DropoutLayer(net['pool2'], p=.3)

    # ----------------- 3rd Conv layer group --------------
    net['conv3a'] = Conv3DDNNLayer(net['dropout2'], 32, (3,3,3), nonlinearity=rectify)
    net['pool3']  = MaxPool3DDNNLayer(net['conv3a'],pool_size=(1,2,2))
    net['dropout3'] = DropoutLayer(net['pool3'], p=.5)

    # ----------------- Dense Layers -----------------
    net['fc4']  = DenseLayer(net['dropout3'], num_units=500, nonlinearity=rectify)
    net['dropout4'] = DropoutLayer(net['fc4'], p=.5)
    net['fc5']  = DenseLayer(net['dropout4'], num_units=500, nonlinearity=rectify)

    # ----------------- Output Layer -----------------
    net['output']  = DenseLayer(net['fc5'], num_units=1, nonlinearity=None)

    return net

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
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
        rotate_indcs = swap_indcs[distorts_per_cat:(2*distorts_per_cat)]
        batch_sample_input[flip_indcs] = batch_sample_input[flip_indcs, :, :, :, ::-1] 
        for i in rotate_indcs:
            batch_sample_input[i, :, :, :, :] = random_image_generator(batch_sample_input[i, :, :, :, :])
        yield batch_sample_input, batch_sample_target

if __name__ == '__main__':

    #Large network, need to increase Python's recursion limit
    #sys.setrecursionlimit(10000)

    # Load data (did not standardize b/c images in 0-256)
    X = np.load('../data/train/images_by_time_mat.npy')
    X = X.astype(np.float32)

    # Only have 1 channel, need to reshape in order to match 5d required input
    X.shape = (3064, 1, 10, 81, 144) # FIX hardcoded - make general  
     
    Y = np.load('../data/train/labels.npy')

    # Shuffle data (already shuffled before. if not uncomment)
    # num_samps = X.shape[0]
    # indcs = np.arange(num_samps)
    # np.random.shuffle(indcs)
    # X = X[indcs]
    # Y = Y[indcs]

    # Convert Y into a binary vector 
    # 0 means nothing, 1 only driver text, 2 both text, 3 only passanger text
    Y[Y == 2] = 1 #1466 total 1s 
    Y[Y == 3] = 0 #1598 total 0s 
    Y = Y.astype(np.int32)

    # 85% train, 15% validation
    num_samps = 3064
    indcs = np.arange(num_samps)
    train_indcs = indcs[:2604]
    test_indcs = indcs[2604:]
    X_train, X_val = X[train_indcs], X[test_indcs]
    y_train, y_val = Y[train_indcs], Y[test_indcs] 

    # Delete X and Y from memory to save space
    X = None 
    Y = None 

    # Fit model 
    dtensor5 = TensorType('float32', (False,)*5)
    input_var = dtensor5('inputs')
    target_var = T.ivector('targets')
    # target_var = T.imatrix('targets')
    network = build_cnn(input_var)['output']

    # create loss function
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_hinge_loss(prediction, target_var)
    loss = loss.mean()

    # create parameter update expressions
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.001,
                                            momentum=0.99)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.binary_hinge_loss(test_prediction,
                                                        target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.sqn(test_prediction), target_var),
                  dtype=theano.config.floatX)

    # compile training function that updates parameters and returns training loss
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    #num_epochs = 10000
    num_epochs = 100 
    # train network 
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        #start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 16, shuffle=True):
            inputs, targets = batch
            # targets = [[0], [1], [1], [0], [0], [1], [0], [1], [0], [1], [1], [0], [1], [1], [1], [0]]
            train_err += train_fn(inputs, targets)
            print(train_err)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 16, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            print(val_err)
            val_acc += acc
            val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        #epoch + 1, num_epochs, time.time() - start_time))
        epoch + 1, num_epochs, 0))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))

        # stopEarly() logic 
        # Save weights logic 
