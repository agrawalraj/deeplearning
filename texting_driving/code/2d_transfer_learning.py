
# Here we will load pretrained weights from the Lasagne Model Zoo  
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

# We will use the pretrained weights and strip the first two layers 

# Things to extend: TODO The learning weights in the first two layers should be
# smaller than the next convolutional and dense layers: See 
# https://github.com/Lasagne/Lasagne/issues/648 

# Since this takes 3 color channels, we need to modify the input data 
# to include 3 channels 

import cPickle as pickle

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax
from 2d_cnn_lasagne import iterate_minibatches, stop_early, # TEST FOR COLOR IMAGES
from 2d_cnn_lasagne import save_network_weights, save_weights # TEST FOR COLOR IMAGES 
from lasagne.layers import set_all_param_values

# The actual architecture of the model 
from vgg16 import build_model

def load_vgg16(path_to_pkl):
    net = build_model()
    output_layer = net['prob']
    with open(path_to_pkl, 'rb') as f:
        params = pickle.load(f)
    MEAN_IMAGE = params['mean value']
    set_all_param_values(output_layer, params['param values'])
    return (output_layer, MEAN_IMAGE)  

def strip_first_two_layers(output_layer):
    # 32 total trainable lasagne layers 
    layer_two = None
    layer_two = None 
    layer_three = None 
    for i in range(24):
        if i == 20:
            layer_three = output_layer
        if i == 22:
            layer_two = output_layer
        if i == 23:
            layer_one = output_layer
        output_layer = output_layer.input_layer
    return (layer_one, layer_two, layer_three)

def init_cnn_model(layer_one_params, layer_two_params, layer_three_params, input_var):
    net = {}
    W_one = layer_one_params[0]
    W_two = layer_two_params[0]
    W_three = layer_three_params[0]
    b_one = layer_one_params[1]
    b_two = layer_two_params[1]
    b_three = layer_three_params[1]
    net['input'] = InputLayer((None, 3, 81, 144), input_var=input_var)

    # ----------- 1st Conv layer group ---------------
    net['conv1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False, W=W_one.get_value(), b=b_one.get_value())
    net['conv2'] = ConvLayer(net['conv1'], 64, 3, pad=1, flip_filters=False, W=W_two.get_value(), b=b_two.get_value())
    net['pool1']  = MaxPool2DDNNLayer(net['conv2'], 2)

    # ------------- 2nd Conv layer group --------------
    net['conv3'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False, W=W_three.get_value(), b=b_three.get_value())
    net['pool1']  = MaxPool2DDNNLayer(net['conv2'],2)
    net['pool2']  = MaxPool2DDNNLayer(net['conv3'],pool_size=(2,2))
    net['dropout1'] = DropoutLayer(net['pool2'], p=.5)

    # ----------------- Dense Layers -----------------
    net['fc4']  = DenseLayer(net['dropout1'], num_units=256, nonlinearity=rectify)
    net['dropout4'] = DropoutLayer(net['fc4'], p=.5)
    net['fc5']  = DenseLayer(net['dropout4'], num_units=256, nonlinearity=rectify)

    # ----------------- Output Layer -----------------
    net['output']  = DenseLayer(net['fc5'], num_units=1, nonlinearity=None)

    return net

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Overview: 
        An iterator that randomly rotates or flips 3/4 of the 
        samples in the minibatch. The remaining 1/4 of the samples
        are left the same. This is used for (minibatch) stochastic gradient 
        decent updating.
    ----------
    inputs: numpy array  
        This should be the training data of shape 
        (num_train, num_frames, length, width)
    
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
        #for i in rotate_indcs:
            #batch_sample_input[i, :, :, :] = random_2D_image_generator(batch_sample_input[i, :, :, :]) # !
        yield batch_sample_input, batch_sample_target

if __name__ == 'main':
    # Might need to increase Python's recursion limit (I didn't need to)
    # sys.setrecursionlimit(10000)

    # Load data (did not standardize b/c images in 0-256)
    X = np.load('../data/train/images_by_time_mat_color.npy')
   
    # Just use 5th frame of each .5 second or 10 frame video sequence 
    X = X[:, :, 5, :, :] # ! -
    #X = X - MEAN_IMAGE
    X = X.astype(np.float32)
    
    Y = np.load('../data/train/labels_color.npy')

    # Convert Y into a binary vector 
    # 0 means nothing, 1 only driver text, 2 both text, 3 only passanger text
    Y[Y == 0] = -1
    Y[Y == 1] = 1
    Y[Y == 2] = 1 #1466 total 1s 
    Y[Y == 3] = -1 #1598 total -1s 
    Y = Y.astype(np.int32)

    # 85% train, 15% validation
    num_samps = 3064
    indcs = np.arange(num_samps)
    np.random.shuffle(indcs)
    train_indcs = indcs[:2604]
    test_indcs = indcs[2604:]
    X_train, X_val = X[train_indcs], X[test_indcs]
    y_train, y_val = Y[train_indcs], Y[test_indcs] 

    # Delete X and Y from memory to save disk space
    X = None 
    Y = None 

    # Fit model 
    dtensor4 = TensorType('float32', (False,)*4) # !
    input_var = dtensor4('inputs') # !
    target_var = T.ivector('targets')
    output_layer, MEAN_IMAGE = load_vgg16('vgg16.pkl')
    layer_one, layer_two, layer_three = strip_first_two_layers(output_layer)
    network = init_cnn_model(layer_one.get_params(), layer_two.get_params(), layer_three.get_params(), input_var)['output']

    # Create loss function
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_hinge_loss(prediction, target_var)
    loss = loss.mean()

    # Create parameter update expressions (later I will make rates adaptive)
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = nesterov_momentum(loss, params, learning_rate=0.003,
    #                                         momentum=0.99)
    updates = adam(loss, params)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = binary_hinge_loss(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.sgn(test_prediction), target_var),
                  dtype=theano.config.floatX)

    # Compile training function that updates parameters and returns training loss
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    num_epochs = 8000 # Will probably not do this many b/c of early stopping 
    best_network_weights_epoch = 0 
    epoch_accuracies = [] 
    # Train network 
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train, y_train, 16, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 16, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Print the results for this epoch:
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        print("Current Epoch = " + str(epoch))
        
        # Check if we are starting to overfit  
        if stop_early(val_acc, epoch_accuracies): 
            # Save best weights in models directory
            best_weight_path = '../data/train/weights/2dcnn' + str(best_network_weights_epoch) + '.npz'
            os.rename(best_weight_path, '../models/2d_cnn_' + str(best_network_weights_epoch) + '.npz')
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
