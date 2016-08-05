# Author: Raj Agrawal 

# In this script, I have different functions that will allow you to:
# - Vary learning rates layer by layer 
# - Making a custom learning rate decay function 

def learning_rates_by_layer(network, layer_learn_dic, loss_update):
	for layer, learning_rate in layer_learn_dic.items():
		updates.update(loss_update(loss, layer.get_params, learning_rate)

def learning_decay_logic(init, errors, epoch):

