import sys
sys.path.append("/Users/dev/Documents/Gisi/02_Analysis/03_CodeKrams/04_HackSunday/SpikeTrainAnalysis")
import SpikeDataLoading as data

import numpy as np
import pandas as pd

def subfolder_pre_post(masterpath, subfolders, extensions):
	subfolders_pre = list()
	subfolders_post = list()
	subfolders_delay = list()
	subfolders_init_weights = list()
	subfolders_weights = list()

	for subfolder in range(len(subfolders)):
		tmp1 = str(np.asarray(subfolders[subfolder]))
		print(tmp1)
		extension_pre = list()
		extension_post = list()
		extension_delay = list()
		extension_init_weights = list()
		extension_weights = list()
		
		for extension in range(len(extensions)):       
			tmp2 = str(np.asarray(extensions[extension]))
			path = masterpath + tmp1 + tmp2
			pre_tmp, post_tmp, delays_tmp, init_weights_tmp, weights_tmp = data.load_network(path, True, False)
			
			extension_pre.append(pre_tmp)
			extension_post.append(post_tmp)
			extension_delay.append(delays_tmp)
			extension_init_weights.append(init_weights_tmp)
			extension_weights.append(weights_tmp)
		subfolders_pre.append(extension_pre)
		subfolders_post.append(extension_post)
		subfolders_delay.append(extension_delay)
		subfolders_init_weights.append(extension_init_weights)
		subfolders_weights.append(extension_weights)

		return subfolders_pre, subfolders_post, subfolders_delay, subfolders_init_weights, subfolders_weights



def set_layer_bools(pre, info_neurons):

	exc_bool = list()
	inh_bool = list()

	num_exci = info_neurons[0]
	num_inhi = info_neurons[1]
	layers = info_neurons[2]
	num_neu_layer = info_neurons[3]

	for layer in range(layers):
		tmp_exc_bool = (pre >=  num_neu_layer*layer)  & (pre < (layer+1) * num_exci)
		tmp_inh_bool = (pre >=  ((layer + 1)  * num_exci + layer * num_inhi) & (pre < (layer+1) * num_neu_layer)) 

		exc_bool.append(tmp_exc_bool)
		inh_bool.append(tmp_inh_bool)
		
	return exc_bool, inh_bool


def subfolder_weights_calc(subfolders, extensions, subfolders_pre, subfolders_weights, info_neurons):
	subfolder_exc = list()
	subfolder_inh = list()
	layers = info_neurons[2]

	for subfolder in range(len(subfolders)):
		extension_exc = list()
		extension_inh = list()
	
		for extension in range(len(extensions)):
		
			current_weights = subfolders_weights[subfolder][extension]
		
			pre = subfolders_pre[subfolder][extension] 
			exc_bool, inh_bool = set_layer_bools(pre, info_neurons)
		
			layer_exc = list()
			layer_inh= list()
		
			for layer in range(layers):
				layer_exc_tmp = current_weights[exc_bool[layer]]
				layer_inh_tmp = current_weights[inh_bool[layer]]
				layer_exc.append(layer_exc_tmp)
				layer_inh.append(layer_inh_tmp)            
			extension_exc.append(layer_exc)
			extension_inh.append(layer_inh)
		subfolder_exc.append(extension_exc)
		subfolder_inh.append(extension_inh)
	return subfolder_exc, subfolder_inh


def set_only_FF_bool(pre, post, info_neurons):
	num_exci = info_neurons[0]
	num_inhi = info_neurons[1]
	layers = info_neurons[2]
	num_neu_layer = info_neurons[3]

	L_L_FF_exc_bool = list()
	L_L_FF_inh_bool = list()
	for layer in range(layers-1):
		
		lower_layer_exc_bool = (pre >= layer * num_neu_layer) & (pre < (layer+1) * num_exci)
		upper_layer_exc_bool = (post >= (layer+1) * num_neu_layer) & (post < (layer+2) * num_exci)
		
		lower_layer_inh_bool = (pre >= num_exci * (layer+1) + num_inhi * layer) & (pre < (layer+1) * num_neu_layer)
		upper_layer_inh_bool = (pre >= num_exci * (layer+2) + num_inhi * (layer+1) & (pre < (layer+2) * num_neu_layer))
		
		L_L_FF_exc_bool_tmp =  lower_layer_exc_bool & upper_layer_exc_bool
		L_L_FF_exc_bool.append(L_L_FF_exc_bool_tmp)
		
		L_L_FF_inh_bool_tmp =  lower_layer_inh_bool & upper_layer_inh_bool
		L_L_FF_inh_bool.append(L_L_FF_exc_bool_tmp)        
		
	return L_L_FF_exc_bool, L_L_FF_inh_bool

def set_only_FF(subfolders_pre, subfolders_post, subfolders_weights, subfolders, extensions, info_neurons):
	subfolder_exc_FF = list()
	subfolder_inh_FF = list()
	layers = info_neurons[2]

	for subfolder in range(len(subfolders)):
		extension_exc_FF = list()
		extension_inh_FF = list()
	
		for extension in range(len(extensions)):
		
			current_weights = subfolders_weights[subfolder][extension]
		
			pre = subfolders_pre[subfolder][extension]
			post = subfolders_post[subfolder][extension]
			FF_exc_bool, FF_inh_bool = set_only_FF_bool(pre, post, info_neurons)
		
			layer_exc_FF = list()
			layer_inh_FF = list()
		
			for layer in range(layers-1):
				layer_exc_tmp = current_weights[FF_exc_bool[layer]]
				layer_inh_tmp = current_weights[FF_inh_bool[layer]]
				layer_exc_FF.append(layer_exc_tmp)
				layer_inh_FF.append(layer_inh_tmp)
			
			extension_exc_FF.append(layer_exc_FF)
			extension_inh_FF.append(layer_inh_FF)
		subfolder_exc_FF.append(extension_exc_FF)
		subfolder_inh_FF.append(extension_inh_FF)
	return subfolder_exc_FF, subfolder_inh_FF









		