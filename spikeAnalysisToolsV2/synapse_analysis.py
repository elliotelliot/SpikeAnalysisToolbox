import numpy as np
import pandas as pd
from . import helper

class Synapse_Mask:
    """
    Class that provides masks for a given network
    """
    def __init__(self, network_architecture_info, synapses):
        self.n_exc = network_architecture_info["num_exc_neurons_per_layer"]
        self.n_inh = network_architecture_info["num_inh_neurons_per_layer"]
        self.total_per_layer = self.n_exc + self.n_inh
        self.n_layer = network_architecture_info["num_layers"]
        self.pre  = synapses.pre.values
        self.post = synapses.post.values

        self.post_layers = self.post // self.total_per_layer # for each synapse the layer id of the postsynamptic layer

        error =  (self.total_per_layer * self.n_layer) - max(np.max(self.pre), np.max(self.post))-1
        if (error < 0):
            raise ValueError("There is a synapse involving a neuron that should not exist.")
        elif (error > 0):
            raise ValueError("The last neuron is not part of any synapses")



    def exc_feed_forward(self):
        in_lower_layer = self.pre < (self.post_layers * self.total_per_layer)
        # pre_neuron_id < lowest id neuron for all neurons in the same layer as pre_id
        #assert(np.all((self.pre > (self.post_layers-1) * self.total_per_layer) == in_lower_layer)) # making sure they are not further down below
        return in_lower_layer

    def exc_feed_back(self):
        in_higher_layer = self.pre >= ((self.post_layers+1) * self.total_per_layer)
        # pre_id >= lowest neuron in next layer
        return in_higher_layer

    def exc_lateral(self):
        above_first_neuron_in_layer  = ((self.post_layers * self.total_per_layer) <= self.pre)
        below_first_neuron_in_next   =  (self.pre < (self.post_layers * self.total_per_layer + self.n_exc))
        pre_in_same_layer_exc =  above_first_neuron_in_layer & below_first_neuron_in_next

        post_in_same_layer_exc = self.post < (self.post_layers * self.total_per_layer + self.n_exc)

        return pre_in_same_layer_exc & post_in_same_layer_exc
        # higher id then the first of the layer, lower id then first_of_layer + n_exc neurons

    def inh_lateral(self):
        above_first_in = (self.post_layers * self.total_per_layer + self.n_exc) <= self.pre
        below_first_in_next = self.pre < (self.post_layers+1) * self.total_per_layer
        return above_first_in & below_first_in_next

    def exc_to_inhibitory(self):
        post_above_first_inh = (self.post_layers * self.total_per_layer + self.n_exc) <= self.post
        post_below_first_in_next_layer = self.post < (self.post_layers+1) * self.total_per_layer
        return post_above_first_inh & post_below_first_in_next_layer



def count_incoming_with_mask(synapses, mask):
    synapses_in_category_post_ids = synapses[mask].post.values

    post_ids, counts = np.unique(synapses_in_category_post_ids, return_counts=True)
    return pd.DataFrame(data={'ids':post_ids, 'synapse_count': counts})

def count_outgoing_with_mask(synapses, mask):
    synapses_in_category_pre_ids = synapses[mask].pre.values
    pre_ids, counts = np.unique(synapses_in_category_pre_ids, return_counts=True)
    return pd.DataFrame(data={'ids': pre_ids, 'synapse_count': counts})

def multiple_connections_histogram(synapses):
    """
    Calculates Count of synapses between pre and post pair
    :param synapses: with columns "pre" and "post"
    :return: pandas series (.values are the counts)
    """
    count_of_synapses = synapses.groupby(['pre', 'post']).size()
    return count_of_synapses


def neuron_stats_to_layer(neuron_info, input_layer_count, input_neurons_per_layer, network_info):
    """

    :param neuron_info: pandas dataframe with columns "ids", "synapse_count"
    :param input_layer_count: number of input layers (neurons with negative ids)
    :param input_neurons_per_layer: neurons per input layer
    :param network_info: the usual dict
    :return input, exc, inh: each is a numpy array of shape [n_layer, n_neurons]
    """
    # assuming the first neuron in neuron
    input_layer = list()

    input_neuron_ids = neuron_info[ neuron_info.ids < 0].ids.values - np.min(neuron_info.ids)
    input_neuron_values = neuron_info[ neuron_info.ids < 0].synapse_count

    input_collector = np.zeros((input_layer_count, input_neurons_per_layer))

    for inp_l in range(input_layer_count):
        mask_above_first = inp_l * input_neurons_per_layer <= input_neuron_ids
        below_last       = input_neuron_ids < (inp_l + 1 ) * input_neurons_per_layer

        mask = mask_above_first & below_last

        ids_for_this_layer = input_neuron_ids[mask] - inp_l * input_neurons_per_layer

        input_collector[inp_l, ids_for_this_layer] = input_neuron_values[mask]

    # now the normal layers
    normal_neuron_info = neuron_info[ neuron_info.ids >= 0]


    exc_collector = np.zeros((network_info["num_layers"], network_info["num_exc_neurons_per_layer"]))
    inh_collector = np.zeros((network_info["num_layers"], network_info["num_inh_neurons_per_layer"]))

    layerwise = helper.split_into_layers(normal_neuron_info, network_info)
    for l_id, layer in enumerate(layerwise):
        exc, inh = helper.split_exc_inh(layer, network_info)
        exc_collector[l_id, exc.ids] = exc.synapse_count.values
        inh_collector[l_id, inh.ids] = inh.synapse_count.values

    return input_collector, exc_collector, inh_collector


def receptive_field_of_neuron(pos, is_excitatory, synapses, network_info):
    """
    Given position of a neuron calculate its receptive field. i.e. density of neurons in the 'previous' layer
    mapping onto the one given by pos

    :param pos: tubple (layer, line, column)
    :param is_excitatory: flag true-> pos neuron is excitatory
    :param synapses: synapses should only contain synapses of one type!
    :param network_info:
    :return: numpy array of shape [lines, columns]-> number of connections from that neuron to the pos one
    """

    id = helper.position_to_id(pos, is_excitatory, network_info)

    pre_ids = synapses[synapses.post == id].pre.values

    return _synapse_endpoint_density(pre_ids, network_info)


def targets_of_neuron(pos, is_excitatory, synapses, network_info):
    """
    Given position of a neuron calculate the density of the neurons in the 'next' layer that the neuron given by
    pos maps onto

    :param pos: tubple (layer, line, column)
    :param is_excitatory: flag true-> pos neuron is excitatory
    :param synapses: synapses should only contain synapses of one type!
    :param network_info:
    :return: numpy array of shape [lines, columns]-> number of connections from that neuron to the pos one
    """

    id = helper.position_to_id(pos, is_excitatory, network_info)

    post_ids = synapses[synapses.pre == id].post.values

    return _synapse_endpoint_density(post_ids, network_info)


def _synapse_endpoint_density(target_ids, network_info):
    """
    aranges a set of target ids in a numpy array representing the layer. This is a generic function for calculating 'receptive fields'
    get the distribution of target_neurons maping onto a given neuron or the distribution of target_neurons that a specific neuron maps onto.

    :param target_ids:
    :param target_is_excitatory:
    :param network_info:
    :return:
    """

    target_positions = [helper.id_to_position(i, network_info) for i in target_ids]

    # determine layer in which the receptive field is
    target_layer = target_positions[0][1][0] # first target neuron, second value is the position -> (layer, line, column)
    target_is_exc = target_positions[0][0] # flag if the target layer (i.e. the one from wich the connections come) is excitatory

    if target_is_exc:
        n_per_relevant_layer = network_info["num_exc_neurons_per_layer"]
    else:
        n_per_relevant_layer = network_info["num_inh_neurons_per_layer"]

    side_length = helper.get_side_length(n_per_relevant_layer)

    receptive_field = np.zeros((side_length, side_length))

    for target_pos in target_positions:
        current_is_exc, current_pos = target_pos
        current_layer, current_line, current_column = current_pos
        if(current_is_exc != target_is_exc or current_layer != target_layer):
            raise ValueError("Not all presynaptic neurons are in the same layer")

        receptive_field[current_line, current_column] += 1

    return receptive_field


def receptive_field_of_neuron_input(pos, synapses, network_info, side_length, num_layers):
    """
    Given position of a neuron calculate its receptive field
    :param pos: tubple (layer, line, column)
    :param synapses: synapses should only contain synapses of one type!
    :param side_length: length of the input layer side
    :param num_layers: num of input layers
    :return: numpy array of shape [layer, lines, columns]-> number of connections from that neuron to the pos one
    """
    assert(pos[0] == 0)

    id = helper.position_to_id(pos, is_excitatory=True, network_info=network_info) # only excitatory neurons receive input

    pre_ids = synapses[synapses.post == id].pre.values

    assert(np.all(pre_ids < 0))

    pre_positions = [helper.id_to_position_input(i, side_length=side_length, n_layer=num_layers) for i in pre_ids]

    receptive_field = np.zeros((num_layers, side_length, side_length))

    for pre_pos in pre_positions:
        current_layer, current_line, current_column = pre_pos

        receptive_field[current_layer, current_line, current_column] += 1

    return receptive_field





def incoming_synapses_of_all_types(synapses, network_architekture):
    """

    :param synapses:
    :param network_architekture:
    :return: dict each field is numpy array of shape [layer, neuron_id]
    """

    synapse_count_tensor = lambda tmp_mask: helper.neuron_target_column_to_numpy_array(count_incoming_with_mask(synapses, tmp_mask), "synapse_count", network_architekture)

    mask = Synapse_Mask(network_architekture, synapses)

    exc_FF = synapse_count_tensor(mask.exc_feed_forward())
    exc_FB = synapse_count_tensor(mask.exc_feed_back())
    exc_L = synapse_count_tensor(mask.exc_lateral())
    inh_L = synapse_count_tensor(mask.inh_lateral())
    exc_to_inh = synapse_count_tensor(mask.exc_to_inhibitory())

    return dict(exc_FF=exc_FF, exc_FB=exc_FB, exc_L=exc_L, inh_L=inh_L, exc_to_inh=exc_to_inh)


def outgoing_synapses_of_all_types(synapses, network_architekture):


    synapse_count_tensor = lambda tmp_mask: helper.neuron_target_column_to_numpy_array(count_incoming_with_mask(synapses, tmp_mask), "synapse_count", network_architekture)

    mask = Synapse_Mask(network_architekture, synapses)

    exc_FF = synapse_count_tensor(mask.exc_feed_forward())
    exc_FB = synapse_count_tensor(mask.exc_feed_back())
    exc_L = synapse_count_tensor(mask.exc_lateral())
    inh_L = synapse_count_tensor(mask.inh_lateral())
    exc_to_inh = synapse_count_tensor(mask.exc_to_inhibitory())

    return dict(exc_FF=exc_FF, exc_FB=exc_FB, exc_L=exc_L, inh_L=inh_L, exc_to_inh=exc_to_inh)



