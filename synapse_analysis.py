import numpy as np
import pandas as pd
from numba import jit

from . import helper


class Synapse_Mask:
    """
    Class that provides masks for a given network
    Each function returns a boolean array that is true where the synapse is of the correct type
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

    def post_in_layer(self, l):
        return self.post_layers == l

    def from_input(self):
        return self.pre < 0




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


def neuron_stats_to_layer(neuron_info, input_layer_count, input_neurons_per_layer, network_info, target_column="synapse_count"):
    """
    Split synapse count information of neurons into layers. Pandas dataframe -> numpy array

    :param neuron_info: pandas dataframe with columns "ids", target_column
    :param input_layer_count: number of input layers (neurons with negative ids)
    :param input_neurons_per_layer: neurons per input layer
    :param network_info: the usual dict
    :return input, exc, inh: each is a numpy array of shape [n_layer, n_neurons]
    """
    # assuming the first neuron in neuron
    input_layer = list()

    input_neuron_ids = neuron_info[ neuron_info.ids < 0].ids.values - np.min(neuron_info.ids)
    input_neuron_values = neuron_info[ neuron_info.ids < 0][target_column]

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
        exc_collector[l_id, exc.ids] = exc[target_column].values
        inh_collector[l_id, inh.ids] = inh[target_column].values

    return input_collector, exc_collector, inh_collector


def weighted_presynaptic_actvity(mask, net, weights, firing_rates):
    """
    Calculates the sum over (presynaptic_FR * synapse_weight) for all synapses masked with 'mask'
    :param mask: np boolean array to select the synapses
    :param net: pandas dataframe with columns 'pre', 'post'
    :param weights: np array of shape [epochs, synapses] -> weight for that synapse at that epoch
    :param firing_rates: (exc_rates, inh_rates) -> each a numpy array of shape [epochs, objects, layer, neuron_id]
    :return: numpy array of length [n_epochs, object] -> weighted sum of presynaptic firing rates within each object
    """

    if not np.any(mask):
        raise ValueError("The mask is completely False (every entry is false)")

    exc_rates, inh_rates = firing_rates

    n_epochs, n_objects, n_layer, n_neurons_exc = exc_rates.shape
    assert((n_epochs, n_objects, n_layer) == inh_rates.shape[:3])
    n_neurons_inh = inh_rates.shape[-1]

    network_info = dict(
        num_exc_neurons_per_layer=n_neurons_exc,
        num_inh_neurons_per_layer=n_neurons_inh,
        num_layers=n_layer
    )

    pre_ids = net[mask].pre
    if(np.any(pre_ids < 0 )):
        raise ValueError("Some of the presynaptic neurons are input neurons. This would require a seperate function.")

    assert(len(pre_ids.shape) == 1)

    neuron_info = [helper.id_to_position(pre_id, network_info, pos_as_2d=False) for pre_id in pre_ids]
    # each entry has shape (is_excitatory, (layer, pos))

    is_excitatory, position = zip(*neuron_info)

    if np.all(is_excitatory):
        relevant_rates = exc_rates
    elif np.all(np.invert(is_excitatory)):
        relevant_rates = inh_rates
    else:
        raise ValueError("There are excitatory and inhibitory presynaptic neurons. Blindly summing over them does not seem to make sense")

    layer_id, neuron_id = zip(*position)


    presynaptic_firing_rates = relevant_rates[:, :, layer_id, neuron_id]

    weights = weights[:, mask]

    assert(presynaptic_firing_rates.shape[-1] == weights.shape[-1])
    assert(presynaptic_firing_rates.shape[0] == weights.shape[0])
    # but in presynaptic_firing_rates the second dimension is different (objects)
    # weights are the same for different object presentations (within epoch)
    weights_reshaped = np.repeat(np.expand_dims(weights, 1), n_objects, axis=1)


    weighted_fr = weights_reshaped * presynaptic_firing_rates


    return np.sum(weighted_fr, axis=2)


def paths_to_neurons(input_neurons, architecture, coefficient, max_path_length=1, maximum_neuron_id=0):
    """
    For each neuron compute a value that reflects how many paths reach this neuron from an active input neuron.

    $$n_i = \sum_{p \in AllPathsFromInputNeuronTo_n_i} c ^{|p|}$$

    :param input_neurons: list of source neuron ids
    :param architecture: pandas dataframe with columns pre and post
    :param coefficient: each path is weighted by coefficient^(path_length)
    :param maximum_neuron_id: highest neuron id (in case there are no synapses to this neuron, for example if it is inhibitory)
    :return:
    """

    all_neurons = np.unique(np.concatenate([architecture.pre.values, architecture.post.values]))

    minimum_id = np.min(all_neurons)
    maximum_id = np.max(all_neurons)


    # assert(len(all_neurons) == (maximum_id - minimum_id + 1))
    maximum_id = max(maximum_id, maximum_neuron_id)

    pre_ids = architecture.pre.values - minimum_id
    post_ids = architecture.post.values - minimum_id
    n_neurons = maximum_id - minimum_id + 1

    incoming_value = np.zeros(n_neurons) # pd.DataFrame({"value": np.zeros(n_neurons)}, index=all_neurons)
    incoming_value[np.array(input_neurons) - minimum_id] = 1

    collector = np.zeros(n_neurons)



    for l in range(max_path_length+1):
        # calculate value thats propagated along the synapses
        outgoing_value = coefficient * incoming_value

        # save the new paths that we just found
        collector += incoming_value
        incoming_value[:] = 0

        # propagate path along the synapses value n found at pos i
        incoming_value = np.bincount(post_ids, outgoing_value[pre_ids], minlength=n_neurons)
        # for each synapse (which has a post_id and a pre_id: incoming_value[post_id] += outgoing_value[pre_id]


    final_actual_ids = np.arange(n_neurons) + minimum_id

    return pd.DataFrame({'ids': final_actual_ids, 'path_values': collector}, index=final_actual_ids)






def shuffle_weights_within_layer(synapses, network_architecture, n_neurons_per_input_layer):
    """
    Shuffle weighs within each category of neurons within each layer
    :param synapses: pandas dataframe with 'pre', 'post', 'weights'
    :param network_architecture: usual dict
    :param n_neurons_per_input_layer: how many input neurons are in each input layer
    :return: pandas dataframe of with same columns as the original one, but weights are no shuffled
    """

    #check that n_neurons_per_input_layer is reasonable
    first_input_neuron = np.min(synapses.pre)
    assert(first_input_neuron % n_neurons_per_input_layer == 0)
    #checks wether the last input neuron is the last neuron of a layer with n_neurons_per_input_layer many neurons


    mask = Synapse_Mask(network_architecture, synapses)

    old_weights = synapses.weights.values
    new_weights = np.copy(old_weights)

    #first go through all inner synapses (not comming from input)
    synapse_types = [mask.exc_feed_forward(), mask.exc_feed_back()]

    not_input = np.invert(mask.from_input())

    for post_layer in range(network_architecture["num_layers"]):
        layer_selector = mask.post_in_layer(post_layer)
        for type in synapse_types:
            this_population_mask = not_input & layer_selector & type
            assert(this_population_mask.shape == not_input.shape)

            weights_this_population = old_weights[this_population_mask]

            new_weights[this_population_mask] = np.random.permutation(weights_this_population)


    # now for the input layers
    n_input_layers = -1 * (first_input_neuron // n_neurons_per_input_layer)
    print("{} many input layers".format(n_input_layers))

    input_layer_nr = (-1 * (synapses.pre.values + 1)) // n_neurons_per_input_layer # this will only give reasonable values for neurons that are actually in the input layer

    for inp_l in range(n_input_layers):
        is_in_that_layer = (input_layer_nr == inp_l)
        this_population_mask = is_in_that_layer & mask.from_input()

        assert (this_population_mask.shape == not_input.shape)

        weights_this_population = old_weights[this_population_mask]

        new_weights[this_population_mask] = np.random.permutation(weights_this_population)


    # check if the new weights are reasonable
    # assert(not np.any(np.isnan(new_weights))) # each
    assert(np.any(old_weights != new_weights)) #some changed

    synapses.weights = new_weights
    return synapses









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


def get_weight_development_random_sample(synapses, n_synapses):
    """
    Get matrix with random sample of synapses
    :param synapses: numpy array shape [epoch, synapse]
    :param n_synapses: count of synapses to look at
    :return: numpy array with dimensions [epoch, choosen_synapse] (choosen_synapse has size n_synapses)
    """
    n_epochs, n_possible_synapses = synapses.shape
    rand_indices = np.random.choice(n_possible_synapses, size=(n_synapses,), replace=False)
    return synapses[:, rand_indices]



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



class BackTracer:
    """Class to trace back synapses to V1"""

    def __init__(self, synapses, network_info, input_dim=128, n_input_layer=8, weights=None, percentage_thresholds=None, mask=None):
        """
        :param synapses: pandas data frame with at least pre and post ids
        :param weights: same shape as network. It will take the synapses with the strongest value her
        :param percentage_thresholds: top n% synapses are traced
        :param network_info:  dict(num_inh_neurons_per_layer=32 * 32, num_exc_neurons_per_layer=64 * 64, num_layers=4)

        """
        if weights is None:
            weights = synapses.weights.values

        if mask is not None:
            synapses = synapses[mask]
            weights = weights[mask]

        assert(len(synapses) == len(weights))
        self.pre = synapses.pre.values
        self.post = synapses.post.values
        self.weights = weights
        self.network_info = network_info
        self.percentage_thresholds_default = percentage_thresholds
        self.input_dim = input_dim
        self.n_input_layer = n_input_layer

    @jit
    def trace_back(self, layer, row, column, scaled_input=False, percentage_thresholds=None):
        """
        Trace back a neuron indexed by layer, row, column
        :param layer:
        :param row:
        :param column:
        :param scaled_input: False: returned matrix will be binary 1 if input neuron is connected to target 0 otherwise
        True: returned matrix is scalar. product of synapse weights on the path that connects this neuron to the target neuron
        :param percentage_thresholds: Optional if specified in the initializer
        :return: input_layer, numpy matrix of shape [input_layer, input_dim, input_dim]
        """

        if percentage_thresholds:
            self._percentage_thresholds = percentage_thresholds
        elif self.percentage_thresholds_default:
            self._percentage_thresholds = self.percentage_thresholds_default
        else:
            raise ValueError("A Percentage threshold has to be given either in the constructor or when calling this function")

        self._input_space = np.zeros((self.n_input_layer, self.input_dim, self.input_dim))

        target_neuron_id = helper.position_to_id((layer, row, column), True, self.network_info)

        if scaled_input:
            self._recursive_trace(target_neuron_id, 1)
        else:
            self._recursive_trace(target_neuron_id)

        result = self._input_space

        # tidy up temporary global variables
        self._input_space = None
        self._percentage_thresholds = None
        return result

    @jit
    def _recursive_trace(self, current_id, weight_product=None):
        """weight_product if it is None the result in self._input_space is just binary, i.e. this input neuron was connected to the target or not
        otherwise weight_product=1 it will be the product of all the synaptic weights on the path from the input neuron to the target.
        """
        # recursion anchor
        if current_id<0:
            # we reached an input neuron
            layer, row, column = helper.id_to_position_input(current_id, self.n_input_layer, self.input_dim)
            if weight_product is None:
                weight_product = 1

            self._input_space[layer, row, column] += weight_product

            return
        elif weight_product is not None and np.isclose(weight_product, 0):
            return
        else:

            # proceed backwards


            affereten_synapses_mask = self.post == current_id

            assert(np.any(affereten_synapses_mask))


            weights_afferent_synapses = self.weights[affereten_synapses_mask]
            pre_ids_afferent_synapses = self.pre[affereten_synapses_mask]

            _is_excitatory, (layer,_id_in_layer) = helper.id_to_position(current_id, self.network_info, pos_as_2d=False)
            assert(_is_excitatory)

            n_syn_to_trace = int(len(weights_afferent_synapses) * self._percentage_thresholds[layer])

            # unique_values, count = np.unique(weights_afferent_synapses, return_counts=True)
            # if np.max(count) > n_syn_to_trace:
            #     raise RuntimeError("There are {} synapses with exactly the weight {} ".format(np.max(count), unique_values[np.argmax(count)]))

            sorted_indices = np.argsort(weights_afferent_synapses)[::-1]

            for i in sorted_indices[:n_syn_to_trace]:
                weight_this_synapse = weights_afferent_synapses[i]
                current_pre_neuron = pre_ids_afferent_synapses[i]

                if weight_product is not None:
                    weight_product = weight_product * weight_this_synapse

                self._recursive_trace(current_pre_neuron, weight_product)
