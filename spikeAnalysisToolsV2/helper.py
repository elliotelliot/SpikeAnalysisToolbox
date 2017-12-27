import numpy as np
import pandas as pd
"""
Converts a long spike train into separate stimuli based upon a duration

Args:
    spikes: pandas DataFrame with ids and times
    stimduration: float indicating the length of time by which to split the stimuli

Returns:
    spikes_per_stimulus: a list of pandas data frames with spikes and times
"""
def splitstimuli(spikes, stimduration):
    assert ("ids" in spikes)
    assert ("times" in spikes)
    num_stimuli = int(np.ceil(np.max(spikes.times) / stimduration))

    spikes_per_stimulus = list()

    for i in range(num_stimuli):
        mask = (spikes.times > (i * stimduration)) & (spikes.times < ((i + 1) * stimduration))
        spikes_in_stim = spikes[mask].copy()
        spikes_in_stim.times -= (i * stimduration)
        spikes_per_stimulus.append(spikes_in_stim)

    return spikes_per_stimulus


"""
Takes a nested list with firing rates and arranges them in two numpy tensors (exc, inh)

Args: 
    all_stimuli_rates: nested list of shape [stimulus][layer][exc/inh] -> pandas dataframe with fields "ids", "firing_rate"

Returns: 
    (excitatory, inhibitory) 
    each is a numpy array of shape [stimulus, layer, neuron_id] -> firing rate value

"""
def nested_list_of_stimuli_2_np(all_stimuli_rates):
    n_stimuli = len(all_stimuli_rates)
    n_layer = len(all_stimuli_rates[0])
    n_neurons_exc = len(all_stimuli_rates[0][0][0])
    n_neurons_inh = len(all_stimuli_rates[0][0][1])
    excitatory_rates = np.empty((n_stimuli, n_layer, n_neurons_exc))
    inhibitory_rates = np.empty((n_stimuli, n_layer, n_neurons_inh))

    for stimulus in range(n_stimuli):
        for layer in range(n_layer):
            excitatory_rates[stimulus, layer, :] = all_stimuli_rates[stimulus][layer][0].sort_values('ids').firing_rates # sorting should be unnecessary
            inhibitory_rates[stimulus, layer, :] = all_stimuli_rates[stimulus][layer][1].sort_values('ids').firing_rates

    return excitatory_rates, inhibitory_rates


def nested_list_of_epochs_2_np(all_epoch_rates):
    """
    Convert nested list to two numpy arrays
    :param all_epoch_rates: nestd list of shape [epoch][stimulus][layer][exc/inh] -> pandas dataframe with fields "ids" firing_rate
    :return: exc, inh - each a numpy array of shape [epoch, stimulus, layer, nueron_id] -> firing rate value
    """
    list_of_np_arrays = [nested_list_of_stimuli_2_np(epoch) for epoch in all_epoch_rates]
    exc, inh = zip(*list_of_np_arrays)
    # they are wrong now
    exc_np = np.concatenate([np.expand_dims(e, 0) for e in exc], axis=0)
    inh_np = np.concatenate([np.expand_dims(i, 0) for i in inh], axis=0)
    return exc_np, inh_np


def neuron_target_column_to_numpy_array(data, target_column, network_architecture):
    """pandas dataframe with a column about each neuron to 2 numpy array with the values in that shape

    Args:
        data: pandas dataframe with columns "ids" and target_column
        target_column: string of the name for the target column
        network_architecture: dict with usual fields
    Returns:
        (exc, inh)
        each a numpy array of shape [layer, neuron_id]
    """
    neurons_per_layer = network_architecture["num_exc_neurons_per_layer"] + network_architecture["num_inh_neurons_per_layer"]
    result_exc = np.zeros(([network_architecture["num_layers"], network_architecture["num_exc_neurons_per_layer"]]))
    result_inh = np.zeros(([network_architecture["num_layers"], network_architecture["num_inh_neurons_per_layer"]]))

    layerwise = split_into_layers(data, network_architecture)
    for i, layer_data in enumerate(layerwise):
        exc, inh = split_exc_inh(layer_data, network_architecture)
        result_exc[i, exc.ids.values] = exc[target_column].values
        result_inh[i, inh.ids.values] = inh[target_column].values

    return result_exc, result_inh



def id_to_position(id, network_info, pos_as_2d=True):
    """
    given the id of a neuron it calculates its coordinates in the network
    :param id: global id of the neuron
    :param network_info: usual dict
    :param pos_as_2d: if True returned position is tuple [layer, row, column] if False tuple [layer, neuron_id]
    :return: is_it_excitatory? , tuple with position of neuron [layer, row, column] or [layer, neuron_id]
    """
    num_exc_neurons_per_layer = network_info["num_exc_neurons_per_layer"]
    num_inh_neurons_per_layer = network_info["num_inh_neurons_per_layer"]
    total_per_layer = num_exc_neurons_per_layer + num_inh_neurons_per_layer

    layer = id // total_per_layer
    id_within_layer = id - (layer * total_per_layer)

    exc_neuron = id_within_layer < num_exc_neurons_per_layer

    if exc_neuron:
        n_in_layer_type = num_exc_neurons_per_layer
    else:
        n_in_layer_type = num_inh_neurons_per_layer
        id_within_layer -= num_exc_neurons_per_layer

    if pos_as_2d:
        side_length = get_side_length(n_in_layer_type)

        x = id_within_layer // side_length
        y = id_within_layer % side_length

        return exc_neuron, (int(layer), int(y), int(x))
    else:
        return exc_neuron, (int(layer), int(id_within_layer))


def id_within_layer_to_pos(id, network_info, exc_neuron=True):
    """
    Calculate position of neuron with its layer
    :param id: tuple of shape (layer, neuron_id) or neuron_id (as int)
    :param network_info: usual dict
    :return: tupple of shape: (layer, row, column) or (row, column)
    """
    if len(id) == 2:
        neuron_id = id[1]
        layer = (id[0],)
    else:
        layer, neuron_id = tuple(), id

    num_exc_neurons_per_layer = network_info["num_exc_neurons_per_layer"]
    num_inh_neurons_per_layer = network_info["num_inh_neurons_per_layer"]

    if exc_neuron:
        n_in_layer_typ = num_exc_neurons_per_layer
    else:
        n_in_layer_typ = num_inh_neurons_per_layer

    side_length = get_side_length(n_in_layer_typ)

    x = neuron_id // side_length
    y = neuron_id % side_length

    result = (int(y), int(x))

    return layer + result





def position_to_id(pos, is_excitatory, network_info):
    """
    Calculate receptive field of a neuron
    :param pos: position of the neuron as a tuple [layer, line, column], or [layer, id_within_layer]
    :param is_excitatory: True -> excitatory neuron, False -> inhibitory neuron
    :param network_info: usual dict
    :return id: overall id of the neuron
    """
    if len(pos) == 3:
        layer, line, column = pos
        neuron_id = None # we cant calculate the neuron_id without knowing which layer type it is
    elif len(pos) == 2:
        layer, neuron_id = pos
    else:
        raise ValueError("pos does not have the right shape (tupple of length 2 or 3")

    num_exc_neurons_per_layer = network_info["num_exc_neurons_per_layer"]
    num_inh_neurons_per_layer = network_info["num_inh_neurons_per_layer"]
    total_per_layer = num_exc_neurons_per_layer + num_inh_neurons_per_layer

    first_in_layer_id = layer * total_per_layer

    if is_excitatory:
        n_in_layer_type = num_exc_neurons_per_layer
    else:
        n_in_layer_type = num_inh_neurons_per_layer
        first_in_layer_id += num_exc_neurons_per_layer

    side_length = np.sqrt(n_in_layer_type)

    if neuron_id is None:
        neuron_id = (column * side_length) + line

    if (side_length % 1 != 0):
        raise RuntimeError("The number of neurons ber layer is not a square number: {}".format(n_in_layer_type))

    id = first_in_layer_id + neuron_id

    return id

def get_side_length(n_in_layer_type):
    side_length = np.sqrt(n_in_layer_type)
    if (side_length % 1 != 0):
        raise RuntimeError("Tried to reshape something into square that wasn't actually a square number: {}".format(n_in_layer_type))
    return int(side_length)

def id_to_position_input(id, n_layer, side_length):
    """
    get input neuron coordinates
    :param id: id of the neuron
    :param n_layer: number of input layers
    :param side_length: length of the input layer grid
    :return: coordinates as (layer, y, x)
    """
    assert(id < 0)

    id = (-1 * id) - 1

    n_per_layer = side_length ** 2

    layer = id // n_per_layer

    id_within_layer = id - (n_per_layer * layer)

    x = id_within_layer // side_length
    y = id_within_layer % side_length

    return (layer, y, x)







"""
Splits layer into excitatory and inhibitory neurons

Args: 
    neuron_activity: pandas data frame with columnd "ids" the rest is arbitrary, only one layer
    network_architecture_info: dictionary with the fields: num_exc_neurons_per_layer, num_inh_neurons_per_layer

Returns:
    excitatory: containing only excitatory ones
    inhibitory: pandas data frame with same columns as neuron_activity containing only the inhibitory ones
"""
def split_exc_inh(neuron_activity, network_architecture_info):
    assert ('ids' in neuron_activity)
    num_exc_neurons_per_layer = network_architecture_info["num_exc_neurons_per_layer"]
    num_inh_neurons_per_layer = network_architecture_info["num_inh_neurons_per_layer"]
    total_per_layer = num_exc_neurons_per_layer + num_inh_neurons_per_layer

    excitatory = neuron_activity[neuron_activity.ids < num_exc_neurons_per_layer]
    inhibitory = neuron_activity[neuron_activity.ids >= num_exc_neurons_per_layer].copy()
    inhibitory.ids -= num_exc_neurons_per_layer

    return excitatory, inhibitory


"""
Divides the neuron activity into the different layers.
it is agnostic about which neuron information is saved in the table (e.g. spike timings or firing rates) 

Args:
    neuron_activity:  pandas data frame with a column "ids" 
    network_architecture_info: dictionary with the fields: num_exc_neurons_per_layer, num_inh_neurons_per_layer, num_layers
    
Returns:
    list of data frames each with columsn ids and whater it was before. (ids are reduced to start with 0 in each layer)
"""
def split_into_layers(neuron_activity, network_architecture_info):
    assert('ids' in neuron_activity)

    layerwise_activity = list()

    num_exc_neurons_per_layer = network_architecture_info["num_exc_neurons_per_layer"]
    num_inh_neurons_per_layer = network_architecture_info["num_inh_neurons_per_layer"]
    n_layers = network_architecture_info["num_layers"]
    total_per_layer = num_exc_neurons_per_layer + num_inh_neurons_per_layer

    for l in range(n_layers):
        mask = (neuron_activity.ids >= (l * total_per_layer)) & (neuron_activity.ids < ((l + 1) * total_per_layer))
        neurons_in_current_layer = neuron_activity[mask].copy()
        neurons_in_current_layer.loc[:, 'ids'] -= l * total_per_layer
        layerwise_activity.append(neurons_in_current_layer)

    return layerwise_activity



def _combine_spike_ids_and_times(ids, times):
    return pd.DataFrame({"ids": ids, "times": times})


def z_transform(data, axis=0):
    """
    z transform of the given data along the given axis
    :param data:
    :param axis: defaults to 0 which for data of shape [stimulus, layer, neuron_id] gives you the relative response for each stimulus
    :return:
    """
    mean = np.mean(data, axis=axis)
    sigma = np.std(data, axis=axis)

    transformed = (data - mean) / sigma

    return np.nan_to_num(transformed)



def reshape_into_2d(unshaped):
    """
    takes an arbitrary numpy array and replaces the last dimension with 2 dimensions of same length
    Args:
        unshaped: numpy array of shape [..., n_neurons]
    Returns:
        numpy array of shape [..., sqrt(n_neurons), sqrt(n_neurons]
    Raises:
        Exception if n_neurons is not a square
    """
    dimensions = unshaped.shape
    n_neurons = dimensions[-1]

    side_length = np.sqrt(n_neurons)
    if(side_length % 1 != 0):
        raise RuntimeError("The last dimension is not a square number: {}".format(n_neurons))

    side_length = int(side_length)


    return np.reshape(unshaped, dimensions[:-1] + (side_length, side_length), order="F")


def epoch_subfolders_to_tensor(all_epochs):
    """
    Converts a nested list of firing rates to 2 numpy arrays
    :param all_epochs: nested list of shape [epoch][stimulus][layer][excitatory/inhibitory] -> pandas dataframe with "ids" and "firing_rates"
    :return: exc_rates,
    """
    raise NotImplementedError("don't know what happend here")


def random_label(n_objects, n_transforms, n_repeats):
    """
    Create random label where each stimuli always is part of the same object througout multiple repeats
    :param n_objects: number of objects
    :param n_transforms: number of transforms per object
    :param n_repeats: how often is each stimulus presented. (assumed that all stimulus are presented before one is presented again)
    :return: array of indices belonging to an object
    """

    n_stimuli = n_objects * n_transforms

    one_presentation_of_all = np.random.choice(n_stimuli, size=(n_objects, n_transforms), replace=False)

    # repeat for multiple presentations
    all_presentations = [one_presentation_of_all + (r * n_stimuli) for r in range(n_repeats)]

    all_presentations_np = np.concatenate(all_presentations, axis=1)

    return [list(l) for l in all_presentations_np] # unpack the numpy into a list of lists



