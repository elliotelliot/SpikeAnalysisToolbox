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
    each is a numpy array of shape [stimulus, layer, neuron_id] -> integer firing rate

"""
def stimulus_layer_nested_list_2_numpy_tensor(all_stimuli_rates):
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


