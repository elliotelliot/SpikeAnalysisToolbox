import numpy as np
from numba import jit

from . import firing_rates as firing

def old_average_response(responses, list_of_objects):
    """
    Combine Stimuli by averaging the responses with a flat prior
    :param response: numpy array of dimensions [stimulus, layer, neuron_id]
    :param list_of_objects: list of lists containing the ids of stimuli that belong to one object [objectID]->list of stimulus ids
    :return: numpy array of shape [objects, layer, neuron_ID] -> average response of neuron for the OBJECT
    """
    prev_shape = responses.shape
    new_shape = (len(list_of_objects),) + prev_shape[1:]

    object_responses = np.empty(new_shape)

    for i, object in enumerate(list_of_objects):
        object_responses[i, :, :] = np.mean(responses[object, :, :], axis=0)

    return object_responses

def average_responses(responses, list_of_objects, axis=0):
    """
    Combine Stimuli by averaging the responses with a flat prior
    :param response: numpy array of dimensions [stimulus, layer, neuron_id]
    :param list_of_objects: list of lists containing the ids of stimuli that belong to one object [objectID]->list of stimulus ids
    :param axis: index of axis along which the objects are
    :return: numpy array of shape [objects, layer, neuron_ID] -> average response of neuron for the OBJECT
    """
    # prev_shape = responses.shape
    # new_shape = prev_shape
    # new_shape[axis] = len(list_of_objects)
    assert(responses.shape[axis] == np.sum([len(obj) for obj in list_of_objects]))

    object_responses = list()

    for i, object in enumerate(list_of_objects):
        object_response = np.mean(np.take(responses, object, axis=axis), axis=axis)
        object_responses.append(np.expand_dims(object_response, axis=axis))

    return np.concatenate(object_responses, axis=axis)

def min_responses(responses, list_of_objects):
    """
    Combine Stimuli into objects by giving for each of the objects transform the minimum response of the neuron
    given that that neuron has a higher firing rate for this object then normally. Only neurons have value >0 that have a higher
    firing rate to ALL transforms of the given stimulus.


    :param response: numpy array of dimensions [stimulus, layer, neuron_id], could be firing rate
    :param list_of_objects: list of lists containing the ids of stimuli that belong to one object [objectID]->list of stimulus ids
    :return: numpy array of shape [objects, layer, neuron_ID] -> min response of neuron for the OBJECT
    """
    prev_shape = responses.shape
    new_shape = (len(list_of_objects),) + prev_shape[1:]


    object_responses = np.empty(new_shape)

    for i, object in enumerate(list_of_objects):
        object_responses[i, :, :] = np.min(responses[object, :, :], axis=0)

    object_responses[object_responses < 0] = 0

    return object_responses



# @jit(cache=True)
def response_freq_table(firing_rates, objects, n_bins=3, allow_nan_as_seperate_bin=False):
    """
    Combine multiple presenations (stimuli) of the same objects into a frequency table that gives you
    the probability of a certain response given an object (which might be present in one of its different transforms)

    :param firing_rates: numpy array of shape [stimuli, layer, neurons]
    :param objects: list of lists each is the set of stimulus_ids that belong to one object
    :param n_bins: how many different response types a neuron can have
    :return: numpy array of shape [object_id, layer, neuron, response_type] -> probability p(response = response_type | object = object_id)
    """

    digitized_fr = firing.digitize_firing_rates_with_equispaced_bins(firing_rates, n_bins=n_bins, allow_nan_as_seperate_bin=allow_nan_as_seperate_bin)

    n_stimuli, n_layer, n_neurons = firing_rates.shape

    freq_table = np.empty((len(objects), n_layer, n_neurons, n_bins), dtype=float)

    for object_id, belonging_stimuli in enumerate(objects):
        for l in range(n_layer):
            for n in range(n_neurons):
                freq_table[object_id, l, n, :] = np.bincount(digitized_fr[belonging_stimuli, l, n], minlength=n_bins).astype(float)
                # print(freq_table[object_id, l, n, :])
                # on the right side we have a 1D array with one value for each stimulus
                # that gets counted into as many bins as we have so we have (n_bins)

        freq_table[object_id, :, :, :] /= len(belonging_stimuli)

    assert(np.all(np.isclose(np.sum(freq_table, axis=3), 1)))

    return freq_table
