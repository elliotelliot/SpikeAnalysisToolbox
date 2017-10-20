import numpy as np
import pandas as pd

# from helper import *
from . import helper
from . import firing_rates as firing
from . import data_loading as data
# from .. import data_loading as data

"""
computes the time course of the instantanious Firing rate for layers
Args:
    spikes
    network_architecture: dict containing
    time_step: the temporal resolution
Returns:
    excitatory_timecourse: numpy array with dimensions [layer, timepoint, neuronid]
    inhibitroy_timecourse: same shape

"""
def instant_FR_for_all_layers(spikes, network_architecture, time_step):
    n_neurons = (network_architecture["num_exc_neurons_per_layer"] + network_architecture["num_exc_neurons_per_layer"]) * network_architecture["num_layers"]
    #spikes_to_instantanious_FR(spikes, (0, n_neurons), time_step)

    t_start = np.min(spikes.times)
    t_end = np.max(spikes.times)

    all_inh_collector = list()
    all_exc_collector = list()

    layerwise = helper.split_into_layers(spikes, network_architecture)
    for layer in layerwise:
        exc_spikes, inh_spikes = helper.split_exc_inh(layer, network_architecture)
        time_exc, inh_InstantFR = spikes_to_instantanious_FR(inh_spikes, (0, network_architecture["num_inh_neurons_per_layer"]), time_step, (t_start, t_end))
        time_inh, exc_InstantFR = spikes_to_instantanious_FR(exc_spikes, (0, network_architecture["num_exc_neurons_per_layer"]), time_step, (t_start, t_end))

        assert(np.all(time_exc == time_inh))

        all_inh_collector.append(inh_InstantFR)
        all_exc_collector.append(exc_InstantFR)

    excitatory_timecourse = np.stack(all_exc_collector, axis=0)
    inhibitory_timecourse = np.stack(all_inh_collector, axis=0)

    return excitatory_timecourse, inhibitory_timecourse


"""
Convert spikes into an instantaneus firing rate
Args:
    spikes: pandas data frame with fields "ids" and "times"
    neuron_range: int tuple with start and end index
    time_step: the time resolution
    time_range: float tuple with the target time window

Returns:
    time: numpy array with the time steps
    instantaneus_firing: numpy array of dimensions [timepoint, neuron_id] giving the scalar firing rate at every timepoint

"""
def spikes_to_instantanious_FR(spikes, neuron_range, time_step, time_range=None):
    assert ('ids' in spikes)
    assert ('times' in spikes)

    neuron_start, neuron_end = neuron_range
    mask = (neuron_start <= spikes.ids) & (spikes.ids < neuron_end)

    if time_range:
        t_start, t_end = time_range
        mask = mask & ((t_start <= spikes.times) & (spikes.times <= t_end))
    else:
        t_start = np.min(spikes.times.values)
        t_end = np.max(spikes.times.values)

    t_start = int(np.floor(t_start * (1 / time_step)))
    t_end = int(np.ceil(t_end * (1 / time_step)))

    relevant_spikes = spikes[mask].copy()

    spike_times = relevant_spikes.times.values
    spike_ids = relevant_spikes.ids.values

    int_spike_times = np.floor((1 / time_step) * spike_times).astype(dtype=int)

    id_time_tuple_array = np.stack([spike_ids, int_spike_times], axis= 0)
    # shape (2, n_spikes) first row is the ids, second is the times
    #

    id_time_pairs, occurance_count = np.unique(id_time_tuple_array, return_counts=True, axis=1)
    # will count the number of occurances of the same columns (because axis 1) which represents a specific neuron spiking at a specific time

    ids_that_spiked = id_time_pairs[0, :]
    times_they_spiked = id_time_pairs[1, :]
    count_they_spiked = occurance_count


    instantanious_firing = np.zeros(((t_end - t_start), (neuron_end - neuron_start)))

    instantanious_firing[times_they_spiked, ids_that_spiked] = count_they_spiked

    # for int_spike_t, neuron_id in zip(int_spike_times, spike_ids):
    #     instantanious_firing[int_spike_t - t_start, neuron_id - neuron_start] += 1

    instantanious_firing /= time_step
    return np.array(range(t_start, t_end)) * time_step, instantanious_firing


"""
 Function to convert a spike train into a set of firing rates

 Args:
    ids: a list of numpy arrays of ids (e.g. for all stimuli)
    times: a list of numpy arrays if times
    neuron_range: (int, int) tuple, the ID of the first (inclusive) neuron to consider (needs to be known cause the last neuron could have never spiked)
    time_range: range in which the spikes are considered. if None all the spikes are taken (full stimulus)

Returns:
    rates: A list of numpy arrays. Each array of length equal to the number neurons, representing their rates. 
"""
def pandas_spikesToFR(spikes, neuron_range, time_range=None):
    assert ('ids' in spikes)
    assert ('times' in spikes)
    # Calculating the average firing rates (since we only present sounds for
    # 1s, just the spike count)


    neuronstart, neuronend = neuron_range
    if time_range:
        timestart, timeend = time_range
        spikes_in_window = spikes[(spikes.times.values >= timestart) & (spikes.times.values <= timeend)]
        timelength = timeend - timestart
    else:
        spikes_in_window = spikes
        timelength = np.max(spikes.times)

    spike_counts = np.zeros(neuronend - neuronstart)
    spike_ids_in_window = spikes_in_window.ids.values
    assert (spike_ids_in_window.shape[0] == spikes_in_window.shape[0])

    neurons_that_fired, count_of_spikes = np.unique(spike_ids_in_window, return_counts=True)

    #faster alternative
    spike_counts[neurons_that_fired - neuronstart] = count_of_spikes

    # for i, neuron_id in enumerate(neurons_that_fired):
    #     spike_counts[neuron_id - neuronstart] = count_of_spikes[i]

        # firing_rates.loc[i, "firing_rates"] = np.count_nonzero(spikes_in_window.ids.values == neuronID) / timelength

    # timelength = 2.0
    # print(timelength)
    firing_rates = pd.DataFrame(
        {"ids": range(neuronstart, neuronend), "firing_rates": spike_counts / timelength})

    return firing_rates


"""
    Given a  nested list with folder, calculate firing rates for every neuron in every layer for every stimulus
    
Args:
    spikes_for_folder: nested list of shape [subfolder][extension]-> containing pandas dataframe with spike times and ids
    info_neurons: dict -> (will throw an error if it does not have the right field ;) )
    info_times:  dict
"""
def calculate_rates_subfolder(spikes_for_folder, info_neurons, info_times):

    subfolder_rates = list()

    for subfolder in range(len(spikes_for_folder)):

        extension_rates = list()

        for extension in range(len(spikes_for_folder[0])):
            atomic_folder_rates = stimuli_and_layerwise_firing_rates(spikes_for_folder[subfolder][extension], info_neurons, info_times)
            extension_rates.append(atomic_folder_rates)

        subfolder_rates.append(extension_rates)

    return subfolder_rates

"""
Turn Spikes into firing rates for each stimulus and layer and neuron type

Args:
    spikes: pandas data frame with spike times and ids
    network_architecture_info: dict
    info_times: dict

Returns:
    nested list with dimensions meaning [stimulus, layer, (exc/inh)]
"""
def stimuli_and_layerwise_firing_rates(spikes, network_architecture_info, info_times):
    length_of_stimulus = info_times["length_of_stimulus"]
    total_length = length_of_stimulus * info_times["num_stimuli"]
    timestart = info_times["time_start"]
    timeend = info_times["time_end"]

    num_exc_neurons_per_layer = network_architecture_info["num_exc_neurons_per_layer"]
    num_inh_neurons_per_layer = network_architecture_info["num_inh_neurons_per_layer"]
    total_per_layer = num_exc_neurons_per_layer + num_inh_neurons_per_layer

    spikes_in_stimuli = helper.splitstimuli(spikes, length_of_stimulus)


    stimuli_responses = list()
    for stimulus_nr in range(info_times["num_stimuli"]):
        # print("stimulus: {}".format(stimulus_nr))
        all_fr_stimulus = firing.pandas_spikesToFR(spikes_in_stimuli[stimulus_nr], neuron_range = (0, total_per_layer * network_architecture_info["num_layers"]), time_range=(timestart, timeend))
        # print("done with firing rates for all neurons")

        layerwise = helper.split_into_layers(all_fr_stimulus,  network_architecture_info)
        # print("done with dividing them into layers")

        exc_inh_layerwise = [helper.split_exc_inh(layer, network_architecture_info) for layer in layerwise]
        # print("done with splitting them into excitatory inhibitory")
        # this is no [(exc_l1, inh_l1), (exc_l2, inh_l2), ... , (excl4, inhl4)]
        stimuli_responses.append(exc_inh_layerwise)
    return stimuli_responses