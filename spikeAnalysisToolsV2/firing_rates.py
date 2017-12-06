import numpy as np
import pandas as pd
from multiprocessing import Pool
from numba import jit

# from helper import *
from . import helper
from . import firing_rates as firing
from . import data_loading as data
# from .. import data_loading as data

"""
computes the time course of the instantanious Firing rate for layers
Args:
    spikes: pandas data frame
    network_architecture: dict 
    time_step: the temporal resolution
Returns:
    times: the start times for the bins. i.e. excitatory_timecourse[layer, timepoint_id, neuron_id] contains
     the firint rate of the respective neuron in the intevall (times[timepoint_id], times[timepoint_id] + time_step)
    excitatory_timecourse: numpy array with dimensions [layer, timepoint, neuronid]
    inhibitroy_timecourse: numpy array of same shape. 
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

        # assert(np.all(time_exc == time_inh))

        all_inh_collector.append(inh_InstantFR)
        all_exc_collector.append(exc_InstantFR)

    excitatory_timecourse = np.stack(all_exc_collector, axis=1)
    inhibitory_timecourse = np.stack(all_inh_collector, axis=1)

    return time_exc, excitatory_timecourse, inhibitory_timecourse,


"""
Convert spikes into an instantaneus firing rate
Args:
    spikes: pandas data frame with fields "ids" and "times"
    neuron_range: int tuple with start and end index
    time_step: the time resolution
    time_range: float tuple with the target time window

Returns:
    time: numpy array with the time steps
    instantaneus_firing: numpy array of dimensions [timepoint, neuron_id] giving the scalar firing rate of the neuron in the intervall
        time[timepoint], time[timepoint]+time_step

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

    instantanious_firing = np.zeros(((t_end - t_start), (neuron_end - neuron_start)))

    if len(spikes)==0:
        return np.array(range(t_start, t_end)) * time_step, instantanious_firing

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
    times_they_spiked = id_time_pairs[1, :] - t_start
    count_they_spiked = occurance_count



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
    rates: pandas dataframe with columns "ids", "firing_rates"
"""
# @jit(cache=True)
def spikesToFR(spikes, neuron_range, time_range=None):
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
    
Returns:
    nested list of dimensions [subfolder][extension][stimulus][layer][exh/inh]-> pandas data frame with columns "ids", "firing_rates"
    
    
"""
def calculate_rates_subfolder(spikes_for_folder, info_neurons, info_times):


    worker_pool = Pool(processes=5)

    subfolder_rates = list()

    for subfolder in range(len(spikes_for_folder)):

        # extension_rates = list()

        extension_rates_pro = worker_pool.map(Caller(stimuli_and_layerwise_firing_rates, info_neurons, info_times), spikes_for_folder[subfolder])

        # for extension in range(len(spikes_for_folder[0])):
        #     print(extension, end="  ")
        #     atomic_folder_rates = stimuli_and_layerwise_firing_rates(spikes_for_folder[subfolder][extension], info_neurons, info_times)
        #     extension_rates.append(atomic_folder_rates)
        # print("")
        # assert(np.all(extension_rates[0][3][3][0] == extension_rates_pro[0][3][3][0]))

        subfolder_rates.append(extension_rates_pro)

    worker_pool.close()
    worker_pool.join()
    return subfolder_rates

def slow_calculate_rates_subfolder(spikes_for_folder, info_neurons, info_times):



    subfolder_rates = list()

    for subfolder in range(len(spikes_for_folder)):

        extension_rates = list()
        print("Starting subfolder >>  ", end="")

        for extension in range(len(spikes_for_folder[0])):
            print(extension, end="  ")
            atomic_folder_rates = stimuli_and_layerwise_firing_rates(spikes_for_folder[subfolder][extension], info_neurons, info_times)
            extension_rates.append(atomic_folder_rates)
        print("")

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
# @jit(cache=True)
def stimuli_and_layerwise_firing_rates(spikes, network_architecture_info, info_times):
    length_of_stimulus = info_times["length_of_stimulus"]
    total_length = length_of_stimulus * info_times["num_stimuli"]
    timestart = info_times["time_start"]
    timeend = info_times["time_end"]

    num_exc_neurons_per_layer = network_architecture_info["num_exc_neurons_per_layer"]
    num_inh_neurons_per_layer = network_architecture_info["num_inh_neurons_per_layer"]
    total_per_layer = num_exc_neurons_per_layer + num_inh_neurons_per_layer

    # check if the number of stimuli and the stimuli length are reasonable. i.e. if total_length is ok
    last_spike = np.max(spikes.times)
    if(last_spike > total_length):
        raise ValueError("The last spike was after the last stimulus was presented: length_of_stimulus or num_stimuli seems to be wrong")
    if(last_spike < total_length - length_of_stimulus):
        raise ValueError("There were no spikes during the last stimulus: length_of_stimulus or num_stimuli seems to be wrong")

    spikes_in_stimuli = helper.splitstimuli(spikes, length_of_stimulus)


    stimuli_responses = list()
    for stimulus_nr in range(info_times["num_stimuli"]):
        # print("stimulus: {}".format(stimulus_nr))
        all_fr_stimulus = firing.spikesToFR(spikes_in_stimuli[stimulus_nr], neuron_range = (0, total_per_layer * network_architecture_info["num_layers"]), time_range=(timestart, timeend))
        # print("done with firing rates for all neurons")

        layerwise = helper.split_into_layers(all_fr_stimulus,  network_architecture_info)
        # print("done with dividing them into layers")

        exc_inh_layerwise = [helper.split_exc_inh(layer, network_architecture_info) for layer in layerwise]
        # print("done with splitting them into excitatory inhibitory")
        # this is no [(exc_l1, inh_l1), (exc_l2, inh_l2), ... , (excl4, inhl4)]
        stimuli_responses.append(exc_inh_layerwise)
    return stimuli_responses


@jit(cache=True)
def digitize_firing_rates_with_equispaced_bins(firing_rates, n_bins):
    """
    Digitizes firing rates such that every firing rate has a value in [0, n_bins[
    For every neuron, the different firing rates for the different stimuli are divided into equally spaced bins. Stringer 2005

    The bin borders are the same for a given neuron over all stimuli, but different for 2 neurons in the same stimulus

    :param firing_rates:
    :param n_bins:
    :return:
    """
    n_stimuli, n_layer, n_neurons = firing_rates.shape

    minimal_response = np.min(firing_rates, axis=0)
    maximal_response = np.max(firing_rates, axis=0) + 1 # + 1 cause the last one is <

    categorized_firing_rates = np.empty((n_stimuli, n_layer, n_neurons), dtype=int)

    for l in range(n_layer):
        for n in range(n_neurons):
            bins = np.linspace(minimal_response[l, n], maximal_response[l, n], n_bins+1) # +1 because for n bin boundraries there are only n-1 buckets
            categorized_firing_rates[:, l, n] = np.digitize(firing_rates[:, l, n], bins) - 1


    # assert(not np.any((categorized_firing_rates.flatten() < 0)))
    # assert(not np.any((categorized_firing_rates.flatten() >= n_bins)))

    return categorized_firing_rates


def inplace_normailize_mean_per_stimulus(firing_rates):
    """
    Remove mean in layer for each layer in each stimulus in each epoch
    :param firing_rates: nested list of shape [extension(epoch)][stimulus][layer][exc/inh] -> pandas dataframe with columns 'ids', and 'firing_rates'
    :return: firing_rates of excactly the same shape, but the pandas dataframes all have mean 0 now
    """
    for extension in firing_rates:
        for stimulus in extension:
            for layer in stimulus:
                for neuron_type_fr in layer:
                    mean = neuron_type_fr.firing_rates.mean()
                    neuron_type_fr.firing_rates -= mean


class Caller(object):
    def __init__(self, function, *args):
        """
        when you call an instance of this object with obj(input) it will call function(input, *args)

        :param function:  the function that should be called
        :param args:  oter params that the function takes
        """
        self.function = function
        self.args = args[:]
    def __call__(self, input):
        return self.function(input, *self.args)
