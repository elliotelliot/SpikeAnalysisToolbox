import numpy as np
import pandas as pd
from . import helper

def general_pre_spike_histogram(*args, normalize_population, **kwargs):
    """convinience function that calls either pre_spike_hist or pouplation_normalised_pre_spike_hist if normalise_poulation is false or true respectively"""
    assert(len(args)==0)

    if normalize_population:
        return population_normalised_pre_spike_hist(**kwargs)
    else:
        try:
            del kwargs['network_architecture']
        except:
            pass

        return pre_spike_hist(**kwargs)

def population_normalised_pre_spike_hist(stimuli_spikes, target_neuron, preneurons, time_step, max_time_delay, start_time, network_architecture, multiple_spikes_per_bin_possible=None):
    """
    Generated a histogram of which neurons spiked what time before the target neuron.
    !!!! This is then devided by how the population of the corresponding pre neuron spikes relative to the target neuron !!!!


    :param stimuli_spikes: list of pandas dataframes, one for each stimulus, each containing columns "ids" and "times", the times should always start at 0 in each stimulus
    :param target_neuron: global index of the target neuron.
    :param preneurons: indices of preneurons to be considered (they have to be sorted in ascending order)
    :param time_step: how broad the timebins in the histogram are
    :param max_time_delay: how long before the target neurons spike will we consider prespikes
    :param start_time: only spikes of the target_neuron that occur after this will be considered
    :param network_architecture: dict
    :param multiple_spikes_per_bin_possible: (optinal) can a neuron spike multiple times within one time_step (histogram bin witdh)
    :return:
        histogram: numpy array of shape (preneuron, timebin) -> for how many of the target neurons spikes the preneuron spiked timebin*timestep much time before the target neuron
        times: numpy array -> indices for the times corresponding to the second dimension in the array 'histogram'
    """

    hist, times = pre_spike_hist(stimuli_spikes, target_neuron, preneurons, time_step, max_time_delay, start_time, multiple_spikes_per_bin_possible)
    prepopulations = -1 * helper.get_popluation_id(np.array(preneurons), network_architecture)
    relevant_populations = np.sort(np.unique(prepopulations))

    stims_with_population_ids = list()
    for stim in stimuli_spikes:
        id_is_target = stim.ids == target_neuron
        tmp_ids = -1 * helper.get_popluation_id(stim.ids.values, network_architecture)
        tmp_ids[id_is_target] = target_neuron
        new_stim = pd.DataFrame(dict(ids=tmp_ids, times=stim.times.values))
        stims_with_population_ids.append(new_stim)

    population_hist, pop_times = pre_spike_hist(stimuli_spikes=stims_with_population_ids, target_neuron=target_neuron,
                                                preneurons=relevant_populations, # now we have populations as 'virtual' neurons
                                                time_step=time_step, max_time_delay=max_time_delay,
                                                start_time=start_time,
                                                multiple_spikes_per_bin_possible=True) # of course a population will spike multiple times before the target

    assert(np.all(pop_times == times))

    prepopulation_ids = np.searchsorted(relevant_populations, prepopulations)
    normed_hist =  hist / population_hist[prepopulation_ids, :]

    return normed_hist, times



def pre_spike_hist(stimuli_spikes, target_neuron, preneurons, time_step, max_time_delay, start_time, multiple_spikes_per_bin_possible=None):
    """
    Generated a histogram of which neurons spiked what time before the target neuron
    :param stimuli_spikes: list of pandas dataframes, one for each stimulus, each containing columns "ids" and "times", the times should always start at 0 in each stimulus
    :param target_neuron: global index of the target neuron.
    :param preneurons: indices of preneurons to be considered (they have to be sorted in ascending order)
    :param time_step: how broad the timebins in the histogram are
    :param max_time_delay: how long before the target neurons spike will we consider prespikes
    :param start_time: only spikes of the target_neuron that occur after this will be considered
    :param multiple_spikes_per_bin_possible: (optinal) can a neuron spike multiple times within one time_step (histogram bin witdh)
    :return:
        histogram: numpy array of shape (preneuron, timebin) -> for how many of the target neurons spikes the preneuron spiked timebin*timestep much time before the target neuron
        times: numpy array -> indices for the times corresponding to the second dimension in the array 'histogram'
    """
    assert(not target_neuron in preneurons)

    if multiple_spikes_per_bin_possible is None:
        refractory_period = 0.002
        multiple_spikes_per_bin_possible = (time_step > refractory_period)
        print("Multiple spikes per time_step possible: {}".format(multiple_spikes_per_bin_possible))

    if(len(np.unique(preneurons)) != len(preneurons)):
        raise ValueError("The preneurons have to be unique")

    # make sure that preneuron indicies are sorted
    if np.any(np.diff(preneurons)<0):
        raise ValueError("preneurons have to be sorted in ascending order")


    time_bin_edges = -1 * np.arange(0, max_time_delay + time_step, time_step)[::-1] # [-max_delay, ..., -4ms, -2ms, -0ms]

    histogram = np.zeros((len(preneurons), len(time_bin_edges)-1))

    all_spikes_count = 0

    for stimulus in stimuli_spikes:
        # spikes of our target neuron after we started looking
        target_spike_mask = (stimulus.times > start_time) & (stimulus.ids == target_neuron)
        target_spikes = stimulus.times.values[target_spike_mask]

        all_spikes_count += len(target_spikes)

        # spikes of 'other' neurons that wea re looking at
        possible_other_spikes = stimulus[np.isin(stimulus.ids, preneurons)]

        # go through each of the target neuron's spike
        for target_spike_time in target_spikes:
            spikes_not_to_far_in_past = possible_other_spikes.times.values >= (target_spike_time - max_time_delay)
            spikes_before_target = possible_other_spikes.times.values < target_spike_time
            mask_spikes_in_time_win =  spikes_not_to_far_in_past & spikes_before_target

            # select all spikes of the other neurons that happen in the winow of max_time_delay length befor the target spike
            possible_other_spikes_in_window = possible_other_spikes[mask_spikes_in_time_win]

            if len(possible_other_spikes_in_window) == 0:
                continue


            #calculate the delay of those spikes relative to the current target spike
            relative_spike_times = possible_other_spikes_in_window.times.values - target_spike_time

            # Sort them into the bins
            digitized_relative_spike_times = np.digitize(relative_spike_times, time_bin_edges)-1

            assert(np.all(digitized_relative_spike_times >=0))

            # indices of preneurons in histogram: For each preneuron id we have to find out which place in the histogram it should go too
            assert(np.all(np.isin(possible_other_spikes_in_window.ids.values, preneurons)))



            if multiple_spikes_per_bin_possible:
                ####################################################
                #new version
                id_time_tuple_array = np.stack([possible_other_spikes_in_window.ids.values, digitized_relative_spike_times], axis=0)
                # shape (2, n_spikes) first row is the ids, second is the times

                id_time_pairs, occurance_count = np.unique(id_time_tuple_array, return_counts=True, axis=1)
                id_time_pairs = id_time_pairs.astype(int)
                spike_ids_for_hist = np.searchsorted(preneurons, id_time_pairs[0, :])
                digitized_times = id_time_pairs[1, :]
                # print(spike_ids_for_hist)
                spike_count = occurance_count
                histogram[spike_ids_for_hist, digitized_times] += spike_count

            else:
                spike_ids_for_hist = np.searchsorted(preneurons, possible_other_spikes_in_window.ids.values)
                # searchsorted goes through each possible_other_spike_in_window and returns the insertion_indices where in preneurons(_ids) it would fit.
                # Since preneurons is a sorted list with unique elements this insertion_index is exactly the index at which the other_spike_in_window-ID appears in the preneurons array.

                histogram[spike_ids_for_hist, digitized_relative_spike_times] += 1


    # calculate the times
    if all_spikes_count ==0:
        raise ValueError("Target neuron did not spike")
    print("The Target neuron spiked {} times".format(all_spikes_count))

    normalized_hist = histogram / all_spikes_count

    times = (time_bin_edges[1:] + time_bin_edges[:-1])/2

    return normalized_hist, times

def histogram_for_incoming_synapses(stimuli_spikes, target_neuron, synapses, time_step, time_around_synaptic_delay, start_time,
                                    simulation_timestep=None, population_normalize=False, network_architecture=None):
    """
    Generate a histogram of when presynaptic neurons spiked relative to the synaptic delay to the target neuron.
    :param stimuli_spikes: list of pandas dataframes with columns 'ids' 'spikes'
    :param target_neuron: id of target neuron
    :param synapses: pandas dataframe with columnd 'pre' 'post' 'delays'
    :param time_step: binwidth of the array
    :param time_around_synaptic_delay:
    :param start_time:
    :param simulation_time_step: timestep of the simulation, since the synaptic delays are given in multiples of it by spike
    :return:
        histogram
        times
    """
    relevant_synapses = synapses[synapses.post == target_neuron]

    pre_ids = relevant_synapses.pre.values
    pre_ids = np.sort(np.unique(pre_ids))

    if(simulation_timestep is None):
        simulation_timestep = time_step
        print("!!! Simulation timestep was not given and is assumed to be {} ms".format(simulation_timestep*1e3))

    if population_normalize:
        if network_architecture is None:
            raise ValueError("For normalisation architecture info has to be given")

    max_delay = np.max(relevant_synapses.delays.values) * simulation_timestep

    window_start_before_target_spike = max_delay + time_around_synaptic_delay

    if population_normalize:
        raw_hist, raw_times = population_normalised_pre_spike_hist(stimuli_spikes=stimuli_spikes, target_neuron=target_neuron,
                                                                   preneurons=pre_ids, time_step=time_step,
                                                                   max_time_delay=window_start_before_target_spike,
                                                                   start_time=start_time, network_architecture=network_architecture)
    else:
        raw_hist, raw_times = pre_spike_hist(stimuli_spikes=stimuli_spikes, target_neuron=target_neuron, preneurons=pre_ids,
                                             time_step=time_step, max_time_delay=window_start_before_target_spike, start_time=start_time)

    result_times = np.arange(-time_around_synaptic_delay, time_around_synaptic_delay, time_step)
    n_times = len(result_times)
    result_histogramm = np.zeros((len(relevant_synapses), n_times))

    for syn_id in range(len(relevant_synapses)):
        current_pre_id_global = relevant_synapses.iloc[syn_id].pre
        current_pre_id = np.searchsorted(pre_ids, current_pre_id_global)
        current_delay = relevant_synapses.iloc[syn_id].delays * simulation_timestep

        window_start_time = - current_delay - time_around_synaptic_delay # time_around_synaptic_delay many ms before the spike would go 'into' the synapse and reach the target neuron current_delay many ms later at timepoint 0 according to the raw_times

        id_of_start_timepoint_in_raw_times = np.argmin(np.abs(raw_times - window_start_time)) # at this point the histogram starts

        that_neurons_spikes_in_window = raw_hist[current_pre_id, id_of_start_timepoint_in_raw_times:(id_of_start_timepoint_in_raw_times + n_times)]

        result_histogramm[syn_id, 0:len(that_neurons_spikes_in_window)] = that_neurons_spikes_in_window

    return result_histogramm, result_times