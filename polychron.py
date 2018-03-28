from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import combine_stimuli
from . import helper
from . import information_scores as info_analysis


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
    assert(np.all(np.isfinite(hist)))
    assert(np.all(np.isfinite(hist)))
    normed_hist =  hist / population_hist[prepopulation_ids, :]

    normed_hist[population_hist[prepopulation_ids, :] == 0] = 0

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
    if (target_neuron in preneurons):
        print("In pre_spike_hist the target neuron is actually in preneurons")

    if multiple_spikes_per_bin_possible is None:
        refractory_period = 0.002
        multiple_spikes_per_bin_possible = (time_step > refractory_period)
        # print("Multiple spikes per time_step possible: {}".format(multiple_spikes_per_bin_possible))

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
        if len(target_spikes)==0:
            continue

        all_spikes_count += len(target_spikes)

        # spikes of 'other' neurons that wea re looking at
        possible_other_spikes = stimulus[np.isin(stimulus.ids, preneurons)]

        # go through each of the target neuron's spike
        for target_spike_time in target_spikes:
            spikes_not_to_far_in_past = possible_other_spikes.times.values >= (target_spike_time - max_time_delay)
            spikes_before_target = possible_other_spikes.times.values < target_spike_time
            mask_spikes_in_time_win =  spikes_not_to_far_in_past & spikes_before_target

            # select all spikes of the other neurons that happen in the winow of max_time_delay length before the target spike
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
        # raise ValueError("Target neuron did not spike")
        # print("The Target Neuron did not spike")
        normalized_hist = histogram
    else:
        # print("The Target neuron spiked {} times".format(all_spikes_count))

        normalized_hist = histogram / all_spikes_count

    times = (time_bin_edges[1:] + time_bin_edges[:-1])/2

    return normalized_hist, times

def all_combination_histogram(stimuli_spikes, target_neuron_range, time_step, max_time_delay, start_time, multiple_spikes_per_bin_possible=None):
    """
    for each pair of target neuron a histogram is computed about how often they spike in that delay to each other

    :param stimuli_spikes: list of pandas dataframes, one for each stimulus, each containing columns "ids" and "times", the times should always start at 0 in each stimulus
    :param target_neuron_range: range of global indices of the target neurons.

    :param time_step: how broad the timebins in the histogram are
    :param max_time_delay: how long before the target neurons spike will we consider prespikes
    :param start_time: only spikes of the target_neuron that occur after this will be considered
    :param multiple_spikes_per_bin_possible: (optinal) can a neuron spike multiple times within one time_step (histogram bin witdh)
    :return:
        histogram: numpy array of shape (preneuron, postneuron, time_delta) -> given that postneuron spiked at time t, probability that preneuron spiked at t-time_delta
        times: numpy array -> same shape as time_delta
    """
    # postneuron refers to the later spiking neuron. not necessarily postsynaptic
    target_neuron_start_id, _end_id = target_neuron_range
    target_neurons = np.arange(*target_neuron_range)

    n_neurons = len(target_neurons)
    n_times =int(np.ceil(max_time_delay/time_step))

    target_neurons = np.sort(target_neurons)

    result = np.zeros((n_neurons, n_neurons, n_times))

    times = None
    for postneuron in target_neurons:
        pre_neurons = target_neurons[target_neurons != postneuron]

        tmp_hist, new_times = pre_spike_hist(stimuli_spikes=stimuli_spikes, target_neuron=postneuron, preneurons=pre_neurons, time_step=time_step, max_time_delay=max_time_delay, start_time=start_time, multiple_spikes_per_bin_possible=multiple_spikes_per_bin_possible)
        if times is None:
            times = new_times
        else:
            assert(np.all(times==new_times))

        result[pre_neurons-target_neuron_start_id, postneuron-target_neuron_start_id, :] = tmp_hist

    return result, times

def all_stimuli_neuroncombination_histogram(stimuli_spikes, *args, **kwargs):
    """
    Calls all_combination_histogram for each stimulus seperatly and transfrers the kwargs to it
    :param stimuli_spikes:
    :param args:
    :param kwargs:
    :return: hist, times
        hist ... numpy array of shape (stimulus, preneuron, postneuron, delta_time) -> probability of preneuron spiking delat_time before postneuron
        times ... meaning of the times
    """
    assert(len(args)==0)
    times = None
    hist_collector = list()
    print("Stimulus: ")
    for i, stim in enumerate(stimuli_spikes):
        hist, new_times = all_combination_histogram([stim], **kwargs)
        if times is None:
            times = new_times
        else:
            assert(np.all(times==new_times))

        hist1d = np.expand_dims(hist, axis=0)
        hist_collector.append(hist1d)

        print(i, end=' ')

    return np.concatenate(hist_collector, axis=0), times

def spike_pair_info(stimuli_spikes, neuron_ids, object_indices, max_time_delay, time_step, n_bins_prob_table, start_time, prob_2_info_fun=info_analysis.single_cell_information):
    """
    For each spike pair of 2 ids and one delay (i,j,d)  we compute the probability of that spikepair = p(neuron i spikes at time t-d | neuron j spikes at t)
    we do this seperatly for each stimulus presentation. so we have one of these tables for each stimulus presentation which contains an 'activation' value of the pair (pre, post, delay) for the stimulus.
    This can be interpreted in the same way as a firing rate. one pair (pre, post, delay) corresponds to one neuron.
    So we then make a probability table out of these.
    And use it to calculate the single cell information

    :param stimuli_spikes: list of spikes (each is a pandas dataframe with ids and times)
    :param neuron_ids: list of neuron ids (for each pair of them the information is computed)
    :param object_indices: nested list [object] -> ids of stimuli that contain that object
    :param max_time_delay: max delay betwen pre and post spike of a pair
    :param time_step: width of the bins for the delay betwen pr and post spike
    :param n_bins_prob_table: how many different responses of a pair are possible for the probability table
    :param start_time: time within a presentation after which we look for pre post pairs
    :param prob_2_info_fun: function that computes information based on the
    :return: numpy array of shape (object, time_delay, pre_neuron, post_neuron) information for the pair (pre_neuron, post_neuron, time_delay)
    """
    assert(np.all(neuron_ids[:-1] < neuron_ids[1:]))# sorted and unique

    n_time_delays = int(np.ceil(max_time_delay/time_step))
    n_stimuli = len(stimuli_spikes)
    n_neurons = len(neuron_ids)
    n_objects = len(object_indices)

    all_information = np.zeros((n_objects, n_time_delays, n_neurons, n_neurons))

    times = None
    print("Neuron: ", end='')
    for i, post_neuron_id in enumerate(neuron_ids):
        remaining_ids = neuron_ids[neuron_ids != post_neuron_id]

        histogram_all_stimuli = np.zeros((n_stimuli, n_time_delays, n_neurons))
        #  stimulus, time_delay(which we will pretend to be layers), n_postneurons

        for stim_id, stim_spikes in enumerate(stimuli_spikes):

            hist, new_times = pre_spike_hist(stimuli_spikes=[stim_spikes], target_neuron=post_neuron_id, preneurons=remaining_ids, time_step=time_step, max_time_delay=max_time_delay, start_time=start_time)
            # hist has shape [remaining_ids, time_delay]
            if times is not None:
                assert(np.all(new_times==times))
            else:
                times = new_times



            remaining_ids_relative = np.searchsorted(neuron_ids, remaining_ids)

            histogram_all_stimuli[stim_id, :, remaining_ids_relative] = hist # the histogram for post_neuron_id, post_neuron_id is 0
            # I expected to need hist.transpose() here but I don't

        # now we have one response for all stimuli (for all neurons and time delays, now we compute a frequency table


        probability_table_that_post_neuron = combine_stimuli.response_freq_table(histogram_all_stimuli, object_indices, n_bins=n_bins_prob_table)
        # [n_objects, n_time_delays, n_pre_neurons, n_bins]

        information_that_post_neuron = prob_2_info_fun(probability_table_that_post_neuron)
        # [n_objects, n_time_delays, n_pre_neurons]

        all_information[:, :, :, i] = information_that_post_neuron
        print(i, end=' ')

    return all_information, times


def threaded_spike_pair_info(stimuli_spikes, neuron_ids, object_indices, max_time_delay, time_step, n_bins_prob_table, start_time, n_threads, post_min_spike_count=10, prob_2_info_fun=info_analysis.single_cell_information):
    """
    For each spike pair of 2 ids and one delay (i,j,d)  we compute the probability of that spikepair = p(neuron i spikes at time t-d | neuron j spikes at t)
    we do this seperatly for each stimulus presentation. so we have one of these tables for each stimulus presentation which contains an 'activation' value of the pair (pre, post, delay) for the stimulus.
    This can be interpreted in the same way as a firing rate. one pair (pre, post, delay) corresponds to one neuron.
    So we then make a probability table out of these.
    And use it to calculate the single cell information

    :param stimuli_spikes: list of spikes (each is a pandas dataframe with ids and times)
    :param neuron_ids: list of neuron ids (for each pair of them the information is computed)
    :param object_indices: nested list [object] -> ids of stimuli that contain that object
    :param max_time_delay: max delay betwen pre and post spike of a pair
    :param time_step: width of the bins for the delay betwen pr and post spike
    :param n_bins_prob_table: how many different responses of a pair are possible for the probability table
    :param start_time: time within a presentation after which we look for pre post pairs
    :param prob_2_info_fun: function that computes information based on the
    :param post_min_spike_count: if a postneuron spikes less then this number. it is ignored
    :return: numpy array of shape (object, time_delay, pre_neuron, post_neuron) information for the pair (pre_neuron, post_neuron, time_delay)
    """
    assert(np.all(neuron_ids[:-1] < neuron_ids[1:]))# sorted and unique

    caller = SpikePairInfoCaller(stimuli_spikes=stimuli_spikes, time_step=time_step, max_time_delay=max_time_delay,
                                 start_time=start_time, neuron_ids=neuron_ids, object_indices=object_indices,
                                 n_bins_prob_table=n_bins_prob_table, post_min_spike_count=post_min_spike_count,
                                 prob_2_info_fun=prob_2_info_fun)

    worker_pool = Pool(processes=n_threads)

    all_info = worker_pool.map(caller, neuron_ids)
    worker_pool.close()
    worker_pool.join()

    expanded_info = [np.expand_dims(h, axis=-1) for h in all_info]

    all_info_numpy = np.concatenate(expanded_info, axis=-1)
    times= None
    return all_info_numpy, times

class SpikePairInfoCaller:
    def __init__(self, stimuli_spikes, time_step, max_time_delay, start_time, neuron_ids, object_indices, n_bins_prob_table, prob_2_info_fun, post_min_spike_count=10):
        self.stimuli_spikes = stimuli_spikes
        self.time_step = time_step
        self.max_time_delay = max_time_delay
        self.start_time = start_time
        self.neuron_ids = neuron_ids
        self.object_indices = object_indices

        self.n_stimuli = len(self.stimuli_spikes)
        self.n_time_delays = int(np.ceil(max_time_delay / time_step))

        self.n_neurons = len(neuron_ids)
        self.n_bins_prob_table = n_bins_prob_table
        self.prob_2_info_fun = prob_2_info_fun
        # n_objects = len(object_indices)

        self.post_spike_threshold = post_min_spike_count
        print("If the target neuron spikes less then {} times in a stimulus. Then the histogram won't be computed.".format(self.post_spike_threshold))



    def __call__(self, post_neuron_id):
        remaining_ids = self.neuron_ids[self.neuron_ids != post_neuron_id]

        histogram_all_stimuli = np.zeros((self.n_stimuli, self.n_time_delays, self.n_neurons))
        #  stimulus, time_delay(which we will pretend to be layers), n_postneurons

        for stim_id, stim_spikes in enumerate(self.stimuli_spikes):

            n_target_neuron_spikes = np.count_nonzero(stim_spikes.ids.values == post_neuron_id)
            if n_target_neuron_spikes < self.post_spike_threshold:
                continue

            hist, new_times = pre_spike_hist(stimuli_spikes=[stim_spikes], target_neuron=post_neuron_id, preneurons=remaining_ids, time_step=self.time_step, max_time_delay=self.max_time_delay, start_time=self.start_time, multiple_spikes_per_bin_possible=True)
            # hist has shape [remaining_ids, time_delay]

            remaining_ids_relative = np.searchsorted(self.neuron_ids, remaining_ids)

            histogram_all_stimuli[stim_id, :, remaining_ids_relative] = hist # the histogram for post_neuron_id, post_neuron_id is 0
            # I expected to need hist.transpose() here but I don't

        # now we have one response for all stimuli (for all neurons and time delays, now we compute a frequency table


        probability_table_that_post_neuron = combine_stimuli.response_freq_table(histogram_all_stimuli, self.object_indices, n_bins=self.n_bins_prob_table)
        # [n_objects, n_time_delays, n_pre_neurons, n_bins]

        information_that_post_neuron = self.prob_2_info_fun(probability_table_that_post_neuron)
        # [n_objects, n_time_delays, n_pre_neurons]

        # all_information[:, :, :, i] = information_that_post_neuron
        # print(i, end=' ')
        return information_that_post_neuron









def threaded_all_stimuli_neuroncombination_histogram(stimuli_spikes, *args, n_threads, **kwargs):
    """
    Calls all_combination_histogram for each stimulus seperatly and transfrers the kwargs to it
    :param stimuli_spikes:
    :param args:
    :param kwargs:
    :return: hist, times
        hist ... numpy array of shape (stimulus, preneuron, postneuron, delta_time) -> probability of preneuron spiking delat_time before postneuron
        times ... meaning of the times
    """
    assert(len(args)==0)
    times = None
    worker_pool = Pool(processes=n_threads)

    wraped_stimuli = [[s] for s in stimuli_spikes]

    times_hist = worker_pool.map(helper.Caller(all_combination_histogram, **kwargs), wraped_stimuli)
    worker_pool.close()
    worker_pool.join()

    all_hists, all_times = zip(*times_hist)

    times = all_times[0]
    for t in all_times:
        assert(np.all(t==times))


    all_hists_expanded_dim = [np.expand_dims(h, axis=0) for h in all_hists]


    return np.concatenate(all_hists_expanded_dim, axis=0), times




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