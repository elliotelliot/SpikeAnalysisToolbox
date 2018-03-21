import numpy as np
import scipy.signal as ssignal
from . import helper

def population_activity(spike_times, time_range = None, bin_width=2e-3, n_neurons=1):
    """
    Calculate population activity for discrete timepoints
    :param spike_times: numpy array of spiketimes
    :param time_range: (float, float) tuple with start and end time
    :param bin_width: width of a timebin
    :param n_neurons: neurons that the spikes are from. to normalise activity
    :return: times, activity
        times: numpy array with times at which the activity has a certain value
        activity: population activity
    """

    if time_range is None:
        start_time = np.min(spike_times)
        end_time = np.max(spike_times)
    else:
        start_time, end_time = time_range

    bin_edges = np.arange(start_time, end_time+bin_width, bin_width)

    selected_spike_times = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]

    activity, _b = np.histogram(selected_spike_times, bins=bin_edges, range=time_range)

    times = ( bin_edges[1:] + bin_edges[:-1])/2

    avg_firing_rate = (activity/bin_width)/n_neurons

    return times, avg_firing_rate

def fit_fft(population_act, times, smooth_win_hz=3):
    """
    Fit sinusoid to population_activity
    :param population_act: numpy array with intensity values for each timepoint (e.g. optained from oscilations.population_activity)
    :param times:  times at which the population activity was measured
    :param smooth_win_hz: smoothing the frequency spectrum enables saver finding of local maxima
    :return frex, factor, spectrum : describe the oscilation as np.exp(2j * np.pi * x * frex) * factor |
        spectrum... (freqeuncy, intensity)
    """
    bin_width = times[1] - times[0]

    ft = np.fft.rfft(population_act)
    ft_intensity = np.absolute(ft)
    ft_intensity[0]=0 # ignore the constant term

    ft_frex = np.fft.rfftfreq(len(population_act), bin_width)
    d_freq = ft_frex[2] = ft_frex[1]

    if smooth_win_hz:
        ft_intensity_smoothed = ssignal.convolve(ft_intensity, ssignal.gaussian(20, smooth_win_hz/d_freq), mode='same')
        ft_intensity = ft_intensity_smoothed


    max_frequency_intensity = np.argmax(ft_intensity)
    max_frequency_id_candidates = ssignal.argrelextrema(ft_intensity, np.greater)[0]
    if(len(max_frequency_id_candidates)==0):
        return 0, 0, (ft_frex, ft_intensity)

    max_frequency_id = max_frequency_id_candidates[ft_intensity[max_frequency_id_candidates] > max_frequency_intensity/2][0]

    # if False:
    #     act_peaks = fit_activity_peaks(population_act, times)
    #     d_act_peaks = act_peaks[1:] - act_peaks[:-1]
    #     frequency = 1/np.median(d_act_peaks)
    #     max_frequency_id = np.argmin(np.abs(frequency - ft_frex))

    max_frequency = ft_frex[max_frequency_id]

    # max_offset = np.angle(ft[max_frequency_id])

    # max_intensity = ft_intensity[max_frequency_id]/len(population_act)

    offset_correction = np.exp(2j * np.pi * max_frequency * (-times[0]))

    faktor = ft[max_frequency_id] * offset_correction
    faktor /= len(population_act)

    return max_frequency, faktor, (ft_frex, ft_intensity)

def fit_activity_peaks(population_act, times, smooth_win=1e-2, min_peak_intensity=0.3):
    """
    find maxima in population activity
    :param population_act: numpy array eg. from population_activity
    :param times: numpy array of same shape with corresponding times
    :param smooth_win: std of smoothing gaussian in seconds
    :param min_peak_intensity: percent that a peak's intensity has to be above relative to the strongest of all found peaks
    :return: numpy array containing the times of found peaks
    """

    d_times = times[1] - times[0]

    if smooth_win is not None:
        population_act = ssignal.convolve(population_act, ssignal.gaussian(20, smooth_win/d_times), mode='same')

    extrema_ids = ssignal.argrelextrema(population_act, np.greater)[0]
    if len(extrema_ids) ==0:
        return np.zeros((0,))

    # extrema_ids = ssignal.find_peaks_cwt(population_act, np.arange(1, 20))
    extrema_values = population_act[extrema_ids]
    mean_peak_value = np.max(extrema_values)

    selected_extrema_ids = extrema_ids[extrema_values > (mean_peak_value * min_peak_intensity)]

    return times[selected_extrema_ids]

def get_sinusoid(frex, complex_faktor):
    sinusoid = lambda time: np.real(np.exp(2j * np.pi * time * frex) * complex_faktor)
    return sinusoid

def fit_sinus_peaks(population_act, times):
    raise NotImplementedError("Smoothing Parameter not forwarded")
    frequency, faktor, _ = fit_fft(population_act, times)

    freq_factor = frequency * 2 * np.pi

    angle = np.angle(faktor)

    first_peak_at_angle = (2*np.pi - angle) #% 2*np.pi

    peaks = np.arange(first_peak_at_angle/freq_factor, times[-1], 2*np.pi/freq_factor)

    peaks_in_time_range = peaks[peaks >= times[0]]

    return peaks_in_time_range

def spikes_rel_to_population_peaks(population_spikes, population_peaks, neuron_range):
    """
    Within an oscilation compute a neurons first spike time relative to the peak of the oscilation. i.e. when the population is most active.
    Each spike is assigned to the clossest population peak (in poulation_peaks) and for each peak the first spike time of a neuron is taken.

    :param population_spikes: pandas dataframe with 'ids', and 'times'
    :param population_peaks: 1-d numpy array with the times of population peaks. e.g. from oscilations.fit_activity_peaks
    :param neuron_range: (start_id, end_id) ids of the neurons. all ids in population_spikes must be in this range
    :return: numpy array of dimensions (n_neurons, n_population_peaks)
    """
    assert(np.all(population_spikes.times.values[:-1] <= population_spikes.times.values[1:]))

    start_neuron_id, end_neuron_id = neuron_range

    result = np.zeros((end_neuron_id - start_neuron_id, len(population_peaks)))*np.nan

    # first_spike, last_spike = np.min(population_spikes.times.values), np.max(population_spikes.times.values)

    inter_peak_midles = (population_peaks[1:] + population_peaks[:-1])/2

    first_oscilation_start = population_peaks[0] - (inter_peak_midles[0] - population_peaks[0])
    last_oscilation_end    = population_peaks[-1] + (population_peaks[-1] - inter_peak_midles[-1])

    oscilation_edges = np.insert(inter_peak_midles, [0, len(inter_peak_midles)], [first_oscilation_start, last_oscilation_end])

    for i, peak_time in enumerate(population_peaks):
        oscilation_start = oscilation_edges[i]
        oscilation_end   = oscilation_edges[i+1]
        start_spike_id, end_spike_id = np.searchsorted(population_spikes.times.values, [oscilation_start, oscilation_end])

        spikes_in_oscilation = population_spikes.iloc[start_spike_id: end_spike_id]

        # why Reverse?
        # we are only interested in the first spike of a neuron in the population peak
        # but we will set all the values for all spikes in the result array. so with reversing we first set the value for the latest spike
        # then second latest and so on, always overwritting the privious value, only the last *value* we set, will be kept
        # which will be the timing of the first spike, since we reversed.
        # hacky but should be fast

        spike_ids_reverse = spikes_in_oscilation.ids.values[::-1]
        spike_ids_reverse_relative = (spike_ids_reverse - start_neuron_id).astype(int)

        spike_times_reverse = spikes_in_oscilation.times.values[::-1]
        spike_times_reverse_relative_to_peak = spike_times_reverse - peak_time


        assert(np.all(spike_ids_reverse_relative>=0))
        result[spike_ids_reverse_relative, i] = spike_times_reverse_relative_to_peak

    return result

def spikes_2_rel_oscilation_spikes(spikes, network_architecture, time_range):
    """
    Transform absolute spiketimes to times of the first spike in an oscilation relative to it's peak. Seperatly for each population
    :param spikes: pandas with ids and times
    :param network_architecture:
    :param time_range: (start_time, end_time) time range whithin wich spikes will be transformed
    :return: dict with population name as key and a numpy array of shape (n_neurons, n_peaks) as item
    """
    population_spikes = helper.split_into_populations(spikes, network_architecture)
    population_ranges = helper.get_population_neuron_range(network_architecture)

    all_rel_spikes = dict()

    times = None
    for pop_name, pop_spikes in population_spikes.items():
        new_times, pop_activity = population_activity(pop_spikes, time_range=time_range)
        if times is None:
            times = new_times
        else:
            assert(np.all(times == new_times))

        peaks = fit_activity_peaks(pop_activity, times)
        rel_spikes = spikes_rel_to_population_peaks(population_spikes=pop_spikes, population_peaks=peaks, neuron_range=population_ranges[pop_name])

        all_rel_spikes[pop_name] = rel_spikes

    return all_rel_spikes




def spikes_2_population_peaks(spikes, network_architecture, time_range):
    """

    :return:
    """
    populations = helper.split_into_populations(spikes, network_architecture)

    # population activity
    times = None
    all_pop_activity = dict()
    for pop_name, pop_spikes in populations.items():
        new_times, pop_activity = population_activity(pop_spikes, time_range=time_range)
        if times is None:
            times = new_times
        else:
            assert(np.all(times == new_times))
        all_pop_activity[pop_name] = pop_activity
    # // population acitvity

    # peaks
    pop_peaks = dict()
    for pop_name, pop_activity in all_pop_activity.items():
        peaks = fit_activity_peaks(pop_activity, times, smooth_win=5e-2)
        pop_peaks[pop_name] = peaks

    return pop_peaks


