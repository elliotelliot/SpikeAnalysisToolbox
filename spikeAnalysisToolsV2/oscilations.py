import numpy as np
import scipy.signal as ssignal

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
    max_frequency_id = max_frequency_id_candidates[ft_intensity[max_frequency_id_candidates] > max_frequency_intensity/2][0]

    # if False:
    #     act_peaks = fit_activity_peaks(population_act, times)
    #     d_act_peaks = act_peaks[1:] - act_peaks[:-1]
    #     frequency = 1/np.median(d_act_peaks)
    #     max_frequency_id = np.argmin(np.abs(frequency-ft_frex))

    max_frequency = ft_frex[max_frequency_id]

    # max_offset = np.angle(ft[max_frequency_id])

    # max_intensity = ft_intensity[max_frequency_id]/len(population_act)

    offset_correction = np.exp(2j * np.pi * max_frequency * (-times[0]))

    faktor = ft[max_frequency_id] * offset_correction
    faktor /= len(population_act)

    return max_frequency, faktor, (ft_frex, ft_intensity)

def fit_activity_peaks(population_act, times, smooth_win=1e-3, min_peak_intensity=0.3):
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



#TODO comput time between consecutive fitted activity peaks. mean squared error between that and 1/max_frex


