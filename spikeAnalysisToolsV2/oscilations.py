import numpy as np

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

    activity, _b = np.histogram(selected_spike_times, bin_edges)

    times = ( bin_edges[1:] + bin_edges[:-1])/2

    avg_firing_rate = (activity/bin_width)/n_neurons

    return times, avg_firing_rate