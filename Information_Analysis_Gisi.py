import numpy as np
import sys

sys.path.append("/Users/dev/Documents/Gisi/02_Analysis/03_CodeKrams/04_HackSunday/SpikeTrainAnalysis")
import SpikeDataLoading as data
import InformationAnalysis


def folder_info_analysis_main(masterpath, subfolders, extensions, info_neurons, info_times, firing_rate_start_time,
                              firing_rate_end_time, num_categories, num_stimuli):
    exc_neurons = info_neurons[0]
    num_inhi = info_neurons[1]
    layers = info_neurons[2]
    num_layers = info_neurons[3]

    # Extract rates for the first layers
    sizeoflayer = info_neurons[3]
    num_layers = info_neurons[2]
    exc_neurons = info_neurons[0]

    stimulus_duration = info_times[0]

    total_stimuli_lengths = stimulus_duration * num_stimuli
    category_matrix = np.zeros((num_stimuli, num_categories))

    category_matrix[0, :] = np.array([0, 1, 0, 1])
    category_matrix[1, :] = np.array([0, 1, 1, 1])
    category_matrix[2, :] = np.array([0, 1, 0, 0])
    category_matrix[3, :] = np.array([0, 1, 1, 0])
    category_matrix[4, :] = np.array([1, 1, 0, 1])
    category_matrix[5, :] = np.array([1, 1, 1, 1])
    category_matrix[6, :] = np.array([1, 1, 0, 0])
    category_matrix[7, :] = np.array([1, 1, 1, 0])
    category_matrix[8, :] = np.array([0, 0, 0, 1])
    category_matrix[9, :] = np.array([0, 0, 1, 1])
    category_matrix[10, :] = np.array([0, 0, 0, 0])
    category_matrix[11, :] = np.array([0, 0, 1, 0])
    category_matrix[12, :] = np.array([1, 0, 0, 1])
    category_matrix[13, :] = np.array([1, 0, 1, 1])
    category_matrix[14, :] = np.array([1, 0, 0, 0])
    category_matrix[15, :] = np.array([1, 0, 1, 0])

    subfolder_information = list()

    for subfolder in subfolders:
        print(subfolder + "\n")
        epoch_information = list()
        for extension in extensions:
            print(extension, "", end='')
            epoch_information.append([])
            for layer_index in range(num_layers):
                neuronstart = layer_index * (sizeoflayer)
                neuronend = layer_index * (sizeoflayer) + exc_neurons

                epoch_information[-1].append(folder_info_analysis(
                    masterpath,
                    subfolder,
                    extension,
                    neuronstart,
                    neuronend,
                    firing_rate_start_time,
                    firing_rate_end_time,
                    stimulus_duration,
                    total_stimuli_lengths,
                    category_matrix))
        subfolder_information.append(epoch_information)
    return subfolder_information


def folder_info_analysis_main_trace_learning(masterpath, subfolders, extensions, info_neurons, info_times,
                                             firing_rate_start_time, firing_rate_end_time, num_categories, num_stimuli):
    exc_neurons = info_neurons[0]
    num_inhi = info_neurons[1]
    layers = info_neurons[2]
    num_layers = info_neurons[3]

    # Extract rates for the first layers
    sizeoflayer = info_neurons[3]
    num_layers = info_neurons[2]
    exc_neurons = info_neurons[0]

    stimulus_duration = info_times[0]

    total_stimuli_lengths = stimulus_duration * num_stimuli
    category_matrix = np.zeros((num_stimuli, num_categories))

    category_matrix[0, :] = np.array(([1, 0, 0, 0]))
    category_matrix[1, :] = np.array(([1, 0, 0, 0]))
    category_matrix[2, :] = np.array(([1, 0, 0, 0]))
    category_matrix[3, :] = np.array(([1, 0, 0, 0]))
    category_matrix[4, :] = np.array(([0, 1, 0, 0]))
    category_matrix[5, :] = np.array(([0, 1, 0, 0]))
    category_matrix[6, :] = np.array(([0, 1, 0, 0]))
    category_matrix[7, :] = np.array(([0, 1, 0, 0]))
    category_matrix[8, :] = np.array(([0, 0, 1, 0]))
    category_matrix[9, :] = np.array(([0, 0, 1, 0]))
    category_matrix[10, :] = np.array(([0, 0, 1, 0]))
    category_matrix[11, :] = np.array(([0, 0, 1, 0]))
    category_matrix[12, :] = np.array(([0, 0, 0, 1]))
    category_matrix[13, :] = np.array(([0, 0, 0, 1]))
    category_matrix[14, :] = np.array(([0, 0, 0, 1]))
    category_matrix[15, :] = np.array(([0, 0, 0, 1]))

    subfolder_information = list()

    for subfolder in subfolders:
        print(subfolder + "\n")
        epoch_information = list()
        for extension in extensions:
            print(extension, "", end='')
            epoch_information.append([])
            for layer_index in range(num_layers):
                neuronstart = layer_index * (sizeoflayer)
                neuronend = layer_index * (sizeoflayer) + exc_neurons

                epoch_information[-1].append(folder_info_analysis(
                    masterpath,
                    subfolder,
                    extension,
                    neuronstart,
                    neuronend,
                    firing_rate_start_time,
                    firing_rate_end_time,
                    stimulus_duration,
                    total_stimuli_lengths,
                    category_matrix))
        subfolder_information.append(epoch_information)
    return subfolder_information


def folder_info_analysis(masterpath, subfolder, extension, neuronstart, neuronend, firing_rate_start_time,
                         firing_rate_end_time, stimulus_duration, total_stimuli_lengths, category_matrix):
    # Using the current folder name (e.g. testing/epochX), get spikes
    currentpath = masterpath + subfolder + extension
    ids, times = data.get_spikes(currentpath, True)

    # Preprocessing to remove excess spikes etc
    ids = np.asarray(ids)
    times = np.asarray(times)
    ids = ids[times <= total_stimuli_lengths]
    times = times[times <= total_stimuli_lengths]

    # Converts the long spike train into individual stimuli
    ids, times = data.splitstimuli(ids, times, stimulus_duration)

    # Temporary lists to store only the current layer
    ids_tmp = list()
    times_tmp = list()

    # Extract current layer
    for stimulus in range(len(ids)):
        bool_tmp = (ids[stimulus] >= neuronstart) & (ids[stimulus] <= neuronend)
        ids_tmp.append(ids[stimulus][bool_tmp])
        times_tmp.append(times[stimulus][bool_tmp])

    # Convert spikes to rates (in the time range that we want)
    num_stimuli = len(ids)
    rates = data.spikesToFR(ids_tmp, times_tmp, neuronstart, neuronend, firing_rate_start_time, firing_rate_end_time,
                            False)
    # rates is list [stimulus]-> numpy array of all rates

    # Removing any bias in FR
    for n in range(num_stimuli):
        # Individually remove the means
        rates[n] -= np.mean(rates[n]) # mean of all neurons within that stimulus
        
    neuronmeans = np.mean(np.asarray(rates), axis=0)

    assert(np.all(np.isclose(neuronmeans, 0)))

    for n in range(num_stimuli):
        rates[n] -= neuronmeans

    # For each stimulus
    filtered_rates = list()
    for stimulus_index in range(num_stimuli):
        ratemean = np.mean(rates[stimulus_index]) # uses mean of ALL neurons for ONE stimulus
        assert(np.isclose((ratemean, 0)))
        ratestd = np.std(rates[stimulus_index]) # std of all neurons within stimulus
        filtered_rates.append(np.zeros(rates[stimulus_index].shape))

        # Use the Standard Dev to assign > or < or = mean
        filtered_rates[stimulus_index][rates[stimulus_index] > (ratemean + ratestd)] = 1.0
        filtered_rates[stimulus_index][(rates[stimulus_index] < (ratemean - ratestd))] = -1.0

    response_matrix = np.asarray(filtered_rates)

    # Calculate the frequence (occurrence) - how many times a neuron is >, <, = mean for each stimulus
    freq_table = InformationAnalysis.frequency_table(response_matrix, category_matrix, [-1, 0, 1])
    # Based on the frequency table, calculate how much information the neuron has about stimuli
    information = InformationAnalysis.single_cell_information(freq_table, category_matrix)

    return (information)


def subfolder_information_calc(subfolders, extensions, num_categories, info_neurons, subfolder_information,
                               information_value):
    num_layers = num_layers = info_neurons[2]

    subfolder_mean_information = list()
    subfolder_max_information_count = list()

    for subfolder in range(len(subfolders)):

        meaninformation = []
        maxinformation_count = []

        for layer in range(num_layers):
            meaninformation.append([])
            maxinformation_count.append([])

            for cat_index in range(num_categories):
                meaninformation[layer].append([])
                maxinformation_count[layer].append([])

                for extension in range(len(extensions)):
                    meaninformation[layer][cat_index].append(
                        np.mean(subfolder_information[subfolder][extension][layer][cat_index].flatten()))
                    maxinformation_count[layer][cat_index].append(np.sum(
                        subfolder_information[subfolder][extension][layer][cat_index].flatten() >= information_value))

        subfolder_mean_information.append(meaninformation)
        subfolder_max_information_count.append(maxinformation_count)

    return subfolder_mean_information, subfolder_max_information_count
