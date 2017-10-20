import numpy as np
import pandas as pd

import SpikeDataLoading as data


def calculate_rates_subfolder_layer(ids, times, info_neurons, info_times, layers_of_interest):
    layerids = list()
    layertimes = list()
    layer_exc_rates = list()
    layer_inh_rates = list()

    num_exc_neurons_per_layer = info_neurons["num_exc_neurons_per_layer"]
    num_inh_neurons_per_layer = info_neurons["num_inh_neurons_per_layer"]
    total_per_layer = num_exc_neurons_per_layer + num_inh_neurons_per_layer

    length_of_stimulus = info_times["length_of_stimulus"]
    total_length = length_of_stimulus * info_times["num_stimuli"]
    timestart = info_times["time_start"]
    timeend = info_times["time_end"]

    ids = ids[times < total_length]
    times = times[times < total_length]

    layerids.append([])
    layertimes.append([])
    layer_exc_rates.append([])
    layer_inh_rates.append([])

    for layer_id in layers_of_interest:
        print("\t" + str(layer_id), end="")
        mask = (ids > (layer_id * total_per_layer)) & (ids < ((layer_id + 1) * total_per_layer)) # tiny but first one should be >= this way the first neuron of every layer is ignored

        layerids[-1].append(ids[mask] - layer_id * total_per_layer)
        layertimes[-1].append(times[mask])

        l_ids, l_times = data.splitstimuli(layerids[-1][-1], layertimes[-1][-1], length_of_stimulus)
        layerids[-1][-1] = l_ids
        layertimes[-1][-1] = l_times

        layer_exc_rates[-1].append(
            data.spikesToFR(layerids[-1][layers_of_interest.index(layer_id)],
                            layertimes[-1][layers_of_interest.index(layer_id)],
                            0,
                            num_exc_neurons_per_layer,
                            timestart, timeend, False))
        layer_inh_rates[-1].append(
            data.spikesToFR(layerids[-1][layers_of_interest.index(layer_id)],
                            layertimes[-1][layers_of_interest.index(layer_id)],
                            num_exc_neurons_per_layer,
                            total_per_layer,
                            timestart, timeend, False))
    return layerids, layertimes, layer_exc_rates, layer_inh_rates







def calculate_rates_subfolder(import_all_subfolders_ids, import_all_subfolders_times, info_neurons, info_times,
                              layers_of_interest, subfolders, extensions):
    # subfolder = 0
    # extension = 0



    subfolder_ids = list()
    subfolder_times = list()
    subfolder_exc_rates = list()
    subfolder_inh_rates = list()

    for subfolder in range(len(subfolders)):
        print("\nSubfolder: " + str(subfolders[subfolder]))
        extension_ids = list()
        extension_times = list()
        extension_exc_rates = list()
        extension_inh_rates = list()

        for extension in range(len(extensions)):
            extension_ids.append([])
            extension_times.append([])
            extension_exc_rates.append([])
            extension_inh_rates.append([])
            print("\nExtension: " + str(extensions[extension]))
            sub_ids, sub_times, sub_exc_rates, sub_inh_rates = calculate_rates_subfolder_layer(
                import_all_subfolders_ids[subfolder][extension],
                import_all_subfolders_times[subfolder][extension],
                info_neurons,
                info_times,
                layers_of_interest)
            extension_ids[-1] = sub_ids[0]
            extension_times[-1] = sub_times[0]
            extension_exc_rates[-1] = sub_exc_rates[0]
            extension_inh_rates[-1] = sub_inh_rates[0]
        subfolder_ids.append(extension_ids)
        subfolder_times.append(extension_times)
        subfolder_exc_rates.append(extension_exc_rates)
        subfolder_inh_rates.append(extension_inh_rates)
    return subfolder_ids, subfolder_times, subfolder_exc_rates, subfolder_inh_rates






def make_freq_table(all_subfolder_exc_rates, all_subfolder_inh_rates, layers_of_interest, subfolders, extensions,
                    num_stimuli):
    table_all_subfolders = list()

    for subfolder in range(len(subfolders)):
        print("Subfolder: " + str(subfolders[subfolder]))
        table_all_extensions = list()

        for ext_index in range((len(extensions))):
            # table_all_extensions.append([])
            print("Start Extension: " + str(extensions[ext_index]))

            all_layers = list()
            ### Calculate mean, min, max for each layer
            for layer in range(len(layers_of_interest)):
                ### Calculate mean, min,max for each stimulus
                all_stimulus_mean_exci = list()
                all_stimulus_min_exci = list()
                all_stimulus_max_exci = list()

                all_stimulus_mean_inh = list()
                all_stimulus_min_inh = list()
                all_stimulus_max_inh = list()

                for stimulus in range(num_stimuli):
                    ## Excitatory
                    current_mean_exci = np.mean(all_subfolder_exc_rates[subfolder][ext_index][layer][stimulus])
                    all_stimulus_mean_exci.append(current_mean_exci)

                    current_min_exci = np.min(all_subfolder_exc_rates[subfolder][ext_index][layer][stimulus])
                    all_stimulus_min_exci.append(current_min_exci)

                    current_max_exci = np.max(all_subfolder_exc_rates[subfolder][ext_index][layer][stimulus])
                    all_stimulus_max_exci.append(current_max_exci)

                    ## Inhibitory
                    current_mean_inh = np.mean(all_subfolder_inh_rates[subfolder][ext_index][layer][stimulus])
                    all_stimulus_mean_inh.append(current_mean_inh)

                    current_min_inh = np.min(all_subfolder_inh_rates[subfolder][ext_index][layer][stimulus])
                    all_stimulus_min_inh.append(current_min_inh)

                    current_max_inh = np.max(all_subfolder_inh_rates[subfolder][ext_index][layer][stimulus])
                    all_stimulus_max_inh.append(current_max_inh)

                current_layer = list(zip(all_stimulus_mean_exci, all_stimulus_min_exci, all_stimulus_max_exci,
                                         all_stimulus_mean_inh, all_stimulus_min_inh, all_stimulus_max_inh))
                all_layers.append(current_layer)
            table_all_extensions.append(all_layers)
        # table_all_extensions[-1] = all_layers


        table_all_subfolders.append(table_all_extensions)
    return table_all_subfolders
