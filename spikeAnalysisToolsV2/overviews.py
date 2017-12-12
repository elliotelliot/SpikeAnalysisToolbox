import numpy as np
import pandas as pd

from . import helper

def print_firing_rates_tables(subfolder_firing_rates, info_times, subfolders, extensions,
                       detailed_stimuli):
    """
    Prints Summary for Firing rates in the diferent layers. for all subfolders, extensions, stimuli

    """

    n_layers = len(subfolder_firing_rates[0][0][0])

    # Set Information for Columns
    columns1 = 'Exc Mean', 'Exc Min', 'Exc Max', 'Inh Mean', 'Inh Min', 'Inh Max'
    index1 = ['Stimulus ' + str(i) for i in range(info_times["num_stimuli"])]

    columns2 = 'Exc Mean', 'Exc Min', 'Exc Max', 'Exc Diff', 'Inh Mean', 'Inh Min', 'Inh Max', 'Inh Diff'
    index2 = ["Layer " + str(i) for i in range(n_layers)]

    for subfolder in range(len(subfolders)):
        print(subfolders[subfolder])
        for ext_index in range(len(extensions)):

            exc_firing_rates_tensor, inh_firing_rates_tensor = helper.nested_list_of_stimuli_2_np(subfolder_firing_rates[subfolder][ext_index])
            # have dimensions [stimulus, layer, neuronid]


            layerwise_mean_exc = np.mean(exc_firing_rates_tensor, axis=2) # [stimulus, layer] contains mean of that layer
            layerwise_min_exc  = np.min(exc_firing_rates_tensor, axis=2) # [stimulus, layer] contains mean of that layer
            layerwise_max_exc  = np.max(exc_firing_rates_tensor, axis=2) # [stimulus, layer] contains mean of that layer

            layerwise_mean_inh = np.mean(inh_firing_rates_tensor, axis=2) # [stimulus, layer] contains mean of that layer
            layerwise_min_inh  = np.min(inh_firing_rates_tensor, axis=2) # [stimulus, layer] contains mean of that layer
            layerwise_max_inh  = np.max(inh_firing_rates_tensor, axis=2) # [stimulus, layer] contains mean of that layer


            mean_exc = list()
            mean_inh = list()

            min_mean_exc = list()
            min_mean_inh = list()

            max_mean_exc = list()
            max_mean_inh = list()

            print("Extension: ", extensions[ext_index])
            for layer in range(n_layers):

                df = pd.DataFrame(data=np.zeros((len(index1), len(columns1))), columns=columns1,
                                  index=index1)


                df['Exc Mean'] = layerwise_mean_exc[:, layer]
                df['Exc Min']  = layerwise_min_exc[:, layer]
                df['Exc Max']  = layerwise_max_exc[:, layer]

                df['Inh Mean'] = layerwise_mean_inh[:, layer]
                df['Inh Min']  = layerwise_min_inh[:, layer]
                df['Inh Max']  = layerwise_max_inh[:, layer]


                mean_exc.append(df['Exc Mean'].mean())
                mean_inh.append(df['Inh Mean'].mean())

                min_mean_exc.append(df['Exc Max'].min())
                min_mean_inh.append(df['Inh Max'].min())

                max_mean_exc.append(df['Exc Max'].max())
                max_mean_inh.append(df['Inh Max'].max())

                if detailed_stimuli:
                    print("Layer", layer, "\n", df)
                    print("\n")
            diff_exc = list(np.asarray(max_mean_exc) - np.asarray(min_mean_exc))
            diff_inh = list(np.asarray(max_mean_inh) - np.asarray(min_mean_inh))

            means_overview = list(zip(mean_exc, min_mean_exc, max_mean_exc, diff_exc,
                                      mean_inh, min_mean_inh, max_mean_inh, diff_inh))

            df2 = pd.DataFrame(data=means_overview, columns=columns2, index=index2)
            print("***Overview***\n")
            print(df2, '\n')

