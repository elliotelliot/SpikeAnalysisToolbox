import numpy as np
import pandas as pd
import Activation_Analysis


def make_firing_tables(info_times, layers_of_interest, subfolders, extensions, table_all_subfolders,
                       detailed_stimuli):
    # Set Information for Columns
    columns1 = 'Exc Mean', 'Exc Min', 'Exc Max', 'Inh Mean', 'Inh Min', 'Inh Max'
    index1 = ['Stimulus ' + str(i) for i in range(info_times["num_stimuli"])]

    columns2 = 'Exc Mean', 'Exc Min', 'Exc Max', 'Exc Diff', 'Inh Mean', 'Inh Min', 'Inh Max', 'Inh Diff'
    index2 = ["Layer " + str(layers_of_interest[i]) for i in range(len(layers_of_interest))]

    for subfolder in range(len(subfolders)):
        print(subfolders[subfolder])
        for ext_index in range(len(extensions)):

            diff_exc = list()
            diff_inh = list()
            means_overview = list()
            df2 = None

            mean_exc = list()
            mean_inh = list()

            min_mean_exc = list()
            min_mean_inh = list()

            max_mean_exc = list()
            max_mean_inh = list()

            print("Extension: ", extensions[ext_index])
            for layer in range(len(layers_of_interest)):
                df = pd.DataFrame(data=table_all_subfolders[subfolder][ext_index][layer], columns=columns1,
                                  index=index1)

                mean_exc.append(df['Exc Mean'].mean())
                mean_inh.append(df['Inh Mean'].mean())

                min_mean_exc.append(df['Exc Max'].min())
                min_mean_inh.append(df['Inh Max'].min())

                max_mean_exc.append(df['Exc Max'].max())
                max_mean_inh.append(df['Inh Max'].max())

                if detailed_stimuli:
                    print("Layer", layers_of_interest[layer], "\n", df)
                    print("\n")
            diff_exc = list(np.asarray(max_mean_exc) - np.asarray(min_mean_exc))
            diff_inh = list(np.asarray(max_mean_inh) - np.asarray(min_mean_inh))

            means_overview = list(zip(mean_exc, min_mean_exc, max_mean_exc, diff_exc,
                                      mean_inh, min_mean_inh, max_mean_inh, diff_inh))

            df2 = pd.DataFrame(data=means_overview, columns=columns2, index=index2)
            print("***Overview***\n")
            print(df2, '\n')









