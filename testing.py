import numpy as np
import pandas as pd
import sys

import spikeAnalysisToolsV2.data_loading

sys.path.append("/Users/clemens/Documents/Code/AnalysisToolbox")

import matplotlib.pyplot as plt
import SpikeDataLoading as data
import Activation_Analysis as acan
from timeit import default_timer as timer



import Firing_Rates as firing
import Weights_Analysis as weights
import Information_Analysis as info


masterpath = "/Users/dev/Documents/Gisi/01_Spiking_Simulation/01_Spiking_Network/Build/output/"
masterpath = "/Users/clemens/Documents/Code/ModelClemens/Build/output"
## set the subfolder to the Simulation you want to analyse

subfolders = [
#      "ParameterTest_29_9_1_" + str(number_of_epochs) + "_epochs/"
#      "ParameterTest_1_" + str(number_of_epochs) + "_epochs/",
#      "ParameterTest_2_" + str(number_of_epochs) + "_epochs/",
#      "ParameterTest_3_" + str(number_of_epochs) + "_epochs/"
            "2017:10:18-12:56:59_test_100_epochs"

            ]
## if more than the inital epoch is needed *1 needs to be run
extensions = [
            "initial",
            "testing/epoch1",
            "testing/epoch2"
             ]


pandas_spikes = spikeAnalysisToolsV2.data_loading.load_spikes_from_subfolders(masterpath, subfolders, extensions, False)


# info_neurons is just an array of the information from above. This makes it easier to run the functions and pass the information.
# info_times same for times
info_neurons = dict(
    num_exc_neurons_per_layer = 64*64,
    num_inh_neurons_per_layer = 32*32,
    num_layers = 4,
    #total_per_layer = num_exc_neurons_per_layer + num_inh_neurons_per_layer,
    #total_network = total_per_layer * num_layers,
    #num_stimuli = 16
)

info_times = dict(
    length_of_stimulus = 2.0,
    num_stimuli = 16,
    time_start = 0,
    time_end = 2
)

layers_of_interest = [0, 1 ,2, 3]


one_stimulus = data.pandas_splitstimuli(pandas_spikes[0][2], 2.0)[2]


exc, inh = data.instant_FR_for_all_layers(one_stimulus, info_neurons, 0.2)


one_layer = acan.split_into_layers(one_stimulus, info_neurons)[1]
one_exci, one_inhi = acan.split_exc_inh(one_layer, info_neurons)
#excitatory of folder 0, extension 2, stimulus 1, layer 1
overal_rates = data.pandas_spikesToFR(one_exci, (0, 64*64), (0, 2.0))
start = timer()
instant_times, instant_FR = data.spikes_to_instantanious_FR(one_exci, (0, 64*64), 0.2, (0, 2.0))
print("instant FR for one layer and one stimulus took {} s".format(timer()-start))



# one_layer

start = timer()
pandas_rates_subfolders = acan.pandas_calculate_rates_subfolder(
    pandas_spikes,
    info_neurons,
    info_times,
    layers_of_interest,
    subfolders,
    extensions)
print("\n Pandas Version of Subfolder Rates took: {}s".format(timer() - start))



#pandas_rates_subfolders[0][0][5][3][0].firing_rates.values == all_subfolder_exc_rates[0][0][3][5]
#assert(np.all((pandas_rates_subfolders[0][0][5][3][0].firing_rates.values == all_subfolder_exc_rates[0][0][3][5])[1:]))
#the original implementation has a bug where the first neuron of each layer is ignored


# firing.clemens_make_firing_tables(pandas_rates_subfolders, info_times, subfolders, extensions, True)