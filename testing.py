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
import spikeAnalysisToolsV2.data_loading as data
import spikeAnalysisToolsV2.firing_rates as firing
import spikeAnalysisToolsV2.helper as helper
import spikeAnalysisToolsV2.overviews as overview
import spikeAnalysisToolsV2.combine_stimuli as combine
import spikeAnalysisToolsV2.plotting as spikeplot
import spikeAnalysisToolsV2.information_scores as info
import spikeAnalysisToolsV2.synapse_analysis as synapse_analysis




# one_stimulus = data.pandas_splitstimuli(pandas_spikes[0][2], 2.0)[2]
#
#
# exc, inh = data.instant_FR_for_all_layers(one_stimulus, info_neurons, 0.2)
#
#
# one_layer = acan.split_into_layers(one_stimulus, info_neurons)[1]
# one_exci, one_inhi = acan.split_exc_inh(one_layer, info_neurons)
# #excitatory of folder 0, extension 2, stimulus 1, layer 1
# overal_rates = data.pandas_spikesToFR(one_exci, (0, 64*64), (0, 2.0))
# start = timer()
# instant_times, instant_FR = data.spikes_to_instantanious_FR(one_exci, (0, 64*64), 0.2, (0, 2.0))
# print("instant FR for one layer and one stimulus took {} s".format(timer()-start))
#
#
#
# # one_layer
#
# start = timer()
# pandas_rates_subfolders = acan.pandas_calculate_rates_subfolder(
#     pandas_spikes,
#     info_neurons,
#     info_times,
#     layers_of_interest,
#     subfolders,
#     extensions)
# print("\n Pandas Version of Subfolder Rates took: {}s".format(timer() - start))



#pandas_rates_subfolders[0][0][5][3][0].firing_rates.values == all_subfolder_exc_rates[0][0][3][5]
#assert(np.all((pandas_rates_subfolders[0][0][5][3][0].firing_rates.values == all_subfolder_exc_rates[0][0][3][5])[1:]))
#the original implementation has a bug where the first neuron of each layer is ignored


# firing.clemens_make_firing_tables(pandas_rates_subfolders, info_times, subfolders, extensions, True)


def test_functional_freq_table():
    ## set the Masterpath to the folder where your output is saved

    masterpath = "/Users/clemens/Documents/Code/ModelClemens/output"
    ## set the subfolder to the Simulation you want to analyse

    subfolders = [
        "2017:10:20-16:50:34_only_first_location_123_epochs"
    ]
    ## if more than the inital epoch is needed *1 needs to be run
    extensions = [
        "testing/epoch5",
        "testing/epoch123"
    ]

    # info_neurons is just an array of the information from above. This makes it easier to run the functions and pass the information.
    # info_times same for times
    network_architecture = dict(
        num_exc_neurons_per_layer=64 * 64,
        num_inh_neurons_per_layer=32 * 32,
        num_layers=4,
        # total_per_layer = num_exc_neurons_per_layer + num_inh_neurons_per_layer,
        # total_network = total_per_layer * num_layers,
        # num_stimuli = 16
    )

    info_times = dict(
        length_of_stimulus=2.0,
        num_stimuli=16,
        time_start=1.5,
        time_end=1.9
    )

    spikes = data.load_spikes_from_subfolders(masterpath, subfolders, extensions, False)

    rates_subfolders = firing.calculate_rates_subfolder(
        spikes,
        network_architecture,
        info_times)
    final_epoch_rates = rates_subfolders[0][1]
    objects = [[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]]  # each is presented twice
    exc_rates, inh_rates = helper.nested_list_of_stimuli_2_np(final_epoch_rates)
    freq_table =  combine.response_freq_table(exc_rates, objects)
    single_cell_info = info.single_cell_information(freq_table)
    return single_cell_info


def test_animated_hist():
    ## set the Masterpath to the folder where your output is saved
    n_epochs = 19

    masterpath = "/Users/clemens/Documents/Code/ModelClemens/output"
    ## set the subfolder to the Simulation you want to analyse

    subfolders = [
        "10_25-19_10_only_loc_1"
    ]
    ## if more than the inital epoch is needed *1 needs to be run
    extensions = ["initial"] + ["testing/epoch{}".format(n) for n in range(1, n_epochs)]

    object_list = data.load_testing_stimuli_info(
        masterpath + "/" + subfolders[0])  # assuming all the subfolders have the same
    n_stimuli = np.sum(obj['count'] for obj in object_list)

    current_index = 0
    object_indices = []
    for obj in object_list:
        object_indices.append(list(range(current_index, current_index + obj['count'])))
        current_index += obj["count"]

    # info_neurons is just an array of the information from above. This makes it easier to run the functions and pass the information.
    # info_times same for times
    network_architecture = dict(
        num_exc_neurons_per_layer=64 * 64,
        num_inh_neurons_per_layer=32 * 32,
        num_layers=4,
        # total_per_layer = num_exc_neurons_per_layer + num_inh_neurons_per_layer,
        # total_network = total_per_layer * num_layers,
        # num_stimuli = 16
    )

    info_times = dict(
        length_of_stimulus=2.0,
        num_stimuli=n_stimuli,
        time_start=1.5,
        time_end=1.9
    )

    spikes = data.load_spikes_from_subfolders(masterpath, subfolders, extensions, False)
    start = timer()
    # rates_subfolders = firing.slow_calculate_rates_subfolder(
    #     spikes,
    #     network_architecture,
    #     info_times)
    # print("Non multiprocessing version took {}".format(timer() - start))

    start = timer()
    rates_subfolders = firing.calculate_rates_subfolder(
        spikes,
        network_architecture,
        info_times)
    print("Multiprocessing version took {}".format(timer() - start))
    exc_information, inhibitory_information = info.single_cell_information_all_epochs(rates_subfolders[0], object_indices, 3)
    ani = spikeplot.plot_animated_histogram(exc_information)
    import matplotlib.pyplot as plt
    plt.show()


def test_mutual_and_single_cell_info():
    ## set the Masterpath to the folder where your output is saved
    n_epochs = 188

    masterpath = "/Users/clemens/Documents/Code/ModelClemens/output"
    ## set the subfolder to the Simulation you want to analyse

    subfolders = [
        "11_05-20_04_loc1_both"
    ]
    ## if more than the inital epoch is needed *1 needs to be run
    extensions = ["initial"]  # + ["testing/epoch180"]

    object_list = data.load_testing_stimuli_info(
        masterpath + "/" + subfolders[0])  # assuming all the subfolders have the same
    n_stimuli = np.sum(obj['count'] for obj in object_list)

    # info_neurons is just an array of the information from above. This makes it easier to run the functions and pass the information.
    # info_times same for times
    network_architecture = dict(
        num_exc_neurons_per_layer=64 * 64,
        num_inh_neurons_per_layer=32 * 32,
        num_layers=4,
        # total_per_layer = num_exc_neurons_per_layer + num_inh_neurons_per_layer,
        # total_network = total_per_layer * num_layers,
        # num_stimuli = 16
    )

    info_times = dict(
        length_of_stimulus=2.0,
        num_stimuli=n_stimuli,
        time_start=1.5,
        time_end=1.9
    )

    # objects_in_training = [
    #     object_list[0]['indices'] + object_list[1]['indices'] + object_list[2]['indices'] + object_list[3]['indices'],
    #     object_list[4]['indices'] + object_list[5]['indices'] + object_list[6]['indices'] + object_list[7]['indices'],
    # ]
    # # These Objects were bound together in training with temporal trace. so it should have learned information about them.
    # print(objects_in_training)
    object_indices = [obj['indices'] for obj in object_list]

    spikes = data.load_spikes_from_subfolders(masterpath, subfolders, extensions, False)

    rates_subfolders = firing.calculate_rates_subfolder(
        spikes,
        network_architecture,
        info_times)
    exh_mutual_info, inh_mutual_info = info.firing_rates_to_mutual_information(rates_subfolders[0][0], object_indices, 3, calc_inhibitory=True)
    exc_single_cell, inh_single_cell = info.firing_rates_to_single_cell_information(rates_subfolders[0][0], object_indices, 3, calc_inhibitory=True)

    assert(np.all(np.isclose(exh_mutual_info[0], np.mean(exc_single_cell, axis=0))))
    assert(np.all(np.isclose(inh_mutual_info[0], np.mean(inh_single_cell, axis=0))))





def test_network_loading():
    path = "/Users/clemens/Documents/Code/ModelClemens/output/11_06-15_00_loc1_centered"
    subfolder = ["initial"] + ["testing/epoch{}".format(e) for e in range(1, 30)]

    start = timer()
    net, weights = data.load_weights_all_epochs(path, range(1,30))
    print("took {} ".format(timer() - start))

    # test that the weights are always on the same position

    return net, weights

def test_weighted_presynaptic_firing_rates():
    path = "/Users/clemens/Documents/Code/ModelClemens/output/"
    subfolders = ["11_06-15_00_loc1_centered"]
    extensions = ["initial"] + ["testing/epoch{}".format(e) for e in range(1, 30)]

    net, weights = data.load_weights_all_epochs(path+subfolders[0], range(1,30))

    object_list = data.load_testing_stimuli_info(path+subfolders[0])  # assuming all the subfolders have the same
    n_stimuli = np.sum(obj['count'] for obj in object_list)
    object_indices = [obj['indices'] for obj in object_list]

    network_architecture = dict(
        num_exc_neurons_per_layer=64 * 64,
        num_inh_neurons_per_layer=32 * 32,
        num_layers=4
    )

    info_times = dict(
        length_of_stimulus=2.0,
        num_stimuli=n_stimuli,
        time_start=1.5,
        time_end=1.9
    )

    spikes = data.load_spikes_from_subfolders(path, subfolders, extensions, False)

    rates_subfolders = firing.calculate_rates_subfolder(
        spikes,
        network_architecture,
        info_times)

    exc_rates, inh_rates = helper.nested_list_of_epochs_2_np(rates_subfolders[0])

    mymask = synapses.Synapse_Mask(network_architecture, net)

    neuron_id = 5939

    excitatory_mask = np.invert(mymask.inh_lateral())

    overall_mask = excitatory_mask & (net.post.values == neuron_id)

    weighted_current = synapses.weighted_presynaptic_actvity(overall_mask, net=net, weights=weights, firing_rates=(exc_rates, inh_rates))

    return weighted_current


def test_paths_to_neuron():
    path = "/Users/clemens/Documents/Code/ModelClemens/output/11_29-01_52_white_circle_l_vs_r_ALS_smallSTDP/initial"

    network_architecture = dict(num_inh_neurons_per_layer=32 * 32, num_exc_neurons_per_layer=64 * 64, num_layers=4)
    synapses = data.load_network(path, True, True)
    mask = synapse_analysis.Synapse_Mask(network_architecture, synapses)

    filter_path = "/Users/clemens/Documents/Code/ModelClemens/Data/MatlabGaborFilter/centered_inputs/Filtered"
    all_filter = data.load_filter_all_obj(filter_path)
    bla = synapse_analysis.paths_to_neurons([-3485], synapses, 0.9, max_path_length=3)
    return bla




wc = test_paths_to_neuron()
# net, w = test_network_loading()