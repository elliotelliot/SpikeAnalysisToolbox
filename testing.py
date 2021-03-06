import sys

import numpy as np

sys.path.append("/Users/clemens/Documents/Code/AnalysisToolbox")

import matplotlib.pyplot as plt
from timeit import default_timer as timer

import data_loading as data
import firing_rates as firing
import helper as helper
import combine_stimuli as combine
import plotting as spikeplot
import information_scores as info
import synapse_analysis as synapse_analysis
import polychron as poly
import oscilations as oscilations



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

def test_trace_to_gabor():
    path = "/Users/clemens/Documents/Code/ModelClemens/output/01_19-18_02_rounded_trained_after_random_pretraining/multi_e275"

    network_architecture = dict(num_inh_neurons_per_layer=32 * 32, num_exc_neurons_per_layer=64 * 64, num_layers=4)
    synapses = data.load_network(path, True, True)
    mask = synapse_analysis.Synapse_Mask(network_architecture, synapses)

    e2e_ff = mask.exc_feed_forward()

    bt = synapse_analysis.BackTracer(synapses, network_architecture, percentage_thresholds=[0.1, 0.1, 0.1, 0.1], mask=e2e_ff)

    result = bt.trace_back(3, 32, 32)
    return result


def shuffle_weight(path):
    # path = "/Users/clemens/Documents/Code/ModelClemens/output/11_29-01_52_white_circle_l_vs_r_ALS_smallSTDP/initial"
    network_architecture = dict(num_inh_neurons_per_layer=32 * 32, num_exc_neurons_per_layer=64 * 64, num_layers=4)
    synapses = data.load_network(path, True, True)

    new_weights = synapse_analysis.shuffle_weights_within_layer(synapses, network_architecture, 128 ** 2)

    raw_new_weights = new_weights.weights.values
    raw_new_weights.tofile("{}/shuffled_weights.bin".format(path))
    # return new_weights

def test_single_cell_decoder():
    path = "/Users/clemens/Documents/Code/ModelClemens/output"
    experiment = "12_27-18_57_all_no_trace"
    extension = "testing/epoch300"

    object_list = data.load_testing_stimuli_indices_from_wildcarts(path + "/" + experiment, ["***l", "***r"])
    print(object_list)
    object_indices = [o['indices'] for o in object_list]
    print(object_indices)
    n_stimuli = np.sum([o['count'] for o in object_list])

    label_for_classifier = np.zeros((2, n_stimuli), dtype=bool)
    for i, o in enumerate(object_list):
        label_for_classifier[i, o['indices']]= True


    label_for_classifier_from_data_loading = data.load_testing_stimuli_label_matrix(path + "/" + experiment, ["***l", "***r"])
    assert(np.all(label_for_classifier == label_for_classifier_from_data_loading))

    print(label_for_classifier)


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


    spikes = data.load_spikes_from_subfolders(path, [experiment], [extension], False)

    rates_subfolders = firing.calculate_rates_subfolder(
        spikes,
        network_architecture,
        info_times)

    exc_rates, inh_rates = helper.nested_list_of_stimuli_2_np(rates_subfolders[0][0])

    dec = info.SingleCellDecoder()
    perf_train = dec.fit(exc_rates, label_for_classifier)

    perf_sumary = dec.get_performance_summary(exc_rates, label_for_classifier)

    perf_test = perf_sumary.accuracy()

    assert(np.all(perf_train == perf_test))
    print(perf_train == perf_test)
    print("done")


def test_spike_correlations():
    path = "/Users/clemens/Documents/Code/ModelClemens/output"
    experiment = "01_11-15_00_long_test_with_trace"
    extension = "trained_e285"

    stimuli_ids = data.load_testing_stimuli_dict(path + "/" + experiment + "/" + extension)

    network_architecture = dict(
        num_exc_neurons_per_layer=64 * 64,
        num_inh_neurons_per_layer=32 * 32,
        num_layers=4
    )

    neuron_mask = helper.NeuronMask(network_architecture)

    synapses = data.load_network(path + "/" + experiment + "/" + extension)

    spikes = data.load_spikes_from_subfolders(path, [experiment], [extension], False)

    seperated = helper.splitstimuli(spikes[0][0], 2)

    selected = helper.take_multiple_elements_from_list(seperated, stimuli_ids["1wcr"])

    target_id = helper.position_to_id((1, 19, 10), True, network_architecture)
    target_id = 9186

    # pre_ids = neuron_mask.get_ids_of_random_neurons_of_type(200, neuron_mask.is_excitatory)
    pre_ids = synapses.pre.values[synapses.post.values == target_id]

    pre_ids = np.sort(np.unique(pre_ids))

    normed_hist, times = poly.population_normalised_pre_spike_hist(stimuli_spikes=selected, target_neuron=target_id, max_time_delay=0.2,
                                             time_step=0.002, network_architecture=network_architecture,
                                             preneurons=pre_ids, start_time=1.0, multiple_spikes_per_bin_possible=True)

    hist_new, times = poly.pre_spike_hist(stimuli_spikes=selected, target_neuron=target_id, max_time_delay=0.2, time_step=0.1, preneurons=pre_ids, start_time=1.0, multiple_spikes_per_bin_possible=True)
    hist_old, times = poly.pre_spike_hist(stimuli_spikes=selected, target_neuron=target_id, max_time_delay=0.02, time_step=0.1, preneurons=pre_ids, start_time=1.0, multiple_spikes_per_bin_possible=False)

    assert(np.all(hist_new >= hist_old))

    print(hist_new.shape)


def test_oscilation_fitter():
    path = "/Users/shakti/Documents/OFTNAI/test_data"
    # experiment ="03_06-12_46_long_test_smaller_syn_decay"
    experiment ="01_11-15_00_long_test_with_trace"
    extension = "trained_e285"

    # stimuli_ids = data.load_testing_stimuli_dict(path + "/" + experiment + "/" + extension)

    network_architecture = dict(
        num_exc_neurons_per_layer=64 * 64,
        num_inh_neurons_per_layer=32 * 32,
        num_layers=4
    )

    # neuron_mask = helper.NeuronMask(network_architecture)

    # synapses = data.load_network(path + "/" + experiment + "/" + extension)

    spikes = data.load_spikes_from_subfolders(path, [experiment], [extension], False)[0][0]

    # seperated = helper.splitstimuli(spikes, 2)

    pops = helper.split_into_populations(spikes, network_architecture)
    print(pops)

    # import scipy.signal as ssignal

    stimulus_id = 91

    pop_name = "L0_exc"

    times, activity = oscilations.population_activity(pops[pop_name], (2 * stimulus_id + 1, 2 * stimulus_id + 2), bin_width=4e-3)

    # activity = ssignal.convolve(activity, ssignal.gaussian(10, 1e-3/ 4e-3), mode='same')

    peaks = oscilations.fit_activity_peaks(activity, times)
    # extrema = oscilations.fit_sinus_peaks(activity, times)

    # activity = 3 * np.cos(62 * times + 23)

    neuron_ranges = helper.get_popluation_neuron_range(network_architecture)

    peak_rel_spiketimes = oscilations.spikes_rel_to_population_peaks(pops[pop_name], peaks, neuron_ranges[pop_name])


    frex, exp, (frequencies, intensities) = oscilations.fit_fft(activity, times, smooth_win_hz=4)


    plt.figure()
    plt.plot(frequencies, intensities)

    # print(activity)
    plt.figure()
    plt.plot(times, activity)

    # compute sinus
    # sin_y =  scale * np.cos(frex * times * 2 * np.pi + offset)

    exp_y = np.exp(2j * np.pi * times * frex) * exp

    # plt.plot(times, sin_y)
    plt.plot(times, exp_y)
    # for ex in extrema:
    #     plt.axvline(ex, c='r')

    plt.show()
    print('done')


def test_spike_pair_info():
    path = "/Users/clemens/Documents/Code/ModelClemens/output"
    # experiment ="03_06-12_46_long_test_smaller_syn_decay"
    experiment ="01_11-15_00_long_test_with_trace"
    extension = "trained_e285"

    object_list = data.load_testing_stimuli_indices_from_wildcarts(path + "/" + experiment, ["***l", "***r"])
    print(object_list)
    # stimuli_ids = data.load_testing_stimuli_dict(path + "/" + experiment + "/" + extension)
    object_indices = [o['indices'] for o in object_list]
    n_stimuli = len(np.array(object_indices).flatten())
    network_architecture = dict(
        num_exc_neurons_per_layer=64 * 64,
        num_inh_neurons_per_layer=32 * 32,
        num_layers=4
    )

    # neuron_mask = helper.NeuronMask(network_architecture)

    # synapses = data.load_network(path + "/" + experiment + "/" + extension)
    neuron_range = (15860, 16000)

    neuron_ids = np.arange(*neuron_range)

    spikes = data.load_spikes_from_subfolders(path, [experiment], [extension], False)[0][0]
    spikes = spikes[np.isin(spikes.ids.values, np.arange(*neuron_range))]

    splitted_spikes = helper.splitstimuli(spikes, 2, num_stimuli=n_stimuli)

    pair_info, t = poly.spike_pair_info(splitted_spikes, neuron_ids=neuron_ids, time_step=1e-3, max_time_delay=10e-3, start_time=1.0, n_bins_prob_table=3, object_indices=object_indices)
    t_pair_info, t = poly.threaded_spike_pair_info(splitted_spikes, neuron_ids=neuron_ids, time_step=1e-3, max_time_delay=10e-3, start_time=1.0, n_bins_prob_table=3, object_indices=object_indices, n_threads=4, post_min_spike_count=0)

    assert(np.all(t_pair_info==pair_info))

    return pair_info




# wc = test_paths_to_neuron()
# net, w = test_network_loading()
# nw = test_weight_suffeling()
# test_single_cell_decoder()
# p = test_trace_to_gabor()

# test_spike_correlations()
# out = test_oscilation_fitter()
res = test_spike_pair_info()