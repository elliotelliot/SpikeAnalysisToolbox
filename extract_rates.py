import numpy as np
import SpikeDataLoading as data
import InformationAnalysis as info
import os
# Library to parallelize rate loop
from joblib import Parallel, delayed


def cut_to_time_periode(ids, times):
    start_time = 0.5
    end_time = 1.0
    times_cut = list()
    ids_cut = list()

    start_time = float(start_time)
    end_time = float(end_time)

    for n in range(len(times)):
        # get current ids and times that a greater than specific time
        current_bool = [(start_time <= times[n]) & (start_time <= end_time)]
        ids_cut_tmp = ids[n][current_bool]
        times_cut_tmp = times[n][current_bool]

        # Append to list
        ids_cut.append(ids_cut_tmp)
        times_cut.append(times_cut_tmp)

    return ids_cut, times_cut


def set_information():
    info_for_analysis = list()
    layers = input('How many layers in the system? \n')
    LAYERS = int(layers)
    info_for_analysis.append(LAYERS)

    num_exci = input('How many excitatory neurons in each layer? \n ')
    info_for_analysis.append(int(num_exci))
    num_inhi = input('How many inhibitory neurons in each layer? \n')
    info_for_analysis.append(int(num_inhi))
    num_stimuli = input('How Stimuli? \n')
    info_for_analysis.append(int(num_stimuli))
    return info_for_analysis


def set_layer_size(info_for_analysis, neuron_select_cmd):
    neuron_start = list()
    neuron_end = list()

    num_layers = info_for_analysis[0]
    num_exci = info_for_analysis[1]
    num_inhi = info_for_analysis[2]
    num_stimuli = info_for_analysis[3]

    sizeoflayer = int(num_exci) ** 2 + int(num_inhi) ** 2

    # all_neurons_cmd = input("Differentiation between exci and inhi? (y, n) \n")
    # if (all_neurons_cmd == 'n'):
    # 	neuron_select_cmd = 0
    # elif (all_neurons_cmd == 'y'):
    # 	only_exci_cmd = input('Exi (1) or Ini (2)? \n')
    # 	if (only_exci_cmd == '1'):
    # 		neuron_select_cmd = 1
    # 	elif (only_exci_cmd == '2'):
    # 		neuron_select_cmd = 2


    if (neuron_select_cmd == 0):
        for layer in range(num_layers):
            neuronstart_tmp = layer * (sizeoflayer)
            neuronend_tmp = (layer * (sizeoflayer) + 64 ** 2 + 32 ** 2)
            neuron_start.append(neuronstart_tmp)
            neuron_end.append(neuronend_tmp)
    elif (neuron_select_cmd == 1):
        for layer in range(num_layers):
            neuronstart_tmp = layer * (sizeoflayer)
            neuronend_tmp = (layer * (sizeoflayer) + 64 ** 2)
            neuron_start.append(neuronstart_tmp)
            neuron_end.append(neuronend_tmp)
    elif (neuron_select_cmd == 2):
        for layer in range(num_layers):
            neuronstart_tmp = (layer * (sizeoflayer) + 64 ** 2)
            neuronend_tmp = (layer * (sizeoflayer) + 64 ** 2 + 32 ** 2)
            neuron_start.append(neuronstart_tmp)
            neuron_end.append(neuronend_tmp)

    return neuron_start, neuron_end, num_layers, num_stimuli


# @param: ids_cut = these are the ids that you want to analyse. They can be cut with the function cut_to_time_periode. One can also just pass the entire ids. Takes longer does the same thing.
# @param: times_cut = same thing with the ids but in this case with time
# @param: time_start = at which starting point should the rates be calculated (this does not nessesarly have to be the same time cut in the function. It does make sense to use the same though. Saves time)
# @param: time_end = until which ending pint should the rates be calculated (same stuff applies as seen in time_start)

def get_rates_all_layers(ids_cut, times_cut, time_start, time_end, neuron_start, neuron_end, num_layers, num_stimuli):
    rates = list()
    # num_stimuli = 16
    for layer in range(num_layers):
        print("Start Layer", layer + 1)
        rates_tmp = Parallel(n_jobs=num_stimuli)(delayed(data.spikesToFR)
                                                 ([ids_cut[stimulus]], [times_cut[stimulus]], neuron_start[layer],
                                                  neuron_end[layer], time_start, time_end, False)
                                                 for stimulus in range(num_stimuli))
        #         # Calculating rates between 0.5s and 1s of the stimulus time. Full_stimulus == False so that it does not automate
        #         num_stimuli = len(ids_cut)
        #         for stimulus in range(num_stimuli):
        #             rates_tmp = data.spikesToFR([ids_cut[stimulus]], [times_cut[stimulus]], neuron_start[layer], neuron_end[layer], time_start, time_end, False)
        #         # Append to list
        rates.append([])
        for n in range(num_stimuli):
            rates[-1].append(rates_tmp[n][0])
        # rates.append(rates_tmp)
    return rates


def adjust_rates(rates):
    rates_mean_adjusted = list()

    for layer in range(len(rates)):
        current_rates_input = rates[layer]
        current_rates_final = list()
        for stimuli in range(len(current_rates_input)):
            current_rates_tmp = np.copy(current_rates_input[stimuli]) - np.mean(current_rates_input[stimuli])
            current_rates_final.append(current_rates_tmp)

        neuronmeans = np.mean(np.asarray(current_rates_final), axis=0)

        for stimuli in range(len(current_rates_input)):
            current_rates_final[stimuli] -= neuronmeans
        rates_mean_adjusted.append(current_rates_final)
    return rates_mean_adjusted


def calculate_rates(ids, times, info_for_analysis, neuron_select_cmd):
    # Set Start, End and Number of Layers
    neuron_start, neuron_end, num_layers, num_stimuli = set_layer_size(info_for_analysis, neuron_select_cmd)

    # cut the IDS to a given time periode
    ids_cut, times_cut = cut_to_time_periode(ids, times)

    # calculate rates fo all layers and Stimuli (rates[layer][stimuli])
    rates = get_rates_all_layers(ids_cut, times_cut, 0.5, 1.0, neuron_start, neuron_end, num_layers, num_stimuli)

    print("\n Start Adjustement")
    # adjust the rates for each neuron, within mean over all stimuli subtracted, mean of the specific neuron
    rates_adjusted = adjust_rates(rates)

    return rates, rates_adjusted


def filtered_rates(rates, std_factor=1):
    # std_factor = int(input("Factor of the STD (i.e. 1 or 0.5)?\n"))

    filtered_rates_all = list()
    for layer in range(len(rates)):
        current_layer = rates[layer]
        filtered_rates = list()
        for stimulus_index in range(len(current_layer)):
            ratemean = np.mean(current_layer[stimulus_index])
            ratestd = 0.5 * np.std(current_layer[stimulus_index])
            filtered_rates.append(np.zeros(current_layer[stimulus_index].shape))

            filtered_rates[stimulus_index][current_layer[stimulus_index] > (ratemean + ratestd)] = 1.0
            filtered_rates[stimulus_index][(current_layer[stimulus_index] < (ratemean - ratestd))] = -1.0
        filtered_rates_all.append(filtered_rates)
    return filtered_rates_all

# # Extract rates for the first layers
# sizeoflayer = 64*64 + 32*32
# neuronstart = 3*(sizeoflayer)
# neuronend = 3*(sizeoflayer) + 64*64


# # Calculating rates between 0.5s and 1s of the stimulus time. Full_stimulus == False so that it does not automate
# rates = data.spikesToFR(ids, times, neuronstart, neuronend, 0.5, 1.0, False)
# plt.figure(figsize=(30,30))
# for stimuli in range(len(times)):
#     plt.subplot(4 , 4, stimuli+1)
#     plt.title("Stimuli " + str(stimuli+1), fontsize=14, fontweight='bold')
#     plt.suptitle("Activation for each stimulus", fontsize = 20, fontweight ='bold')
#     plt.scatter(times[stimuli], ids[stimuli], marker='.', s=1)
