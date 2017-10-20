import csv

import numpy as np
import pandas as pd

import Activation_Analysis as acan

"""
Function to extract the pre, post, weight and delays of a network structure

Args:
    pathtofolder: string path to the folder in which network files are stored

Returns:
    pre: list of synapse presynaptic indices
    post: list of synapse postsynaptic indices
    delays: list of synaptic delays (in units of timesteps)
    init_weights: list of synaptic weights (before training)
    weights: list of synaptic weights after training
"""
def load_network(pathtofolder, binaryfile, inital_weights):
    pre = list()
    post = list()
    delays = list()
    init_weights = list()
    weights = list()

    if (binaryfile):
        pre = np.fromfile(pathtofolder +
                          'Synapses_NetworkPre' + '.bin',
                          dtype=np.int32)
        post = np.fromfile(pathtofolder +
                           'Synapses_NetworkPost' + '.bin',
                           dtype=np.int32)
        delays = np.fromfile(pathtofolder +
                             'Synapses_NetworkDelays' + '.bin',
                             dtype=np.int32)
        if init_weights:
            init_weights = np.fromfile(pathtofolder + 'Synapses_NetworkWeights_Initial' + '.bin',
                                       dtype=np.float32)
        weights = np.fromfile(pathtofolder +
                              'Synapses_NetworkWeights' + '.bin',
                              dtype=np.float32)

        return pre, post, delays, init_weights, weights
    else:

        # For each file type output by the network, load them
        with open(pathtofolder + 'Synapses_NetworkPre.txt', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                pre.append(int(row[0]))

        with open(pathtofolder + 'Synapses_NetworkPost.txt', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                post.append(int(row[0]))

        with open(pathtofolder + 'Synapses_NetworkDelays.txt', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                delays.append(int(row[0]))
        if init_weights:
            with open(pathtofolder + 'Synapses_NetworkWeights_Initial.txt', 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    init_weights.append(float(row[0]))

        with open(pathtofolder + 'Synapses_NetworkWeights.txt', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                weights.append(float(row[0]))

    return (pre, post, delays, init_weights, weights)


"""
Function to extract spike times and IDs from a binary or text file

Args:
    pathtofolder: String, Indicates path to output folder containing SpikeIDs/Times files
    binaryfile: Boolean Flag, Indicates if the output to expect is .bin or .txt (True/False)

Returns:
    ids: numpy array of neuron ids for each spike
    times: numpy array of spike times for each spike (corresponding to the ids
"""
def get_spikes(pathtofolder, binaryfile, input_neurons=False):
    spike_ids = list()
    spike_times = list()
    id_filename = 'Neurons_SpikeIDs_Untrained_Epoch0'
    times_filename = 'Neurons_SpikeTimes_Untrained_Epoch0'
    if (input_neurons):
        id_filename = 'Input_' + id_filename
        times_filename = 'Input_' + times_filename
    if (binaryfile):
        idfile = np.fromfile(pathtofolder +
                             id_filename + '.bin',
                             dtype=np.uint32)
        timesfile = np.fromfile(pathtofolder +
                                times_filename + '.bin',
                                dtype=np.float32)
        return idfile, timesfile
    else:
        # Getting the SpikeIDs and SpikeTimes
        idfile = open(pathtofolder +
                      id_filename + '.txt', 'r')
        timesfile = open(pathtofolder +
                         times_filename + '.txt', 'r')

        # Read IDs
        try:
            reader = csv.reader(idfile)
            for row in reader:
                spike_ids.append(int(row[0]))
        finally:
            idfile.close()
        # Read times
        try:
            reader = csv.reader(timesfile)
            for row in reader:
                spike_times.append(float(row[0]))
        finally:
            timesfile.close()
        return (np.array(spike_ids).astype(np.int),
                np.array(spike_times).astype(np.float))




"""
Converts a long spike train into separate stimuli based upon a duration

Args:
    neuronids: numpy array of spike train neuron ids
    neurontimes: numpy array of spike train times
    stimduration: float indicating the length of time by which to split the stimuli

Returns:
    ids: A list of numpy arrays, each the set of spike ids for separate stimulus
    times: A list of numpy arrays, each the set of spike times for separate stimulus
"""
def splitstimuli(neuronids, neurontimes, stimduration):
    num_stimuli = int(np.ceil(np.max(neurontimes) / stimduration))

    ids = list()
    times = list()

    for i in range(num_stimuli):
        mask = [(neurontimes > (i * stimduration)) & (neurontimes < ((i + 1) * stimduration))]
        times.append(neurontimes[mask] - (i * stimduration))
        ids.append(neuronids[mask])

    return (ids, times)





"""
 Function to convert a spike train into a set of firing rates
 
 Args:
    ids: a list of numpy arrays of ids (e.g. for all stimuli)
    times: a list of numpy arrays if times
    neuronstart: int, the ID of the first (inclusive) neuron to consider
    neuronend: int, the ID of the last (excluding) neuron to consider
    timestart: float, the time from which to begin calculating the rate
    timeend: float, the time from which to stop calculating the rate
    full_stimulus: boolean flag, if true overrides timestart/end and uses the full time available

Returns:
    rates: A list of numpy arrays. Each array of length equal to the number neurons, representing their rates. 
"""
def spikesToFR(ids, times, neuronstart, neuronend, timestart=0.0, timeend=1.0, full_stimulus=True):
    # Calculating the average firing rates (since we only present sounds for
    # 1s, just the spike count)
    vowelrates = list()
    for s in range(len(ids)):
        rates = np.zeros(neuronend - neuronstart)
        if (full_stimulus):
            timelength = np.max(times[s])
            idswindow = ids[s]
            for i in range(neuronend - neuronstart):
                rates[i] = np.sum(idswindow == (neuronstart + i)) / timelength
            vowelrates.append(rates)
        else:
            timelength = timeend - timestart
            idswindow = ids[s][(times[s] >= timestart) & (times[s] < timeend)]
            for i in range(neuronend - neuronstart):
                rates[i] = np.sum(idswindow == (neuronstart + i)) / timelength
            vowelrates.append(rates)
    return (vowelrates)














"""
Using a descriptor file for the stimuli, extracts stimulus categories

Args:
    filenamepath: String to the file containing the stimuli details
 
Returns:
    Vowel: list of the vowel categories (len=num_stimuli)
    Gender: list of gender categories (len=num_stimuli)
    Speaker: list of speaker categories (len=num_stimuli)
    F0: list of the F0 for each stimulus (len=num_stimuli)
    Duration: list of the durations of each stimulus (len=num_stimuli)
"""
def stimuliCats(filenamepath):
    # Getting the SpikeIDs and SpikeTimes
    stimulifile = open(filenamepath, 'r')
    # filepath + 'trainingset.csv'
    #
    stimuli = list()

    # Read Stimuli
    try:
        reader = csv.reader(stimulifile)
        for row in reader:
            stimuli.append(row)
    finally:
        stimulifile.close()

    ordered_vowel = list()
    ordered_gender = list()
    ordered_spkr = list()
    ordered_f0 = list()
    ordered_dur = list()
    for i in range(len(stimuli)):
        if i == 0:
            pass
        else:
            # The vowel is the index 3 position
            ordered_vowel.append(int(stimuli[i][3]))
            # Group/Gender is at index 1
            ordered_gender.append(int(stimuli[i][1]))
            # Speaker is at index 2
            ordered_spkr.append(int(stimuli[i][1]) * 100 + int(stimuli[i][2]))
            # F0 is at index 5
            ordered_f0.append(int(stimuli[i][5]))
            # Duration is at index 6
            ordered_dur.append(int(stimuli[i][6]))
    # Return
    return (
        ordered_vowel, ordered_gender, ordered_spkr, ordered_f0, ordered_dur)


"""
Imports the ids and times for all supfolders and stores them in list

Args:
    masterpath: The Masterpath (i.e. "/Users/dev/Documents/Gisi/01_Spiking_Simulation/01_Spiking Network/Build/output/") 
    subfolders: All of the Stimulations in and list that are supossed to be analysed (i.e.["ParameterTest_0_epochs_all/", "ParameterTest_0_epochs_8_Stimuli/"]).
                If only one is of interest use ["ParameterTest_0_epochs/"]
    extensions: All epochs that are supposed to be imported (i.e. ["initial/""] or ["initial", "testing/epoch1/", "testing/epoch2/", ..., "testing/epoch_n/"])
    input_layer: If you want to look at the input layer only set this to true. 
 
Returns:
    all_subfolders_ids: all supfolders ids. For first subfolder all_subfolders_ids[0]
    all_subfolders_times: all supfolders ids. For first subfolder all_subfolders_times[0]
"""
def subfolders_ids_times(masterpath, subfolders, extensions, input_layer):
    print("Start")
    all_subfolders_ids = list()
    all_subfolders_times = list()
    if input_layer:
        for subfol in subfolders:
            print(subfol)
            # print(subfolders[subfol])
            for ext in extensions:
                all_extensions_ids = list()
                all_extensions_times = list()
                currentpath = masterpath + "/" + subfol + "/" + ext + "/"
                ids, times = get_spikes(currentpath, True, True)
                all_extensions_ids.append(ids)
                all_extensions_times.append(times)

            all_subfolders_ids.append(all_extensions_ids)
            all_subfolders_times.append(all_extensions_times)
    else:
        for subfol in subfolders:
            all_extensions_ids = list()
            all_extensions_times = list()
            for ext in extensions:
                currentpath = masterpath + "/" + subfol + "/" + ext + "/"
                ids, times = get_spikes(currentpath, True)
                all_extensions_ids.append(ids)
                all_extensions_times.append(times)

            all_subfolders_ids.append(all_extensions_ids)
            all_subfolders_times.append(all_extensions_times)

    return all_subfolders_ids, all_subfolders_times





