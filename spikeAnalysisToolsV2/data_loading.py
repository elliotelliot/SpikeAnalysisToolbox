import pandas as pd
import numpy as np
import csv


"""
Function to extract spikes from a binary or text file

Args:
    pathtofolder: String, Indicates path to output folder containing SpikeIDs/Times files
    binaryfile: Boolean Flag, Indicates if the output to expect is .bin or .txt (True/False)

Returns:
    pandas data frame with columns "ids" and "times" for the neuron id and spike time
"""
def pandas_load_spikes(pathtofolder, binaryfile, input_neurons=False):
    ids, times = get_spikes(pathtofolder=pathtofolder, binaryfile=binaryfile, input_neurons=input_neurons)
    return pd.DataFrame({"ids": ids, "times": times})


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
Imports the ids and times for all supfolders and stores them in a list of pandas data frames

Args:
    masterpath: The Masterpath (i.e. "/Users/dev/Documents/Gisi/01_Spiking_Simulation/01_Spiking Network/Build/output/") 
    subfolders: All of the Stimulations in and list that are supossed to be analysed (i.e.["ParameterTest_0_epochs_all/", "ParameterTest_0_epochs_8_Stimuli/"]).
                If only one is of interest use ["ParameterTest_0_epochs/"]
    extensions: All epochs that are supposed to be imported (i.e. ["initial/""] or ["initial", "testing/epoch1/", "testing/epoch2/", ..., "testing/epoch_n/"])
    input_layer: If you want to look at the input layer only set this to true. 

Returns:
    all_subfolders: all supfolder spikes. shape [subfolder][extension]-> pandas data frame with all the spikes
"""
def load_spikes_from_subfolders(masterpath, subfolders, extensions, input_layer):
    print("Start")
    all_subfolders = list()
    if input_layer:
        for subfol in subfolders:
            print(subfol)
            # print(subfolders[subfol])
            for ext in extensions:
                all_extensions = list()
                currentpath = masterpath + "/" + subfol + "/" + ext + "/"
                spikes = pandas_load_spikes(currentpath, True, True)
                all_extensions.append(spikes)

            all_subfolders.append(all_extensions)
    else:
        for subfol in subfolders:
            all_extensions = list()
            for ext in extensions:
                currentpath = masterpath + "/" + subfol + "/" + ext + "/"
                spikes = pandas_load_spikes(currentpath, True)
                all_extensions.append(spikes)

            all_subfolders.append(all_extensions)

    return all_subfolders