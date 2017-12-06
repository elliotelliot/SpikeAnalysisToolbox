import pandas as pd
import numpy as np
import csv
from multiprocessing import Pool
from functools import partial
import os
from . import helper



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




"""
Function to extract the pre, post, weight and delays of a network structure

Args:
    pathtofolder: string path to the folder in which network files are stored
    binaryfile: True/False flag if it is binary file
    intial_weighs: True/False flag wether to load initial weights

Returns:
    Pandas data frame with the following colums:
    pre: list of synapse presynaptic indices
    post: list of synapse postsynaptic indices
    delays: list of synaptic delays (in units of timesteps)
    init_weights: list of synaptic weights (before training) only if initial_weights=True
    weights: list of synaptic weights after training
"""
def load_network(pathtofolder, binaryfile=True, initial_weights=False):
    if pathtofolder[-1] != "/":
        pathtofolder += "/"

    pre, post, delays, init_weights , weights = _raw_load_network(pathtofolder, binaryfile, initial_weights)
    data = dict(pre=pre, post=post, delays=delays, weights=weights)
    if initial_weights:
        data['init_weights'] = init_weights

    return pd.DataFrame(data=data)


def _raw_load_network(pathtofolder, binaryfile, initial_weights):
    pre = list()
    post = list()
    delays = list()
    if initial_weights:
        init_weights = list()
    else:
        init_weights = None
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
        if initial_weights:
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
        if initial_weights:
            with open(pathtofolder + 'Synapses_NetworkWeights_Initial.txt', 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in reader:
                    init_weights.append(float(row[0]))

        with open(pathtofolder + 'Synapses_NetworkWeights.txt', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                weights.append(float(row[0]))

    return (pre, post, delays, init_weights, weights)

def load_only_weights(pathtofolder, binaryfile=True):
    if pathtofolder[-1] != "/":
        pathtofolder += "/"

    weights = np.fromfile(pathtofolder +
                          'Synapses_NetworkWeights' + '.bin',
                          dtype=np.float32)
    return weights



def load_weights_all_epochs(basic_path, epoch_indices, epoch_folder="testing", initial_folder_name="initial", binary=True, initial_weights=False):
    """
    Load weights for all epochs
    :param basic_path: the top level folder containing the experiments result
    :param epoch_indices: indices of the epochs to load. (Will look for folders ["epoch_folder/epoch{}".format(e) for e in epoch_indices]
    :param epoch_folder: the name of the folder containing the folders "epochX". Whith each epoch folder containing a weights file.
    :param initial_folder_name: name for the initial run (before training). The Network architecture will be loaded from here
    :param binary: is it a binary file
    :param initial_weights: (Deprecated)
    :return: (full_network_initial, weights_all_epochs)
        full_network_inital: pandas dataframe with columns: pre, post, delays, weights (of the inital state)
        weights_all_epochs:  numpy array of shape [epoch, synapse] -> weight of that synapse at that epoch (epoch 0 is same weights as full_network_initial
    """
    if basic_path[-1] != "/":
        basic_path += "/"

    all_epoch_paths = [basic_path + epoch_folder + "/epoch{}".format(e) for e in epoch_indices]

    full_network_initial = load_network(basic_path + initial_folder_name, binaryfile=binary, initial_weights=initial_weights)

    weights_all_epochs = list(map(load_only_weights, all_epoch_paths))

    weight_matrix = np.stack([full_network_initial.weights.values] + weights_all_epochs, axis=0)

    return full_network_initial, weight_matrix


def load_stimulus_file_to_attribute_matrix(filename, attributes):
    """
    reads stimulus names from file and translates them to a matrix of attribute values

    :param filename: name of file
    :param attributes: list of dicts -> pos_in_stim, and an entry for each possible value of that letter in the stimulus names. the first dictonary in this list will be in the first place in the resulting matrix
    :return:
    """
    import numpy as np
    stimuli_collector = list()

    with open(filename, "r") as file:
        for line in file:
            raw_text = line.strip()
            if raw_text != "*":
                assert(len(raw_text)==len(attributes))

                this_stimulus = -1 * np.ones((1, len(attributes)))

                for atr_id, atr in enumerate(attributes):
                    letter = raw_text[atr["pos_in_stim"]]
                    atr_value = atr[letter]

                    this_stimulus[0, atr_id] = atr_value

                stimuli_collector.append(this_stimulus)

    matrix = np.concatenate(stimuli_collector, axis=0)
    assert(np.all(matrix != -1))

    return matrix




def load_testing_stimuli_info(experiment_folder):
    """
    load the information about the stimuli presented during testing. There is assumed to be a file "testing_list.txt" with the information in the experiment_folder
    :param experiment_folder: top level folder of the experiment
    :return:
    """
    objects = []
    current_object = {'count': 0, 'elements': set(), 'indices': list()}
    current_stim_index = 0
    with open(experiment_folder + "/testing_list.txt", "r") as file:
        for line in file:
            raw_text = line.strip()
            if raw_text == "*":
                # make new object
                objects.append(current_object)
                current_object = {'count': 0, 'elements': set(), 'indices': list()}
            else:
                current_object['elements'].add(raw_text)
                current_object['count'] += 1
                current_object['indices'] += [current_stim_index]
                current_stim_index += 1
    objects.append(current_object)

    proper_objects = [obj for obj in objects if obj['count'] != 0]
    return proper_objects

def load_testing_stimuli_names(experiment_folder):
    cur_obj = 0
    collector = list()
    with open(experiment_folder + "/testing_list.txt", "r") as file:
        for line in file:
            raw_text = line.strip()
            if raw_text != "*":
                collector.append("obj{}:{}".format(cur_obj, raw_text))
            else:
                cur_obj += 1
    return collector

class FilterValues:
    """class to save filter values"""
    def __init__(self, scale, orientation, phase, values, obj_name=None):
        """
        :param scale: int with scale of this filter
        :param orientation: orientation of this object in degree
        :param phase: phase of this filter
        :param values: numpy array with filter values as read out from folder
        """
        self.scale = int(scale)
        self.orientation = int(orientation)
        self.phase = int(phase)
        self.values = self.weird_filter_reading_workaround(values)
        self.obj_name = obj_name

    def __repr__(self):
        return "<Filter: {obj_name}.{scale}.{orientation}.{phase}>".format(**self.__dict__)

    @staticmethod
    def load_from_file(filepath):
        filename = filepath.split("/")[-1]
        filter_parameters = filename.split(".")
        # theses have positions [name.scale.orientation.phase.gbo]
        #                         0     1       2          3   4
        assert(filter_parameters[4] == "gbo")
        values = np.fromfile(filepath, dtype=np.float32)

        return FilterValues(scale=filter_parameters[1], orientation=filter_parameters[2], phase=filter_parameters[3], values=values, obj_name=filter_parameters[0])

    @staticmethod
    def weird_filter_reading_workaround(values):
        """
        Because the filter values are not read consecutively in Spike::Neurons::ImagePoissonInputSpikingNeurons
         we have to use this workaround to end up with an array that resembles the one spike gets after reading

         Basically we are transposing
        """
        n_values = len(values)
        side_length = helper.get_side_length(n_values)

        # Spike reads the values as if they were written in F order
        in2d = np.reshape(values, (side_length, side_length), order='F')

        # it then saves them into 'C' order
        return in2d.flatten(order='C')


    def get_garbor_index(self, *args, **kwargs):
        """
        returns the index of this filter
        :param scales, orientations, phases, any of these can be given as a key word argument as a list, for not given ones defaults will be used.
        :return: index of filter
        """
        assert(len(args)==0)
        all_filter_options = dict(scales = [2], orientations = [0, 45, 90, 135], phases = [0, 180])
        all_filter_options.update(kwargs)

        scale_id = all_filter_options['scales'].index(self.scale)
        orientation_id = all_filter_options['orientations'].index(self.orientation)
        phase_id = all_filter_options['phases'].index(self.phase)

        total_number_of_wavelengths = len(all_filter_options['scales'])
        total_number_of_phases = len(all_filter_options['phases'])

        garbor_id = orientation_id * (total_number_of_wavelengths * total_number_of_phases) + scale_id * total_number_of_phases + phase_id
        return garbor_id

    def get_global_id_values(self, *args, **kwargs):
        """
        returns a pandas dataframe with columns 'ids' (i.e. global id values) and 'filter_values'

        :param scales, orientations, phases, any of these can be given as a key word argument as a list, for not given ones defaults will be used.
        :return: pandas dataframe with 'ids' and 'filter_values'
        """

        gabor_id = self.get_garbor_index(*args, **kwargs)

        neurons_per_filter = self.values.shape[0]

        start_index_for_current_gabor_image = gabor_id * neurons_per_filter

        indices = self.corrected_id(np.arange(neurons_per_filter) + start_index_for_current_gabor_image)

        return pd.DataFrame({'ids': indices, 'filter_values': self.values}, index=indices)




    @staticmethod
    def corrected_id(id):
        return (-1 * (id)) - 1




def load_filter_values_list(path_to_filter):
    """
    return list of FilterValues objects for the given stimulus
    :param path_to_filter: string path to the '.flt' directory
    :return:
    """
    assert(path_to_filter[-4:] == '.flt')
    subfolders = os.listdir(path_to_filter)
    collector = list()

    for garbor_filter in subfolders:
        if garbor_filter[-4:] == '.gbo':
            #it is a filter
            new_filter = FilterValues.load_from_file(path_to_filter+"/"+garbor_filter)
            collector.append(new_filter)

    return collector

def load_filter_values_global_data_frame(path_to_filter):
    """
    load all filter values into one big pandas dataframe with columns "ids" and "filter_values"
    it will have ids corresponding to the ones in the network architecture files

    :param path_to_filter: path to '.flt' directory
    :return: pandas dataframe with columns "ids", "filter_values"
    """

    filter_list = load_filter_values_list(path_to_filter)

    pd_frames = [fil.get_global_id_values() for fil in filter_list]

    big_df = pd.concat(pd_frames)

    sorted = big_df.sort_index(ascending=False)

    assert(np.all(sorted.index == FilterValues.corrected_id(np.arange(128*128*8))))

    return sorted


def load_filter_all_obj(path_to_filter_dir):
    """
    load all filtered values for all objects into a dictonary
    :param path_to_filter_dir: path to folder containing all '.flt' dirs
    :return: dictory with 'object_name': pandas_dataframe with 'ids' and 'filter_values'
    """
    collector = dict()
    for filter in os.listdir(path_to_filter_dir):
        if filter[-4:] == ".flt":
            collector[filter[:-4]] = load_filter_values_global_data_frame(path_to_filter_dir + "/" + filter)
    return collector
