import csv
import os

import numpy as np
import pandas as pd

from . import helper

"""
Function to extract spikes from a binary or text file

Args:
    pathtofolder: String, Indicates path to output folder containing SpikeIDs/Times files
    binaryfile: Boolean Flag, Indicates if the output to expect is .bin or .txt (True/False)

Returns:
    pandas data frame with columns "ids" and "times" for the neuron id and spike time
"""
def pandas_load_spikes(pathtofolder, binaryfile=True, input_neurons=False):
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
    :param epoch_indices: indices of the epochs to load. (Will look for folders ["epoch_folder/epoch{}".format(e) for e in epoch_indices], alternativly it can be the string names of the subfolders directly
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

    if type(epoch_indices[0]) == int:
        print("Epochs given as indices")
        all_epoch_paths = [basic_path + epoch_folder + "/epoch{}".format(e) for e in epoch_indices]
    elif type(epoch_indices[0]) == str:
        print("Epochs given by subfodler names")
        all_epoch_paths = [basic_path + "/" + epoch_name for epoch_name in epoch_indices]
    else:
        raise ValueError("epoch_indices has to be a list of either strings or ints")


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
    :return: dictionary with fields 'elements', 'count', 'indices'
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


def random_label_from_testing_list(experiment_folder, n_objects):
    """
    Assumes that in the experiment folder there is a file called testing_list.txt. In this one the stimuli are presented in order multiple times
    i.e. all stimuli in fixed order, line with star, again all stimuli in the same order, line with star, etc.
    These stimuli will then be assigned random objects. (n_objects many of them)
    ,
    :param n_objects: how many objects we want
    :param experiment_folder: folder with the file testing_list.txt
    :return: list of lists, [object][stimulus] -> index of the stimulus for the object
    """

    stim_ids = load_testing_stimuli_ids(experiment_folder)
    stim_names = list(stim_ids.keys())

    assert((len(stim_names)%n_objects) == 0)

    n_stim_per_obj = len(stim_names) // n_objects

    import random
    random.shuffle(stim_names)

    collector_objects = list()

    for obj in range(n_objects):
        start_id = obj * (n_stim_per_obj)
        end_id = start_id + n_stim_per_obj

        collector_stimuli_in_objects = list()

        for stim_name in stim_names[start_id:end_id]:
            collector_stimuli_in_objects.extend(stim_ids[stim_name])

        collector_objects.append(collector_stimuli_in_objects)

    return collector_objects

def _deprecated_random_label_from_testing_list(experiment_folder, n_objects):
    raise DeprecationWarning("use random_label_from_testing_list")

    repeats = load_testing_stimuli_info(experiment_folder)
    # in the original folder there is a star after each presentation of all objects. each entry in this list is a 'virtual'
    # object containing all stimuli

    n_repeats = len(repeats)
    n_stimuli = repeats[0]['count']
    stimuli = repeats[0]['elements']
    stimulus_indices = np.array(repeats[0]['indices'])

    for i, repeat in enumerate(repeats):
        assert(stimuli == repeat['elements'])
        assert(np.all(stimulus_indices + (i * n_stimuli) == np.array(repeat['indices'])))# this is pretty useless

    assert((n_stimuli % n_objects) == 0)

    n_transforms = n_stimuli // n_objects

    return helper.random_label(n_objects=n_objects, n_transforms=n_transforms, n_repeats=n_repeats)


def load_testing_stimuli_names(experiment_folder):
    cur_obj = 0
    collector = list()
    with open(experiment_folder + "/testing_list.txt", "r") as file:
        for line in file:
            raw_text = line.strip()
            if raw_text != "*":
                collector.append("{}".format(raw_text))
            else:
                cur_obj += 1
    return collector

def load_testing_stimuli_ids(*args, **kwargs):
    import warnings
    warnings.warn("Use the function load_testing_stimuli_dict instead of this one", DeprecationWarning)
    return load_testing_stimuli_dict(*args, **kwargs)

def load_testing_stimuli_dict(experiment_folder):
    """
    :param experiment_folder:
    :return: dict with stimulus name as key and all the indices of this stimulus as list of ints under that key
    """
    cur_stim_idx = 0
    collector = dict()
    with open(experiment_folder + "/testing_list.txt", "r") as file:
        for line in file:
            raw_text = line.strip()
            if raw_text != "*":
                indices_so_far = collector.get(raw_text, list())
                indices_so_far.append(cur_stim_idx)
                collector[raw_text] = indices_so_far
                cur_stim_idx += 1
    return collector



def load_testing_stimuli_indices_from_wildcarts(experiment_folder, objects):
    """
    Load indices of objects, objects are specified with wildcarts. e.g. 1*cl is the object containing all stimuli
    with loc=1, type=circle, position=l but arbitrary color.

    e.g. 1**l would be a right border at location one neuron

    :param experiment_folder: path to the folder containting testing_list.txt
    :param objects: list of strings of type 1wcl (loc, color, type, pos)
    :return: list of dictionaries, first item is the dictionary containing in it's filed 'indices' indices of all stimuli that fullfill the specifications of the first string in objects
    """
    result = [dict(filter=filter_string, elements=set(), indices=list(), count=0) for filter_string in objects]

    #### HELPER CLASS
    class Object_Filter:
        def __init__(self, filter_string):
            """filter_string for example 1*wcl"""
            self.constraints = [(i, value) for i, value in enumerate(filter_string) if value != "*"]

        def __call__(self, stimulus_name):
            """check if the stimulus fullfills the requirements"""
            for ind, val in self.constraints:
                if stimulus_name[ind] != val:
                    return False # one constraint was violated
            return True # went through all the constraints without a problem.
    ### END HELPER CLASS

    filter = [Object_Filter(filter_string) for filter_string in objects]
    cur_stim_id = 0

    with open(experiment_folder + "/testing_list.txt", "r") as file:
        for line in file:
            raw_text = line.strip()
            n_obj_this_stimulus_is_part_of = 0

            if raw_text != "*":
                for fil_id, fil in enumerate(filter):
                    if fil(raw_text): #stimulus part of object fil_id
                        result[fil_id]['indices'].append(cur_stim_id)
                        result[fil_id]['count'] += 1
                        result[fil_id]['elements'].add(raw_text)
                        n_obj_this_stimulus_is_part_of += 1

                # check if the stimulus had exactly one home
                if n_obj_this_stimulus_is_part_of != 1:
                    raise ValueError("The stimulus {} was part of {} objects".format(raw_text,n_obj_this_stimulus_is_part_of))

                cur_stim_id += 1


    return result


def load_testing_stimuli_label_matrix(experiment_folder, objects, multi_object_delimiter="_"):
    """
    objects are specified with wildcarts. e.g. 1*cl is the object containing all stimuli
    with loc=1, type=circle, position=l but arbitrary color.

    If there are multiple objects according to the naming convention 1bcl_2bcl it is enough if one of them satisfies the wildcard

    e.g. 1**l would be a right border at location one neuron

    :param experiment_folder: path to the folder containting testing_list.txt
    :param objects: list of strings of type 1wcl (loc, color, type, pos)
    :param multi_object_delimiter: char (if there are multiple objects in a stimulus the filename should contain <attributes one stim><multi_object_delimiter><attributes other stim> e.g. 1wcl_2bcl
    :return: numpy array of shape [n_objects, n_stimuli] -> true if the stimulus contains an object specified by the corresponding wildcard
    """
    n_objects = len(objects)

    #### HELPER CLASS
    class Object_Filter:
        def __init__(self, filter_string):
            """filter_string for example 1*wcl"""
            self.constraints = [(i, value) for i, value in enumerate(filter_string) if value != "*"]

        def check_object_string(self, object_string):
            """check if an object_string (e.g. 1b*l fullfills the requirements"""
            for ind, val in self.constraints:
                if object_string[ind] != val:
                    return False # one constraint was violated
            return True # went through all the constraints without a problem.

        def __call__(self, stimulus_name):
            for s in stimulus_name.split(multi_object_delimiter):
                if self.check_object_string(s):
                    return True # one is enough
            return False # none of the present objects satified the wildcard
    ### END HELPER CLASS

    all_filter = [Object_Filter(filter_string) for filter_string in objects]

    collector = list()

    with open(experiment_folder + "/testing_list.txt", "r") as file:
        for line in file:
            raw_text = line.strip()


            if raw_text != "*":
                current_stimulus = np.zeros((n_objects, 1), dtype=bool)
                for fil_id, filter in enumerate(all_filter):
                    if filter(raw_text): #stimulus part of object fil_id
                        current_stimulus[fil_id, 0] = True
                collector.append(current_stimulus)

    return np.concatenate(collector, axis=1)






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
        print("loading from {}".format(filepath))
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

def get_epochs(experiment_folder, subfolder="testing"):
    """
    Load list of all epoch folders for an experiment
    :param experiment_folder: path to the folder containing the experiment
    :param subfolder: in which to look for the epochs
    :return: list of strings that give path name like experiment_folder + "/" subfolder
    """
    if type(experiment_folder) == list:
        experiment_folder =  experiment_folder[0]

    import os
    collector = list()
    for potential_exp in os.listdir(experiment_folder+"/"+subfolder):
        if potential_exp[:5] == "epoch":
            collector.append(int(potential_exp[5:]))

    collector.sort()
    return ["{}/epoch{}".format(subfolder, e) for e in collector]


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


def load_filter_all_obj(path_to_filter_dir, output="pandas"):
    """
    load all filtered values for all objects into a dictonary
    :param path_to_filter_dir: path to folder containing all '.flt' dirs
    :return: dictory with 'object_name': pandas_dataframe with 'ids' and 'filter_values'
    """
    load_filter_func = None
    if output == "pandas":
        load_filter_func = load_filter_values_global_data_frame
    elif output == "list":
        load_filter_func = load_filter_values_list


    collector = dict()
    for filter in os.listdir(path_to_filter_dir):
        if filter[-4:] == ".flt":
            collector[filter[:-4]] = load_filter_func(path_to_filter_dir + "/" + filter)
    return collector


