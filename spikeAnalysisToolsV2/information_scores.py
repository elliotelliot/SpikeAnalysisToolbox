import numpy as np
from numba import jit
from multiprocessing import Pool

from . import helper
from . import combine_stimuli as combine

def min_response_to_one_transform(firing_rates, objects):
   """
   Find neurons that have a firing rate over the average firing rate for EVERY transform of the object.
   The neurons get the score of their MINIMAL response to one of the transforms

   :param firing_rates: pandas data frame with columns "ids" and "times"
   :param objects: list containing a list of stimulus_ids that belong to one object
   :return: exh_min_objects, inh_min_objects the minimal response of a neuron to 'the minimally responsive transform of the object'
   shape [objectID, layer, neuronID]
   """
   exc_rates, inh_rates = helper.stimulus_layer_nested_list_2_numpy_tensor(firing_rates)

   z_exh = helper.z_transform(exc_rates)
   z_inh = helper.z_transform(inh_rates)

   exh_min_objects = combine.min_responses(z_exh, objects)
   inh_min_objects = combine.min_responses(z_inh, objects)

   return exh_min_objects, inh_min_objects


@jit(cache=True)
def single_cell_information(freq_table):
   """
   Calculate single cell information according to Stringer 2005

   :param freq_table:  numpy array of shape [object, layer, neuron_id, response_id]
   :return:
   """
   n_objects, n_layer, n_neurons, n_response_types = freq_table.shape


   p_response = np.mean(freq_table, axis=0) #assumes a flat prior of the objects,
   # so the p(r) = p(r|s) * p(s)
   # p(s) is the same so 1/N * sum p(r|s)

   fraction = freq_table / p_response


   log_fraction = np.log2(fraction)

   log_fraction[freq_table == 0] = 0 # a response that never happend will become zero (in entropy 0 * log2(0) = 0 by definition

   before_sum = freq_table * log_fraction

   information = np.sum(before_sum, axis=3) # sum along the response axis

   return information


# @jit(cache=True)
def firing_rates_to_single_cell_information(firing_rates, objects, n_bins, calc_inhibitory=False):
   exc_rates, inh_rates = helper.stimulus_layer_nested_list_2_numpy_tensor(firing_rates)

   exc_table = combine.response_freq_table(exc_rates, objects, n_bins=n_bins)

   exc_info = single_cell_information(exc_table)

   if calc_inhibitory:
       inh_table = combine.response_freq_table(inh_rates, objects, n_bins=n_bins)
       inh_info = single_cell_information(inh_table)
   else:
       inh_info = None

   return exc_info, inh_info

def information_all_epochs(firing_rates_all_epochs, objects, n_bins, calc_inhibitory=False):
    """
    Converts a nested list of firing rates to 2 numpy arrays
    :param firing_rates_all_epochs: nested list of shape [epoch][stimulus][layer][excitatory/inhibitory] -> pandas dataframe with "ids" and "firing_rates"
    :return: exc_info, inh_info
    each is a numpy array of shape [epoch, object, layer, neuronid] -> information value
    """

    #multiprocessing implementation
    worker_pool = Pool(processes=5)

    tmp_caller = Caller(firing_rates_to_single_cell_information, objects, n_bins, calc_inhibitory)


    exc_inh_info = worker_pool.map(tmp_caller, firing_rates_all_epochs)
    # this is a list of tuple [(exc_info_epoch1, inh_info_epoch1), (exc_info_epoch2, inh_info_epoch2),...]

    exc_info_fast, inh_info_fast = zip(*exc_inh_info)

    # exc_list = []
    # inh_list = []
    #
    # print("Epoch: >>  ", end = "")
    # for i, epoch in enumerate(firing_rates_all_epochs):
    #     exc, inh = firing_rates_to_single_cell_information(epoch, objects, n_bins, calc_inhibitory)
    #     exc_list.append(exc)
    #     inh_list.append(inh)
    #     print(i, end = "  ")
    #
    # exc_np = np.stack(exc_list, axis=0)
    exc_np_fast = np.stack(exc_info_fast, axis=0)
    if calc_inhibitory:
        # inh_np = np.stack(inh_list, axis=0)
        inh_np_fast = np.stack(inh_info_fast)
    else:
        inh_np_fast = None

    # assert(np.all(exc_np == exc_np_fast))

    return exc_np_fast, inh_np_fast


def slow_information_all_epochs(firing_rates_all_epochs, objects, n_bins, calc_inhibitory=False):
    """
    Converts a nested list of firing rates to 2 numpy arrays
    :param firing_rates_all_epochs: nested list of shape [epoch][stimulus][layer][excitatory/inhibitory] -> pandas dataframe with "ids" and "firing_rates"
    :return: exc_info, inh_info
    each is a numpy array of shape [epoch, object, layer, neuronid] -> information value
    """

    exc_list = []
    inh_list = []

    print("Epoch: >>  ", end = "")
    for i, epoch in enumerate(firing_rates_all_epochs):
        exc, inh = firing_rates_to_single_cell_information(epoch, objects, n_bins, calc_inhibitory)
        exc_list.append(exc)
        inh_list.append(inh)
        print(i, end = "  ")

    exc_np = np.stack(exc_list, axis=0)
    if calc_inhibitory:
        inh_np = np.stack(inh_list, axis=0)
    else:
        inh_np = None

    # assert(np.all(exc_np == exc_np_fast))

    return exc_np, inh_np

class Caller(object):
    def __init__(self, function, *args):
        """
        when you call an instance of this object with obj(input) it will call function(input, *args)

        :param function:  the function that should be called
        :param args:  oter params that the function takes
        """
        self.function = function
        self.args = args[:]
    def __call__(self, input):
        return self.function(input, *self.args)