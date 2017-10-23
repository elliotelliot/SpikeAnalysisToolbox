import numpy as np

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


def firing_rates_to_single_cell_information(firing_rates, objects, n_bins):
   exc_rates, inh_rates = helper.stimulus_layer_nested_list_2_numpy_tensor(firing_rates)

   exc_table = combine.response_freq_table(exc_rates, objects, n_bins=n_bins)
   inh_table = combine.response_freq_table(inh_rates, objects, n_bins=n_bins)

   exc_info = single_cell_information(exc_table)
   inh_info = single_cell_information(inh_table)

   return exc_info, inh_info
