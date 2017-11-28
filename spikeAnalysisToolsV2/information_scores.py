import numpy as np
from numba import jit
from multiprocessing import Pool
import scipy.stats as scistats
import copy

from . import helper
from . import combine_stimuli as combine



def min_response_to_one_transform(firing_rates, objects):
   """
   Find neurons that have a firing rate over the average firing rate for EVERY transform of the object.
   The neurons get the score of their MINIMAL response to one of the transforms

   :param firing_rates: nested list of shape [stimulus][layer][exc/inh] -> pandas dataframe with fields "ids", "firing_rate"
   :param objects: list containing a list of stimulus_ids that belong to one object
   :return: exh_min_objects, inh_min_objects the minimal response of a neuron to 'the minimally responsive transform of the object'
   shape [objectID, layer, neuronID]
   """
   exc_rates, inh_rates = helper.nested_list_of_stimuli_2_np(firing_rates)

   z_exh = helper.z_transform(exc_rates)
   z_inh = helper.z_transform(inh_rates)

   exh_min_objects = combine.min_responses(z_exh, objects)
   inh_min_objects = combine.min_responses(z_inh, objects)

   return exh_min_objects, inh_min_objects

def t_test_p_value(firing_rates, objects):
    """
    Compute the t-test for the response distribution of object 0 to be different from the one of object 1


    Interpratation: Probability of drawing these two value sets from the same distribution is p. We return 1-p. So the probability of NOT getting these response samples if they are drawn from the same distribution

    :param firing_rates: nested list of shape [stimulus][layer][exc/inh] -> pandas dataframe with fields "ids", "firing_rate"
    :param objects:
    :return: 1-p (p being the p value of getting these samples under the assumption of there not beeing a difference)
    """

    if len(objects) != 2:
        raise ValueError("Can only work for two objects.")

    exc_rates, inh_rates = helper.nested_list_of_stimuli_2_np(firing_rates)

    result = tuple()

    for rates in [exc_rates, inh_rates]:
        object0 = rates[objects[0], :, :]
        object1 = rates[objects[1], :, :]

        _t, p = scistats.ttest_ind(object0, object1, nan_policy='raise')



        #p[np.isnan(p)] = 1.0 # I assume it gives NaN when the variance is 0

        mean0 = np.mean(object0, axis=0)
        mean1 = np.mean(object1, axis=0)
        same_mean = np.isclose(mean0, mean1)

        # assert(np.all(np.isnan(p) == same_mean))

        p[same_mean] = 1.0

        assert(not np.any(np.isnan(p)))

        one_minus_p = np.expand_dims(1-p, 0) # to make it conistent with the scores that have one value for each object


        result += (one_minus_p,)

    return result



def average_higher_z_response_to_object(firing_rates, objects, min_difference=0.0):
    """
    A value for each neuron iff that neuron has a higher average response to a presentation of object object_ID.
    The value is the factor of standardiviations (along the responses of the neuron to different stimuli presentations)
    by which the neurons average response is higher.

   :param firing_rates: nested list of shape [stimulus][layer][exc/inh] -> pandas dataframe with fields "ids", "firing_rate"
   :param objects: list containing a list of stimulus_ids that belong to one object
   :param min_difference: the minimal number of std's that the neuron must have a hihger firing rate by in order to not get a score of 0
   :return: exh, inh: number of std by which the neuron response higher to object objectID
   shape [objectID, layer, neuronID]
    """
    exc_rates, inh_rates = helper.nested_list_of_stimuli_2_np(firing_rates)

    z_exh = helper.z_transform(exc_rates)
    z_inh = helper.z_transform(inh_rates)

    exh_avg = combine.average_responses(z_exh, objects)
    inh_avg = combine.average_responses(z_inh, objects)
    # [objects, layer, neuron_ID] -> average (over_stimuli within object) z_transformed response of neuron for the OBJECT
    # we only give it a positive value if it has an above average (over all stimuli) response
    # assert(np.all(np.isclose(np.sum(exh_avg, axis=0), 0)))

    exh_avg[exh_avg < min_difference] = 0
    inh_avg[inh_avg < min_difference] = 0

    return exh_avg, inh_avg




def mutual_information(freq_table):
    """
    Calculate mutual information between response and stimulus for each neuron. Assumes a flat prior for the stimuli

    :param freq_table: numpy array of shape [object, layer, neuron_id, response_id]-> given the object, the probability of the response_id (in that layer and neuron)
    :return:
    """
    n_objects, n_layer, n_neurons, n_response_types = freq_table.shape
    if(n_objects != 2):
        raise RuntimeWarning("Mutual information gets problamatic for more then two objects because a single neuron can't reasonably distinguish more then 2. ")

    p_response = np.mean(freq_table, axis=0) #assumes a flat prior of the objects,
    p_stimulus = np.tile((1/n_objects), n_objects)

    p_response_and_stimulus = freq_table * (1/n_objects) # assuming flat prior

    p_response_times_p_stimulus =  np.tile(p_response, (n_objects, 1, 1, 1)) * (1/n_objects) # assuming flat prior

    log = np.log2(p_response_and_stimulus/p_response_times_p_stimulus)

    all = p_response_and_stimulus * log

    all[p_response_and_stimulus==0] = 0

    summed_accros_responses = np.sum(all, axis=3)
    summed_accros_objects = np.sum(summed_accros_responses, axis=0)

    return np.expand_dims(summed_accros_objects, 0) # expand dims to make it consistent with the scores, that give one value for each object.

def firing_rates_to_mutual_information(firing_rates, objects, n_bins, calc_inhibitory=False):
    """
    Mutual Information
    :param firing_rates: nested list of shape [stimulus][layer][exc/inh] -> pandas dataframe with fields "ids", "firing_rate"
    :param objects:
    :param n_bins: how many bins the firing rates are sorted into (to make the firing rates discrete)
    :param calc_inhibitory: Flag (to save time)
    :return:
    """
    exc_rates, inh_rates = helper.nested_list_of_stimuli_2_np(firing_rates)

    exc_table = combine.response_freq_table(exc_rates, objects, n_bins=n_bins)

    exc_info = mutual_information(exc_table)

    if calc_inhibitory:
        inh_table = combine.response_freq_table(inh_rates, objects, n_bins=n_bins)
        inh_info = mutual_information(inh_table)
    else:
        inh_info = None

    return exc_info, inh_info

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
   """
   Single Cell information
   :param firing_rates: nested list of shape [stimulus][layer][exc/inh] -> pandas dataframe with fields "ids", "firing_rate"
   :param objects:
   :param n_bins: how many bins the firing rates are sorted into (to make the firing rates discrete)
   :param calc_inhibitory: Flag (to save time)
   :return:
   """
   exc_rates, inh_rates = helper.nested_list_of_stimuli_2_np(firing_rates)
   exc_table = combine.response_freq_table(exc_rates, objects, n_bins=n_bins)

   exc_info = single_cell_information(exc_table)

   if calc_inhibitory:
       inh_table = combine.response_freq_table(inh_rates, objects, n_bins=n_bins)
       inh_info = single_cell_information(inh_table)
   else:
       inh_info = None

   return exc_info, inh_info

def information_all_epochs(firing_rates_all_epochs, strategy, *args, **kwargs):
    """
    :param firing_rates_all_epochs: nested list of shape [epoch][stimulus][layer][excitatory/inhibitory] -> pandas dataframe with "ids" and "firing_rates"
    :return: exc_info, inh_info
    each is a numpy array of shape [epoch, object, layer, neuronid] -> information value
    """
    if len(args) != 0:
        raise ValueError("All extra arguments have to be specified as key word arguments")


    if strategy == "single_cell_info":
        fun = firing_rates_to_single_cell_information
    elif strategy == "average_higher_z":
        fun = average_higher_z_response_to_object
    elif strategy == "t_test_p_value":
        fun = t_test_p_value
    elif strategy == "mutual_info":
        fun = firing_rates_to_mutual_information
    else:
        raise ValueError("There is no strategy with the name {}".format(strategy))

    print("Choosen Strategy: {}, || {}".format(fun.__name__, fun.__doc__))

    caller = Caller(fun, **kwargs)
    return _multiprocess_apply_along_all_epochs(firing_rates_all_epochs, caller)



def single_cell_information_all_epochs(firing_rates_all_epochs, objects, n_bins, calc_inhibitory=False):
    """
    :param firing_rates_all_epochs: nested list of shape [epoch][stimulus][layer][excitatory/inhibitory] -> pandas dataframe with "ids" and "firing_rates"
    :return: exc_info, inh_info
    each is a numpy array of shape [epoch, object, layer, neuronid] -> information value
    """
    # caller = Caller(firing_rates_to_single_cell_information, objects, n_bins, calc_inhibitory)
    # return _multiprocess_apply_along_all_epochs(firing_rates_all_epochs, caller)
    return information_all_epochs(firing_rates_all_epochs, "single_cell_info", objects=objects, n_bins=n_bins, calc_inhibitory=calc_inhibitory)

def average_higher_z_response_all_epochs(firing_rates_all_epochs, objects, min_diff=0.0):
    """
    :param firing_rates_all_epochs: nested list of shape [epoch][stimulus][layer][excitatory/inhibitory] -> pandas dataframe with "ids" and "firing_rates"
    :return: exc_info, inh_info
    each is a numpy array of shape [epoch, object, layer, neuronid] -> information value
    """
    # caller = Caller(average_higher_z_response_to_object, objects, min_diff)
    # return _multiprocess_apply_along_all_epochs(firing_rates_all_epochs, caller)
    return information_all_epochs(firing_rates_all_epochs, "average_higher_z", objects=objects, min_difference=min_diff)


def _multiprocess_apply_along_all_epochs(firing_rates_all_epochs, caller):
    #multiprocessing implementation
    old_settings = np.seterr(all='ignore')
    worker_pool = Pool(processes=5) # global worker pool


    if False:
        exc_inh_info = map(caller, firing_rates_all_epochs)
        raise RuntimeError("The mapping worked so we should go here in debugging")

    exc_inh_info = worker_pool.map(caller, firing_rates_all_epochs)

    worker_pool.close()
    worker_pool.join()

    exc_info_fast, inh_info_fast = zip(*exc_inh_info)

    exc_np_fast = np.stack(exc_info_fast, axis=0)
    if not inh_info_fast[0] is None:
        inh_np_fast = np.stack(inh_info_fast)
    else:
        inh_np_fast = None

    np.seterr(**old_settings)
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


# TODO refactor Caller, such that it first makes the numpy tensor, then applyies some normalisation then calls the information measures for exc (and maybe inhibitory)

class Caller(object):
    def __init__(self, function, **kwargs):
        """
        when you call an instance of this object with obj(input) it will call function(input, *args)

        :param function:  the function that should be called
        :param args:  oter params that the function takes
        """
        self.function = function
        self.kwargs = copy.deepcopy(kwargs)
    def __call__(self, input):
        return self.function(input, **self.kwargs)