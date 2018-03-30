import copy
from multiprocessing import Pool

import numpy as np
import scipy.stats as scistats
from numba import jit

from . import combine_stimuli as combine
from . import helper


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
    :param objects: list of list with stimulus ids for that object
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
   Calculate single cell information according to Stringer 2005 from a frequency table

   :param freq_table:  numpy array of shape [object, layer, neuron_id, response_id]
   :return: numpy array of shape [object, layer, neuron_id]
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
def firing_rates_to_single_cell_information(firing_rates, objects, n_bins=3, calc_inhibitory=False):
   """
   Single Cell information from a list of firing rates
   :param firing_rates: nested list of shape [stimulus][layer][exc/inh] -> pandas dataframe with fields "ids", "firing_rate"
   :param objects: list of list with stimulus ids for that object
   :param n_bins: how many bins the firing rates are sorted into (to make the firing rates discrete)
   :param calc_inhibitory: Flag (to save time)
   :return: exc_info, inh_info, each is a numpy array of shape [n_objects, n_layer, n_neurons]-> info that neuron for the object
   """
   exc_rates, inh_rates = helper.nested_list_of_stimuli_2_np(firing_rates)
   exc_info = firing_rates_numpy_to_single_cell_info(exc_rates, objects, n_bins)

   if calc_inhibitory:
       # inh_table = combine.response_freq_table(inh_rates, objects, n_bins=n_bins)
       # inh_info = single_cell_information(inh_table)
        inh_info = firing_rates_numpy_to_single_cell_info(inh_rates, objects, n_bins)
   else:
       inh_info = None

   return exc_info, inh_info

def firing_rates_numpy_to_single_cell_info(firing_rates, objects, n_bins=3, allow_nan_as_seperate_bin=False):
    """
    Calculate single cell info for the given firing rates
    :param firing_rates: numpy array of shape [stimuli, layer, neuron]
    :param objects: list of list with stimulus ids for that object
    :param n_bins:
    :return:
    """
    table = combine.response_freq_table(firing_rates, objects, n_bins=n_bins, allow_nan_as_seperate_bin=allow_nan_as_seperate_bin)
    info = single_cell_information(table)
    return info


def information_spike_pairs(spike_pair_histogram, objects, n_bins=3):
    """
    Calculate Information for spike pairs. Eguchi et al 2018
    :param spike_pair_histogram: numpy array of shape (n_stimuli, pre_neurons, post_neurons, delta_time)
    :param objects: list of length (n_objects), each of them is a list with ids of stimulus presentations that contain that object
    :param n_bins:
    :return: numpy array of shape (n_objects, delta_time, pre_neurons*post_neurons)
    """
    n_stim, n_pre, n_post, n_delta_time = spike_pair_histogram.shape
    reshaped_hist = np.moveaxis(spike_pair_histogram, 3, 1) # n_stimuli, delta_time, pre_neurons, post_neurons
    fake_layerwise_hist = np.reshape(reshaped_hist, (n_stim, n_delta_time, n_pre * n_post))

    info = firing_rates_numpy_to_single_cell_info(fake_layerwise_hist, objects, n_bins=n_bins)
    return info

def information_all_epochs(firing_rates_all_epochs, strategy, *args, **kwargs):
    """
    :param firing_rates_all_epochs: nested list of shape [epoch][stimulus][layer][excitatory/inhibitory] -> pandas dataframe with "ids" and "firing_rates"
    :param strategy: string that specifies which information score to use (e.g. single_cell_info, mutual_info)
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




class SingleCellDecoder(object):
    """
    Class that fits a threshold value for each neuron for each object.
    A neuron says 'yes the object is present in this stimulus' iff it's firing rate is above the fitted threshold.
    """

    label_bigger_dict = {0: False, 1: True}

    def __init__(self, allow_selectivity_by_being_off=False):
        """

        :param allow_selectivity_by_being_off: if this is True. A neuron that is consistently below the threshold for object A will cary information for ojbect A. Otherwise neurons are only thought to be selective to object A if they have higher response for it.
        True: label for higher firing rates might be true or false, False: label for higher firing rate is always true
        """
        if allow_selectivity_by_being_off:
            import warnings
            warnings.warn("Allowing a neuron to encode an objects present by it having a low firing rate to it is a bad idea and will screw with your brain. (i.e. in the case of mutually exclusive objects: On for object A <-> OFF for object B) ")

        self.thresholds = None # numpy array of shape [n_objects, n_layers, n_neurons] -> decission threshold
        self.label_bigger = None # shape [n_objects, n_layers, n_neurons] -> if true, all stimuli that have a higher FR in this neuron will get the label TRUE for that object, else FALSE

        self.false_positive_count = None # shape [n_thresholds, (label_bigger: false/true), n_objects, n_layers, n_neurons] -> for each decission boundrary configuration the number of false positives
        self.false_negative_count = None # shape [n_thresholds, (label_bigger: false/true), n_objects, n_layers, n_neurons] ->
        self.accuracy_train = None # [n_objects, n_layers, n_neurons] -> performance with the optimal threshold for that neuron

        if allow_selectivity_by_being_off:
            self.label_bigger_dict  = {0: False, 1: True}
        else:
            self.label_bigger_dict  = {0: True}



    def fit(self, firing_rate, label, n_steps=100):
        """
        fit a decision boundrary for each object for each label
        :param firing_rate: numpy array of shape [n_stimuli, n_layers, n_neurons]
        :param label: binary numpy array of shape [n_objects, n_stimuli] => weather or not this object is present in that stimuli
        :return: performance on the training set of shape [n_objects, n_layers, n_neurons]
        """
        n_stimuli, self.n_layers, self.n_neurons = firing_rate.shape
        self.n_objects, _n_stimuli = label.shape
        assert(_n_stimuli == n_stimuli)

        false_negative_count = np.zeros((n_steps, len(self.label_bigger_dict), self.n_objects, self.n_layers, self.n_neurons))
        false_positive_count = np.zeros_like(false_negative_count)
        self.accuracy_train = np.zeros((self.n_objects, self.n_layers, self.n_neurons))
        self.thresholds = np.zeros_like(self.accuracy_train)
        self.label_bigger = np.zeros_like(self.accuracy_train, dtype=bool)

        min_fr = np.min(firing_rate)
        max_fr = np.max(firing_rate)


        all_possible_thresholds = np.linspace(min_fr, max_fr, n_steps)


        tmp_prediction = np.zeros_like(firing_rate, dtype=bool)

        for threshold_i, threshold in enumerate(all_possible_thresholds):
            # try all thresholds
            for bigger_i, label_bigger in self.label_bigger_dict.items():
                # all threshold directions
                tmp_prediction[ firing_rate >  threshold] = label_bigger
                tmp_prediction[ firing_rate <= threshold] = not label_bigger

                # for all objects at a time
                tmp_prediction_all_objects_same = np.tile(tmp_prediction, (self.n_objects, 1, 1, 1)) # since it does not matt

                false_negative_count[threshold_i, bigger_i, :, :, :] = PerformanceSummary.get_false_negative_count(tmp_prediction_all_objects_same, label)
                false_positive_count[threshold_i, bigger_i, :, :, :] = PerformanceSummary.get_false_positive_count(tmp_prediction_all_objects_same, label)



        # find optimal boundrary

        performance = self.performance_measure(false_positive_count=false_positive_count, false_negative_count=false_negative_count, label=label)
        best_threshold_performances_both_sides = np.max(performance, axis=0) #optimal threshold [(label_bigger: (false,true), n_objects, n_layers, n_neurons

        best_threshold_direction = np.argmax(best_threshold_performances_both_sides, axis=0) # n_objects, n_layers, n_neurons
        best_threshold_performances = np.max(best_threshold_performances_both_sides, axis=0)



        for o in range(self.n_objects):
            for l in range(self.n_layers):
                for n in range(self.n_neurons):

                    performance_this = best_threshold_performances[o, l, n]
                    label_bigger_index = best_threshold_direction[o, l, n]

                    self.accuracy_train[o, l, n] = performance_this
                    best_thresholds = all_possible_thresholds[performance[:, label_bigger_index, o, l, n] == performance_this] # given object, layer, neuron all thresholds that had this performance (with the right threshold polarity i.e. label_bigger_index)

                    median_best_thresholds = np.percentile(best_thresholds, 50, interpolation='nearest')
                    self.thresholds[o, l, n]   = median_best_thresholds
                    self.label_bigger[o, l, n] = self.label_bigger_dict[label_bigger_index]

        return self.accuracy_train

    def transform(self, firing_rates):
        """
        Calculate boolean array with predictions for each stimulus.

        Note if a neuron always has firing rate zero in the training it will always predict the same truth value for all objects.

        :param firing_rates: of shape [n_stimuli, n_layers, n_neurons]
        :return: boolean array of shape [n_objects, n_stimuli, n_layers, n_neurons]
        """
        n_stimuli, n_layers, n_neurons = firing_rates.shape
        if self.thresholds is None:
            raise RuntimeError("You have to train first")


        result = np.zeros((self.n_objects, n_stimuli, n_layers, n_neurons), dtype=bool)

        for o in range(self.n_objects):
            for s in range(n_stimuli):
                thresholds_this = self.thresholds[o, :, :] # [n_layers, n_neurons]
                # thresholds_this_tiled = np.tile(thresholds_this, (n_stimuli, 1, 1)) # threshold is the same for each stimulus
                upper_half = firing_rates[s, :, :] > thresholds_this
                lower_half = np.invert(upper_half)

                result[o, s][upper_half] = self.label_bigger[o][upper_half]
                result[o, s][lower_half] = np.invert(self.label_bigger[o][lower_half])

        return result

    def get_performance_summary(self, firing_rates, label):
        """
        Calculate the performance of a trained decoder on this new set of presentations

        :param firing_rates: of shape [n_stimuli, n_layers, n_neurons]
        :param label: binary numpy array of shape [n_objects, n_stimuli] => weather or not this object is present in that stimuli

        :return an intance of type PerormanceSummary, you can call .accuracy on it for example. e.g. `decoder.get_performance_summary(rates, label).accuracy()`
        """
        predictions = self.transform(firing_rates)
        return PerformanceSummary(predictions, label)

    def performance_measure(self, false_positive_count, false_negative_count, label):
        """The threshold is fitted to maximise this value"""
        n_objects, n_stimuli = label.shape
        return 1 - ((false_negative_count + false_positive_count) / n_stimuli)


class PerformanceSummary:

    def __init__(self, predictions, label):
        """
        This Class is used to callculate all kinds of performance measures from a prediction and Labels
        :param predictions: np bool array of shape (objects, stimuli, layer, neurons)
        :param label: np bool array of shape (objects, stimuli)
        """
        assert(predictions.dtype == np.bool)
        assert(label.dtype == np.bool)
        assert(label.shape == predictions.shape[:2])

        self.predictions = predictions
        self.label = label

        self.n_examples = label.shape[1]

        self.n_class_positive = np.expand_dims(np.expand_dims(np.count_nonzero(label, axis=1), 1), 1) # for each object: in how many stimuli was it present
        self.n_class_negative = np.expand_dims(np.expand_dims(np.count_nonzero(np.invert(label), axis=1),1),1)

        self.n_false_positive = self.get_false_positive_count(self.predictions, self.label)
        self.n_true_negative = self.n_class_negative - self.n_false_positive # n_class_negative = true_negative + false_positive

        self.n_false_negative = self.get_false_negative_count(self.predictions, self.label)
        self.n_true_positive = self.n_class_positive - self.n_false_negative # n_class_true = false_negative + true_positive


    def accuracy(self):
        """How many of the examples were predicted correctly
        :return: np array of shape [n_objects, n_layer, n_neuron] -> accuracy of that neuron
        """
        return (self.n_true_negative + self.n_true_positive) / self.n_examples

    def precission(self):
        """How many of the stimuli precited as positive are in fact true positive
        :return: np array of shape [n_objects, n_layer, n_neuron] -> accuracy of that neuron
        """

        result =  self.n_true_positive / (self.n_true_positive + self.n_false_positive)

        result[self.n_true_positive == 0] = 0
        return result

    def recall(self):
        """
        How many stimuli that are in fact positive, will be predicted as positive

        :return: np array of shape [n_objects, n_layer, n_neuron] -> accuracy of that neuron
        """
        return self.n_true_positive / (self.n_class_positive)

    @classmethod
    def get_false_positive_count(cls, prediction, label):
        """
        :param prediction:  numpy array of shape [n_objects, n_stimuli, n_layers, n_neurons]
        :param label: [n_objects, n_stimuli]
        :return: np array of shape [n_objects, n_layers, n_neurons] -> false positive count of that neuron for that object
        """
        label = np.expand_dims(np.expand_dims(label, 2), 3)
        false_positive = np.invert(label) & prediction
        false_positive_count = np.count_nonzero(false_positive, axis=1)
        return false_positive_count.astype(int)

    @classmethod
    def get_false_negative_count(cls, prediction, label):
        """
        :param prediction:  numpy array of shape [n_objects, n_stimuli, n_layers, n_neurons]
        :param label: [n_objects, n_stimuli]
        :return: np array of shape [n_objects, n_layers, n_neurons] -> false negative count of that neuron for that object
        """
        label = np.expand_dims(np.expand_dims(label, 2), 3)
        false_negative = label & np.invert(prediction)
        false_negative_count = np.count_nonzero(false_negative, axis=1)
        return false_negative_count.astype(int)
