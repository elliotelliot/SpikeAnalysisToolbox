import pandas as pd
import numpy as np
import unittest
import sys
from timeit import default_timer as timer


sys.path.append("/Users/clemens/Documents/Code/AnalysisToolbox")

import spikeAnalysisToolsV2.data_loading as data
import spikeAnalysisToolsV2.firing_rates as firing
import spikeAnalysisToolsV2.helper as helper
import spikeAnalysisToolsV2.overviews as overview
import spikeAnalysisToolsV2.combine_stimuli as combine
import spikeAnalysisToolsV2.plotting as spikeplot
import spikeAnalysisToolsV2.information_scores as info




class Test_FiringRates(unittest.TestCase):
    spikes = pd.DataFrame({
        'ids':   [0, 0, 0, 1 , 0, 1],
        'times': [0.1, 0.11, 0.12, 0.5, 0.7, 0.99]
    })
    def test_FR(self):
        rates = firing.spikesToFR(Test_FiringRates.spikes, (0, 10), (0, 1.0))
        assert(rates.firing_rates.values[0] == 4)
        assert(rates.firing_rates.values[1] == 2)

    def test_instantainiousFR(self):
        times, instant_FR = firing.spikes_to_instantanious_FR(Test_FiringRates.spikes, (0, 10), 0.2, (0, 1.0))
        assert(np.all(instant_FR[:, 0] == np.array([15, 0, 0, 5, 0])))
        assert(np.all(instant_FR[:, 1] == np.array([0, 0, 5, 0, 5])))


class Test_combine_stimuli(unittest.TestCase):

    def test_single_cell_information(self):
        masterpath = "/Users/clemens/Documents/Code/ModelClemens/output"
        ## set the subfolder to the Simulation you want to analyse

        subfolders = [
            "2017:10:23-17:39:33_only_loc_1_lots_of_testing_300_epochs"
        ]
        ## if more than the inital epoch is needed *1 needs to be run
        extensions = [
            "testing/epoch20"
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
            num_stimuli=56,
            time_start=1.5,
            time_end=1.9
        )
        n_bins = 3

        spikes = data.load_spikes_from_subfolders(masterpath, subfolders, extensions, input_layer=False)

        firing_rates = firing.calculate_rates_subfolder(spikes, network_architecture, info_times)

        rates = firing_rates[0][0]
        exc_rates, inh_rates = helper.stimulus_layer_nested_list_2_numpy_tensor(rates)

        objects = [list(range(28)), list(range(28, 28 + 28))]  # each is presented twice



        start = timer()
        exc_table = combine.response_freq_table(exc_rates, objects, n_bins=n_bins)

        exc_info = info.single_cell_information(exc_table)

        final_new_info_l2 = exc_info[:, 2, :] # shape [2, n_neurons] for both objects all info in layer 2
        print("New version of info calc took {}s".format(timer() - start))

        ### calculating it the old way
        import InformationAnalysis as infoOld

        categories = np.zeros((56, 1))
        categories[28:, 0] = 1 # simulating the fact that we have 2 objects -> stimuli with only one attribute, object_id 1 or 0

        start = timer()
        digitized_firing_rates = firing.digitize_firing_rates_with_equispaced_bins(exc_rates, n_bins=n_bins)

        digitized_firing_rates_l2 = digitized_firing_rates[:, 2, :]


        freq_table_old = infoOld.frequency_table(digitized_firing_rates_l2, categories, list(range(n_bins)))
        final_old_info_l2 = infoOld.single_cell_information(freq_table_old, categories)[0]
        print("Old version of info calc took {}s".format(timer() - start))

        assert(np.all(np.isclose(final_new_info_l2, final_old_info_l2)))










if __name__ == "__main__":
    unittest.main()
