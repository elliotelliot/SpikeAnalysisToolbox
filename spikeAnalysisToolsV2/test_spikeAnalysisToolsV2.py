import pandas as pd
import numpy as np
import unittest
import sys

# import data_loading as data
sys.path.append("/Users/clemens/Documents/Code/AnalysisToolbox")
import spikeAnalysisToolsV2.firing_rates as firing


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



if __name__ == "__main__":
    unittest.main()
