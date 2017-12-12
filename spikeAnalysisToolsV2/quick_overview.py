"""Maent to be run as a comand line tool to produce some overviews of the experiment"""

from optparse import OptionParser
import numpy as np
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append("/Users/clemens/Documents/Code/AnalysisToolbox")
from spikeAnalysisToolsV2 import data_loading as data
from spikeAnalysisToolsV2 import firing_rates as firing
from spikeAnalysisToolsV2 import information_scores as info
from spikeAnalysisToolsV2 import plotting as spikeplot

output_path = "/Users/clemens/Documents/Code/ModelClemens/output"
overview_folder_name = "overview"

network_architecture = dict(
    num_exc_neurons_per_layer = 64*64,
    num_inh_neurons_per_layer = 32*32,
    num_layers = 4
)

info_times = dict(
    length_of_stimulus = 2.0,
    # num_stimuli = n_stimuli,
    time_start = 1.5,
    time_end = 1.9
)

plot_fr_at = [0, 0.5, 1]

def main():
    parser = OptionParser()
    parser.add_option("-n", "--name", dest="name", help="Name of th experiment", action="store")

    options, args = parser.parse_args()

    if options.name is None:
        print("No experiment name provided")
        raise ValueError("No expeiment name provided")

    summary_for_experiment(options.name)


def make_pdf(name, figs):
    pdf = PdfPages(name)
    for f in figs:
        pdf.savefig(f)
    pdf.close()

def summary_for_experiment(name):
    exp_path = output_path + "/" + name

    overview_path = exp_path + "/" + overview_folder_name

    epoch_names = ["initial"] + data.get_epochs(exp_path)

    object_list = data.load_testing_stimuli_info(exp_path)
    n_stimuli = np.sum(obj['count'] for obj in object_list)
    object_indices = [obj['indices'] for obj in object_list]
    stimuli_names = data.load_testing_stimuli_names(exp_path)

    info_times["num_stimuli"] = n_stimuli

    spikes = data.load_spikes_from_subfolders(output_path, [name], epoch_names, False)

    rates_subfolders = firing.calculate_rates_subfolder(
        spikes,
        network_architecture,
        info_times)

    spikes = None

    exc_information, inh_information = info.single_cell_information_all_epochs(rates_subfolders[0],
                                                                               object_indices, 3)

    ### make and save plots
    os.mkdir(overview_path)

    info_dev_plot = spikeplot.plot_information_development(exc_information, mean_of_top_n=500, threshold=1)
    info_dev_plot.savefig(overview_path+ "/performance_dev.png")

    for r in plot_fr_at:
        epoch_id = int(r * (len(epoch_names)-1))

        cur_epoch_fr_plot = spikeplot.plot_fr_ranked(rates_subfolders[0][epoch_id], stimuli_names, percentage_to_plot=0.5)
        epoch_name = epoch_names[epoch_id].split("/")[-1]
        make_pdf("{}/firing_rates_{}.pdf".format(overview_path, epoch_name), cur_epoch_fr_plot)

    return


if __name__ == "__main__":
    main()
