import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from . import helper


def show_activity_in_layers(excitatory, inhibitory, value_range=None, item_labels=None, cmap='plasma'):
    """
    Plot activity or information for all items in the network. items can be stimuli or objects for example
    :param excitatory: values for excitatory neurons to be plotted, shape [item, layer, neuron_id]
    :param inhibitory: same for inhibitory neurons
    :param value_range: value range of the color map (optional)
    :param item_labels: names for the item subplots (optional)
    :param cmap: colormap (optional)
    """
    n_presentation_items = excitatory.shape[0] #how many stimuli or objects
    num_layers = excitatory.shape[1]
    if not value_range:
        vmin = min(excitatory.min(), inhibitory.min())
        vmax = max(excitatory.max(), inhibitory.max())
    else:
        vmin, vmax = value_range



    if not item_labels:
        item_labels = range(n_presentation_items)
    else:
        assert(len(item_labels) == n_presentation_items)


    exc_rates_imgs = helper.reshape_into_2d(excitatory)
    inh_rates_imgs = helper.reshape_into_2d(inhibitory)

    n_above_exc = np.count_nonzero(excitatory > 0.9 * vmax, axis=2)
    n_above_inh = np.count_nonzero(inhibitory > 0.9 * vmax, axis=2)

    for item_id, item in enumerate(item_labels):
        fig = plt.figure(figsize=(19, 8))
        fig.suptitle("Item: {}".format(item), fontsize=16)

        for layer in range(num_layers):
            subPlotAX = fig.add_subplot(2, num_layers, layer + 1)



            subPlotAX.set_title("Excitatory - Layer {}, ({} info)".format(layer, n_above_exc[item_id, layer]))
            subPlotAX.imshow(exc_rates_imgs[item_id, layer, :, :], vmin=vmin, vmax=vmax, cmap=cmap)

            subPlotAXinh = fig.add_subplot(2, num_layers, num_layers + layer + 1)

            subPlotAXinh.set_title("Inhibitory - Layer {} ({} info)".format(layer, n_above_inh[item_id, layer]))
            im = subPlotAXinh.imshow(inh_rates_imgs[item_id, layer, :, :], vmin=vmin, vmax=vmax, cmap=cmap)

        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        fig.colorbar(im, cax=cax)


def animate_2d_matrix(data, perf, title, label_perf=None, cmap='plasma'):
    """
    Animate a 2d matrix
    :param data: numpy array of shape [frame, width, height]
    :param perf: numpy array of shape [timepoints, layer] -> performace value
    :param title: as string
    :return:
    """
    if len(data.shape) == 2:
        print("Data only has shape {}".format(data.shape))
        n_epochs = data.shape[0]

        side_length =  helper.get_side_length(data.shape[-1])
        data = np.reshape(data, (n_epochs, side_length, side_length))
        print("Data was reshaped into {}".format(data.shape))





    n_timepoints, width, height = data.shape
    _n_t, n_perf_measures = perf.shape
    epochs = np.arange(n_timepoints)
    assert(n_timepoints == _n_t)

    if label_perf is None:
        label_perf = ["Layer {}".format(l) for l in range(n_perf_measures)]
    else:
        assert(len(label_perf) == n_perf_measures)

    vextreme = np.max(np.abs(data))
    vmax = vextreme

    if(np.any(data<0)):
        vmin = - vextreme
        # we are plotting a difference so we want 0 to be in the middle
    else:
        vmin = np.min(data)
        # we are not plotting a difference so we don't care where 0 is

    # vmin= np.min(data)
    # vmax = np.max(data)

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 2, 1)
    ax.set_title(title)
    ax.axis('off')
    # fig.colorbar(ax)
    im = ax.imshow(data[0, :, :], animated=True, cmap=cmap, vmin=vmin, vmax=vmax)

    cax = fig.add_axes([0.05, 0.1, 0.4, 0.05])
    fig.colorbar(im, cax=cax, orientation='horizontal')



    perfAx = fig.add_subplot(1,2,2)
    pmin = np.min(perf)
    pmax = np.max(perf)
    prange = pmax-pmin

    perfAx.set_ylim(pmin - 0.1*prange, pmax + 0.1*prange)
    perfAx.set_xlim(epochs[0], epochs[-1])
    layer_lines = [perfAx.plot(epochs[0], perf[0, l], label=label_perf[l]) for l in range(n_perf_measures)]
    perfAx.legend()

    def update_perf(frame):
        for l, layer_line in enumerate(layer_lines):
            layer_line[0].set_data(epochs[:frame+1], perf[:frame+1, l])


    def update(frame):
        im.set_array(data[frame, :, :])
        update_perf(frame)

    ani = animation.FuncAnimation(fig, update, frames=n_timepoints, interval=500, blit=False, repeat_delay=22000)

    return ani




def animate_neuron_value_development(exc, inh):
    """
    Animate the development of a value per neuron
    :param exc: numpy array of shape [timepoint, layer, neuronid]
    :param inh: same
    :return: Animation
    """
    n_timepoints, n_layers, _n_neurons = exc.shape

    max_firing_rate = max(np.max(exc), np.max(inh))

    exc_img = helper.reshape_into_2d(exc)
    inh_img = helper.reshape_into_2d(inh)

    fig = plt.figure(figsize=(19, 8))

    exc_axes = []
    inh_axes = []
    for l in range(n_layers):
        exc_axes.append(fig.add_subplot(2, n_layers, 1 + l))
        exc_axes[-1].axis('off')
    for l in range(n_layers):
        inh_axes.append(fig.add_subplot(2, n_layers, n_layers + 1 + l))
        inh_axes[-1].axis('off')

    ims = []
    for frame in range(n_timepoints):
        # image_arr = np.reshape(exc[3, frame, :], (64, 64), order='F')
        images_Ex = []
        images_In = []
        for l in range(n_layers):
            imEx = exc_axes[l].imshow(exc_img[frame, l, :, :], animated=True, cmap='hot', vmin=0, vmax=max_firing_rate)
            imIn = inh_axes[l].imshow(inh_img[frame, l, :, :], animated=True, cmap='hot', vmin=0, vmax=max_firing_rate)

            images_Ex.append(imEx)
            images_In.append(imIn)

        ims.append(images_Ex + images_In)

    cax = fig.add_axes([0.92, 0.17, 0.03, 0.67])
    fig.colorbar(imIn, cax=cax)

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=22000)
    return ani

def show_values_all_things(values, figure_title, thing_label = None, cmap='plasma'):
    """
    show values for all thingks

    :param values: np array of shape [thing, neuron_id] or [thing, rows, columns]
    :param figure_title: title of the figure
    :param cmap: colormap to be used
    :return:
    """
    if len(values.shape)==1:
        values = np.expand_dims(values, 0)
    num_layers = values.shape[0]

    if thing_label == None:
        thing_label = ["values for Layer {}".format(l) for l in range(num_layers)]
    else:
        assert(num_layers == len(thing_label))

    vmin =values.min()
    vmax =values.max()

    fig = plt.figure(figsize=(12, 5*np.ceil(num_layers/2)),)
    fig.suptitle(figure_title, fontsize=16)

    if(len(values.shape) > 1 and values.shape[-2] != values.shape[-1]):
        reshaped = helper.reshape_into_2d(values)
    else:
        reshaped = values

    for layer in range(num_layers):
        subPlotAX = fig.add_subplot(np.ceil(num_layers/2), 2, layer + 1)

        subPlotAX.set_title(thing_label[layer])
        im = subPlotAX.imshow(reshaped[layer, :, :], vmin=vmin, vmax=vmax, cmap=cmap)

    #cax = fig.add_axes([0.95, 0.2, 0.03, 0.6])
    #fig.colorbar(im, cax=cax)






def plot_information_measure_advancement(before, after, n_to_plot = 1000, item_label=None):
    assert(before.shape == after.shape)
    n_objects, n_layer, n_neurons = before.shape

    if not item_label:
        item_label = list(range(n_objects))
    else:
        assert(len(item_label) == n_objects)

    before = np.sort(before, axis=2)
    after = np.sort(after, axis=2)

    vmax = max(np.max(before), np.max(after))


    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Information Measure before and after", fontsize=16)


    for layer in range(n_layer):
        for i, item in enumerate(item_label):

            layerAX  = fig.add_subplot(n_layer, n_objects, (n_objects * layer) + i + 1)
            layerAX.set_title("Info Item: {}, Layer {}".format(item, layer))

            layerAX.plot(before[i, layer, :-n_to_plot:-1], label="before")
            layerAX.plot( after[i, layer, :-n_to_plot:-1], label="after")

            layerAX.set_ylim(-0.1 * vmax, 1.1 * vmax)
            layerAX.legend()


def plot_ranked_neurons(list_of_things, title, n_to_plot=100, item_label=None, vmin=None, vmax=None, figsize=(17,6)):
    """
    Plot ranked neuron value. there will be as many subplots as layer. each contains as many lines as there are things
    value of first line at x=5 is the value of the 5th best neuron (with respect to the first thing)
    value of second line at x=5 is value of 5th best (with respect to the second thing) -> both values do NOT belong to the same neuron

    :param title: name of the plot
    :param list_of_things: np array of shape [n_things, layer, neuron]
    :param n_to_plot: how many neurons to plot
    :param item_label: list of strings that name the things (optional)
    :return:
    """
    n_things, n_layer, n_neurons = list_of_things.shape

    if not item_label:
        item_label = ["Thing Nr. {}".format(t) for t in range(n_things)]
    else:
        assert(len(item_label) == n_things)

    sorted_stuff = np.sort(list_of_things, axis=2)

    if vmax is None:
        vmax = np.max(sorted_stuff)
    if vmin is None:
        vmin = np.min(sorted_stuff)


    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16)


    for layer in range(n_layer):
        layerAX  = fig.add_subplot(1, n_layer, layer + 1)
        layerAX.set_title("Layer {}".format(layer + 1))
        layerAX.set_xlabel("Neuron Rank")
        layerAX.set_ylim(vmin, 0.1 * (vmax-vmin) + vmax)
        for i, item in enumerate(item_label):
            layerAX.plot(sorted_stuff[i, layer, :-n_to_plot-1:-1], label=item)

        layerAX.legend()

def plot_fr_histogram(nested_firing_rates, stimulus_names = None, n_bins=100):
    """

    :param nested_firing_rates: firing rates for one epoch (multiple stimuli though)
    :return:
    """
    excitatory_rates, inhibitory_rates = helper.nested_list_of_stimuli_2_np(nested_firing_rates)

    num_stimuli, num_layers, num_neurons = excitatory_rates.shape

    if stimulus_names is None:
        stimulus_names = range(num_stimuli)
    else:
        assert(num_stimuli == len(stimulus_names))

    bins = np.linspace(0, 150, n_bins)

    for stimulus in range(num_stimuli):
        fig = plt.figure(figsize=(15, 28))
        fig.suptitle("Stimulus: {}".format(stimulus_names[stimulus]), fontsize=16)

        subpltEX = fig.add_subplot(1, 2,  1)
        subpltEX.set_title("Excitatory", fontsize=14, fontweight="bold")
        subpltEX.set_xlabel("Frequency [Hz]")

        subpltIN = fig.add_subplot(1, 2,  2)
        subpltIN.set_title("Inhibitory", fontsize=14, fontweight="bold")
        subpltIN.set_xlabel("Frequency [Hz]")

        for layer in range(num_layers):
            subpltEX.hist(excitatory_rates[stimulus, layer], label="Layer: {}".format(layer), bins = bins, histtype='step')
            subpltIN.hist(inhibitory_rates[stimulus, layer], label="Layer: {}".format(layer), bins = bins, histtype='step')


def plot_fr_ranked(nested_firing_rates, stimulus_names = None, ylim=150, percentage_to_plot=0.5):
    """

    :param nested_firing_rates: firing rates for one epoch (multiple stimuli though)
    :return:
    """
    if type(nested_firing_rates) != np.ndarray:
        excitatory_rates, inhibitory_rates = helper.nested_list_of_stimuli_2_np(nested_firing_rates)
    else:
        print("PITA: only one numpy array given, inhibitory rates are fake")
        excitatory_rates = nested_firing_rates
        inhibitory_rates = np.zeros_like(excitatory_rates)

    num_stimuli, num_layers, num_neurons = excitatory_rates.shape

    plot_n_exc = int(excitatory_rates.shape[-1]*percentage_to_plot)
    plot_n_inh = int(inhibitory_rates.shape[-1]*percentage_to_plot)

    if stimulus_names is None:
        stimulus_names = range(num_stimuli)
    else:
        assert(num_stimuli == len(stimulus_names))

    # dimensions: [stimulus, layer, neuron_id]
    exc_rates_sorted = np.sort(excitatory_rates,
                               axis=2)[:, :, :-plot_n_exc:-1]  # sort so that the firing rates in each stimulus and layer are sorted
    # dimensions: [stimulus, layer, neuron_activity_rank]
    inh_rates_sorted = np.sort(inhibitory_rates, axis=2)[:,:, :-plot_n_inh:-1]

    fig_collector = list()
    for stimulus in range(num_stimuli):
        fig = plt.figure(figsize=(15, 7))
        fig.suptitle("Stimulus: {} Nr {}".format(stimulus_names[stimulus], stimulus), fontsize=16)

        subpltEX = fig.add_subplot(1, 2, 1)
        plt.tight_layout(pad=8.0)
        subpltEX.set_title("Excitatory: max: {}".format(exc_rates_sorted[stimulus, :, 0]), fontsize=14, fontweight="bold")
        subpltEX.set_ylabel("Frequency [Hz]")
        subpltEX.set_xlabel("Neuron rank")
        subpltEX.set_ylim(0, ylim)

        subpltIN = fig.add_subplot(1, 2, 2)
        plt.tight_layout(pad=8.0)
        subpltIN.set_title("Inhibitory: max: {}".format(inh_rates_sorted[stimulus, :, 0]), fontsize=14, fontweight="bold")
        subpltIN.set_ylabel("Frequency [Hz]")
        subpltIN.set_xlabel("Neuron rank")
        subpltIN.set_ylim(0, ylim)

        for layer in range(num_layers):
            subpltEX.plot(exc_rates_sorted[stimulus, layer], label="Layer {}".format(layer))
            subpltIN.plot(inh_rates_sorted[stimulus, layer], label="Layer {}".format(layer))

        subpltEX.legend()
        fig_collector.append(fig)
    return fig_collector



def plot_information_difference_development(info, threshold):
    """
    plot how the difference in information between 2 stimuli developed
    :param info: np array of shape [epochs, objects, layer, neuron_id]
    :return:
    """
    n_epochs, n_objects, n_layer, n_neurons = info.shape

    if(n_objects !=2):
        raise NotImplementedError("At the moment, it only knows how to compare 2 objects.")

    avg_info = np.mean(info, axis=3)

    avg_info_1_minus_0 = avg_info[:, 1, :] - avg_info[:, 0, :]

    avg_max = np.max(avg_info_1_minus_0)

    n_above_threshold = np.count_nonzero( (info >= threshold), axis=3 )


    above_max_1_minus_0 = n_above_threshold[:, 1, :] - n_above_threshold[:, 0, :]
    max_n_above_threshold = np.max(above_max_1_minus_0)

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("Development of the difference in information", fontsize=16)


    axAvg  = fig.add_subplot(1, 2, 1)
    axAvg.set_ylim(-1.1 * avg_max, 1.1 * avg_max)
    axAvg.set_title("Average info for stimulus 1 - avg info for stimulus 0")


    axN_neurons = fig.add_subplot(1, 2, 2)
    axN_neurons.set_ylim(-1.1 * max_n_above_threshold, 1.1 * max_n_above_threshold)
    axN_neurons.set_title("Number of neurons above {}".format(threshold))

    for l in range(n_layer):

        axAvg.plot(avg_info_1_minus_0[:, l], label= "Layer {}".format(l))
        axN_neurons.plot(above_max_1_minus_0[:, l], label="Layer {}".format(l))

    axAvg.legend()
    axN_neurons.legend()

def plot_information_development(info, epochs=None, mean_of_top_n = 'all', threshold=0.8, item_label=None, lower_y_lim=0):
    """
    plot how the information developed
    :param info: np array of shape [epochs, objects, layer, neuron_id]
    :return:
    """
    if len(info.shape) != 4:
        info = np.expand_dims(info, axis=1)
        # The case that we have an information score that does not have one value for each item but a combined value
        item_label = ["combined value"]

    n_epochs, n_objects, n_layer, n_neurons = info.shape
    if epochs==None:
        epochs=np.arange(n_epochs)

    if type(epochs) != list and type(epochs) != np.ndarray:
        epochs = list(epochs)

    if not item_label:
        item_label = list(["Item {}".format(i) for i in range(n_objects)])
    else:
        assert(len(item_label) == n_objects)

    if mean_of_top_n == 'all':
        avg_info = np.mean(info, axis=3)
    else:
        info_top_n = np.sort(info, axis=3)[:, :, :, -1-mean_of_top_n:]
        avg_info = np.mean(info_top_n, axis=3)

    avg_max = np.max(avg_info)

    n_above_threshold = np.count_nonzero( (info >= threshold), axis=3 )
    n_above_max = np.max(n_above_threshold)

    fig = plt.figure(figsize=(18, 15))
    fig.suptitle("Development of the information", fontsize=16)

    for i, item in enumerate(item_label):

        axAvg  = fig.add_subplot(2, n_objects, i + 1)
        axAvg.set_ylim(lower_y_lim, 1.1 * avg_max)
        axAvg.set_title("Average info of top {} neurons for {}".format(mean_of_top_n, item))


        axN_neurons = fig.add_subplot(2, n_objects, n_objects + i + 1)
        axN_neurons.set_ylim(0, 1.1 * n_above_max)
        axN_neurons.set_title("Number of neurons above {}".format(threshold))

        for l in range(n_layer):

            axAvg.plot(avg_info[:, i, l], label= "Layer {}".format(l))
            axN_neurons.plot(epochs, n_above_threshold[:, i, l], label="Layer {}".format(l))

        axAvg.legend()
        axN_neurons.legend()

    return fig


def plot_firing_rates_colored_by_object(firing_rates, object_list, title_string):
    if len(firing_rates.shape) != 1:
        raise ValueError("firing_rates has to be a single 'timecourse' of firing rates. only one neuron (or the mean) at a time")


    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(title_string)
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Firing Rates for Stimulus presentations, colored by object which contains the indicated stimuli")

    for obj in object_list:
        ids_in_obj = obj['indices']

        ax.plot(ids_in_obj, firing_rates[ids_in_obj], 'x', label=obj['elements'])

    ax.legend()


def plot_mean_rates_by_stim(firing_rates, stimulus_ids, title_string, ylims=(0, 60), threshold=None, comparison_rates=None, rates_label=None, stimulus_sort_key=None):
    """
    Make plot with labeled stimulus on x axis and firing rate on y axis. then each stimulus presentations is ploted by one marker and an error bar
    Optinally the firing rates untrained can be plotted too.

    :param firing_rates:
    :param stimulus_ids: dict with stimulus_name: list of indices at which that stimulus was presented
    :param title_string:
    :param comparison_rates: same shape as firing_rates. if this is provided two rates will be plotted. The names will be given by rate_labels
    :param rate_labels: tuple with two elemnts which are the labels for (firing_rates, comparison_rates) respectivly.
    :return:
    """
    if len(firing_rates.shape) != 1:
        raise ValueError("firing_rates has to be a single 'timecourse' of firing rates. only one neuron (or the mean) at a time")

    if comparison_rates is not None:
        list_of_rates = [firing_rates, comparison_rates]
        if rates_label is None:
            rates_label = ["Trained Network", "Initial_network"]
    else:
        list_of_rates = [firing_rates]
        assert(rates_label is None)
        rates_label = [None]

    if stimulus_sort_key is not None:
        ordered_stim_names = sorted(stimulus_ids.keys(), key = stimulus_sort_key)
    else:
        ordered_stim_names = list(stimulus_ids.keys())


    fig = plt.figure(figsize=(13, 7))
    # fig.suptitle(title_string)
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title_string)

    x_axis = np.arange(len(stimulus_ids.keys()))

    for i, rates in enumerate(list_of_rates):
        mean_fr = np.zeros(len(stimulus_ids.keys()))
        error_fr = np.zeros_like(mean_fr)
        x_ticks = list()

        # for stim_nr, (stim_name, presentation_ids) in enumerate(stimulus_ids.items()):
        for stim_nr, stim_name in enumerate(ordered_stim_names):
            presentation_ids = stimulus_ids[stim_name]
            x_ticks.append((stim_nr, stim_name))

            mean_fr[stim_nr] = np.mean(rates[presentation_ids], axis=0)
            error_fr[stim_nr] = np.std(rates[presentation_ids], axis=0)

        ax.scatter(x_axis+(0.1*i), mean_fr, marker='x', label=rates_label[i])
        ax.errorbar(x_axis+(0.1*i), mean_fr, yerr=error_fr, fmt='none')

    if rates_label:
        ax.legend()

    if threshold:
        ax.axhline(y=threshold, ls=":", c="black")

    ax.set_ylim(*ylims)

    x_tick_pos, x_tick_string = zip(*x_ticks)

    plt.xticks(x_tick_pos, x_tick_string, rotation=45)

    ax.legend()





def plot_firing_rates_by_stim(firing_rates, stimulus_ids, title_string, mean_with_errorbar=False, color_indices=None, ylims=(0, 60), threshold=None):
    """
    Make plot with labeled stimulus on x axis and firing rate on y axis. then each stimulus presentations is ploted by one marker in a scatter plot

    :param firing_rates:
    :param stimulus_ids: dict with stimulus_name: list of indices at which that stimulus was presented
    :param title_string:
    :return:
    """
    if len(firing_rates.shape) != 1:
        raise ValueError("firing_rates has to be a single 'timecourse' of firing rates. only one neuron (or the mean) at a time")


    colors = 'r'
    if color_indices:
        colors = np.zeros_like(firing_rates)
        for i, one_color_obj in enumerate(color_indices):
            colors[one_color_obj] = i

    x_axis = np.zeros_like(firing_rates)
    x_ticks = list()
    for stim_nr, (stim_name, presentation_ids) in enumerate(stimulus_ids.items()):
        x_axis[presentation_ids] = stim_nr+1
        x_ticks.append((stim_nr+1, stim_name))


    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(title_string)
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Firing Rates for Stimulus presentations, colored by object which contains the indicated stimuli")

    ax.scatter(x_axis, firing_rates, marker='x', c=colors)

    if threshold:
        ax.axhline(y=threshold)

    ax.set_ylim(*ylims)

    x_tick_pos, x_tick_string = zip(*x_ticks)

    plt.xticks(x_tick_pos, x_tick_string, rotation=45)

    ax.legend()

def plot_firing_rates_std_vs_mean_colored_by_object(firing_rates, object_list, title_string):
    """
    within a set of neurons (usually a layer) the mean fr and the std of it is computed and plotted

    :param firing_rates: np array of shape (objects, neuron_ids)
    :param object_list:
    :param title_string:
    :return:
    """
    if len(firing_rates.shape) != 2:
        raise ValueError("firing_rates has to be a single 'timecourse' of firing rates. only one neuron (or the mean) at a time")


    fr_mean = np.mean(firing_rates, axis=1)
    fr_std = np.std(firing_rates, axis=1)

    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(title_string)
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Firing Rate mean vs std for Stimulus presentations, colored by object which contains the indicated stimuli")

    for obj in object_list:
        ids_in_obj = obj['indices']

        ax.plot(fr_std[ids_in_obj], fr_mean[ids_in_obj], 'x', label=obj['elements'])

    ax.set_ylabel("Mean Firing Rate")
    ax.set_xlabel("Standard diviation of Firing Rate")
    ax.legend()

def plot_value_comparisson(name_a, value_a, name_b,  value_b, object_label=None, title="", threshold_a=None, threshold_b=None, vmax=1, vmin=0):
    """
    Make n_layer many
    Two accis scatter plot to compare values of a and b. y axis is value of a for that neuron, x axis is value of b for that neuron

    :param value_a: np.array of shape [n_objects, n_layers, n_neurons]
    :param value_b: same as value_a
    :param threshold_a: scalar, only neurons who's a value is above this will be considered
    :param threshold_b: scala, only neurons who's b value is above this will be considered
    :return:
    """
    assert(value_a.shape == value_b.shape)

    n_objects, n_layers, n_neurons = value_a.shape
    n_rows = np.ceil(n_layers/2)

    if threshold_a is None:
        threshold_a = np.min(value_a)
    if threshold_b is None:
        threshold_b = np.min(value_b)

    if object_label is None:
        object_label = ["Object {}".format(o) for o in range(n_objects)]
    else:
        assert(n_objects == len(object_label))



    fig = plt.figure(figsize=(13, 13))
    fig.suptitle(title)

    for l in range(n_layers):
        ax = fig.add_subplot(n_rows, 2, l+1)

        cor = np.corrcoef((value_a[:, l].flatten()), value_b[:, l].flatten())

        ax.set_title("Layer {}, correlation: {}".format(l, cor[0, 1]))

        for o in range(n_objects):
            mask = (value_a[o, l, :] >= threshold_a) & (value_b[o, l, :] >= threshold_b)
            selected_a = value_a[o, l][mask]
            selected_b = value_b[o, l][mask]
            ax.scatter(selected_b, selected_a, marker=".", label=object_label[o], s=5)
            ax.plot([vmin, vmax], [vmin, vmax], ls=":", c="black")

        ax.set_ylim(vmin, vmax)
        ax.set_xlim(vmin, vmax)
        ax.set_xlabel(name_b)
        ax.set_ylabel(name_a)
        ax.legend()







def animated_histogram(data, n_bins=10, item_label=None, log=True):
    """
    Plot animated histograms of the data, one frame for each epoch

    :param data: numpy array of shape [epochs, objects, layer, neuron_id]
    :param item_label: lables of the items.
    :return:
    """
    n_epochs, n_objects, n_layer, n_neurons = data.shape
    if not item_label:
        item_label = list(range(n_objects))
    else:
        assert(len(item_label) == n_objects)

    vmin = np.min(data)
    vmax = np.max(data)

    bins = np.linspace(vmin, vmax, n_bins +1)

    fig = plt.figure(figsize=(19, 8))

    object_axes = []
    for obj in range(n_objects):
        layer_in_obj_axes = []
        for l in range(n_layer):
            layer_and_obj_axis = fig.add_subplot(n_layer, n_objects, l * n_objects + obj + 1)
            layer_and_obj_axis.hist(data[0, obj, l, :], bins=bins, log=log)
            # layer_and_obj_axis.yscale('log', nonposy='clip')
            layer_in_obj_axes.append(layer_and_obj_axis)
        object_axes.append(layer_in_obj_axes)

    def update_hist(num):
        for obj in range(n_objects):
            for l in range(n_layer):
                object_axes[obj][l].cla()
                object_axes[obj][l].hist(data[num, obj, l, :], bins=bins, log=log)

    ani = animation.FuncAnimation(fig, update_hist, n_epochs)
    return ani



def plot_development_for_object(values, title, object_labels=None):
    """
    Plot the development of an arbitrary measure seperatly for each neuron
    :param values: numpy array of shape [epoch, object, layer]
    :param title: string with title for the figure
    :param object_labels: optional object label
    :return:
    """
    n_epochs, n_objects, n_layer = values.shape

    minval = np.min(values)
    maxval = np.max(values)

    if object_labels is None:
        object_labels = ['Obj {}'.format(o) for o in range(n_objects)]
    else:
        assert(n_objects == len(object_labels))


    fig = plt.figure(figsize=(15,8))
    fig.suptitle(title)

    for obj_id, obj_name in enumerate(object_labels):
        ax = fig.add_subplot(1, n_objects, obj_id+1)
        ax.set_title(obj_name)
        ax.set_ylim(minval, maxval)
        for l in range(n_layer):
            ax.plot(values[:, obj_id, l], label="Layer {}".format(l))

    ax.legend()





def show_connection_fields(synapses, layer, is_excitatory, neuron_positions, network_architecture, mode='sources'):
    """

    :param synapses:
    :param layer:
    :param is_excitatory:
    :param neuron_positions:
    :param network_architecture:
    :param mode: 'sources' or 'targets' plot the source neurons that map onto the ones given by neuron_position or plot the target neurons that the neuron_positions neuron map onto
    :return:
    """

    from . import synapse_analysis
    n_plots = len(neuron_positions)

    if is_excitatory:
        input_neuron_layer_side_length = helper.get_side_length(network_architecture["num_exc_neurons_per_layer"])
        # sidelength of the layer in which the neuron lives, who's position we inputed
    else:
        input_neuron_layer_side_length = helper.get_side_length(network_architecture["num_inh_neurons_per_layer"])


    fig = plt.figure("Receptive field for the following neurons in layer {}".format(layer), figsize=(19,8))
    for i,pos in enumerate(neuron_positions):

        ax = fig.add_subplot(2, np.ceil(n_plots/2), i+1)

        if(mode=='sources'):
            recpField = synapse_analysis.receptive_field_of_neuron((layer,) + pos, is_excitatory, synapses, network_architecture)
        elif(mode=='targets'):
            recpField = synapse_analysis.targets_of_neuron((layer,) + pos, is_excitatory, synapses, network_architecture)
        else:
            raise ValueError("mode can only be sources, or targets")


        ax.set_title("Neuron at {}, n_synapses: {}".format(pos, np.sum(recpField)))

        plotting_layer_side_length = recpField.shape[0]
        # sidelength of layer in which the receptive field lives

        factor = plotting_layer_side_length / input_neuron_layer_side_length
        # if for example we are looking at E2I Lateral synapses, then there are different number of neurons in the presynamptic and the postsynaptic layer
        # this factor mitigates that. the center of the receptive field of inhibitory neuron 16,16 is placed over the excitatory neuron 32, 32
        # because there are 32x32 inhibitory neurons and 64x64 excitatory ones

        im = ax.imshow(recpField, cmap='plasma')
        ax.scatter([pos[1]*factor], [pos[0]*factor], color="green", marker='x', s=500)
        fig.colorbar(im)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_ylim(0, recpField.shape[0])
        ax.set_xlim(0, recpField.shape[1])

