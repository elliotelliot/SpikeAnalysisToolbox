import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from . import helper


def plot_activity_in_layers(excitatory, inhibitory, value_range=None, item_labels=None, cmap='plasma'):
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




def plot_information_development(info, threshold, item_label=None):
    """
    plot how the information developed
    :param info: np array of shape [epochs, objects, layer, neuron_id]
    :return:
    """
    n_epochs, n_objects, n_layer, n_neurons = info.shape
    if not item_label:
        item_label = list(range(n_objects))
    else:
        assert(len(item_label) == n_objects)

    avg_info = np.mean(info, axis=3)
    avg_max = np.max(avg_info)

    n_above_threshold = np.count_nonzero( (info >= threshold), axis=3 )
    n_above_max = np.max(n_above_threshold)

    fig = plt.figure(figsize=(18, 15))
    fig.suptitle("Development of the information", fontsize=16)

    for i, item in enumerate(item_label):

        axAvg  = fig.add_subplot(n_objects, 2, i + 1)
        axAvg.set_ylim(0, 1.1 * avg_max)
        axAvg.set_title("Average info for Item {}".format(item))


        axN_neurons = fig.add_subplot(n_objects, 2, n_objects + i + 1)
        axN_neurons.set_ylim(0, 1.1 * n_above_max)
        axN_neurons.set_title("Number of neurons above {}".format(threshold))

        for l in range(n_layer):

            axAvg.plot(avg_info[:, i, l], label= "Layer {}".format(l))
            axN_neurons.plot(n_above_threshold[:, i, l], label="Layer {}".format(l))

        axAvg.legend()
        axN_neurons.legend()

def plot_animated_histogram(data, n_bins=10, item_label=None):
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
            layer_and_obj_axis.hist(data[0, obj, l, :], bins=bins)
            layer_in_obj_axes.append(layer_and_obj_axis)
        object_axes.append(layer_in_obj_axes)

    def update_hist(num):
        for obj in range(n_objects):
            for l in range(n_layer):
                object_axes[obj][l].cla()
                object_axes[obj][l].hist(data[num, obj, l, :], bins=bins)

    ani = animation.FuncAnimation(fig, update_hist, n_epochs)
    return ani


