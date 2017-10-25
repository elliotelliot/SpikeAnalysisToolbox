import numpy as np
import matplotlib.pyplot as plt
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

    for item_id, item in enumerate(item_labels):
        fig = plt.figure(figsize=(19, 8))
        fig.suptitle("Item: {}".format(item), fontsize=16)

        for layer in range(num_layers):
            subPlotAX = fig.add_subplot(2, num_layers, layer + 1)

            subPlotAX.set_title("Excitatory - Layer " + str(layer))
            subPlotAX.imshow(exc_rates_imgs[item_id, layer, :, :], vmin=vmin, vmax=vmax, cmap=cmap)

            subPlotAXinh = fig.add_subplot(2, num_layers, num_layers + layer + 1)

            subPlotAXinh.set_title("Inhibitory - Layer " + str(layer))
            im = subPlotAXinh.imshow(inh_rates_imgs[item_id, layer, :, :], vmin=vmin, vmax=vmax, cmap=cmap)

        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        fig.colorbar(im, cax=cax)




def plot_information_measure_advancement(before, after, item_label=None):
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

            layerAX.plot(before[i, layer, ::-1], label="before")
            layerAX.plot( after[i, layer, ::-1], label="after")

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

    n_above_threshold = np.count_nonzero( (info >= threshold), axis=3 )

    fig = plt.figure(figsize=(18, 15))
    fig.suptitle("Development of the information", fontsize=16)

    for i, item in enumerate(item_label):

        axAvg  = fig.add_subplot(n_objects, 2, i + 1)
        axAvg.set_title("Average info for Item {}".format(item))


        axN_neurons = fig.add_subplot(n_objects, 2, n_objects + i + 1)
        axN_neurons.set_title("Number of neurons above {}".format(threshold))

        for l in range(n_layer):

            axAvg.plot(avg_info[:, i, l], label= "Layer {}".format(l))
            axN_neurons.plot(n_above_threshold[:, i, l], label="Layer {}".format(l))

        axAvg.legend()
        axN_neurons.legend()