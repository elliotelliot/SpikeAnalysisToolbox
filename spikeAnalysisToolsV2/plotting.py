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



