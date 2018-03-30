import numpy as np
import scipy.signal as sp

class GaborVisualiser:
    """Class to visualize the gabor filters. Note that the filter parameters are currently hardcoded"""

    psi = [0, np.pi]
    scales = [2]
    orient = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    bw = 1.5
    gamma = 0.5
    #         all_filter_options = dict(scales = [2], orientations = [0, 45, 90, 135], psi=phases = [0, 180])
    def __init__(self):
        self.make_all_gabor()

    def make_all_gabor(self):
        total_number_of_wavelengths = len(self.scales)
        total_number_of_phases = len(self.psi)

        self.filter = np.zeros((8, 13, 13))

        for phase_id, phase in enumerate(self.psi):
            for orient_id, orientation in enumerate(self.orient):
                for scale_id, scale in enumerate(self.scales):
                    gabor_id = orient_id * (total_number_of_wavelengths * total_number_of_phases) + scale_id * total_number_of_phases + phase_id

                    current_filter = self.gabor_fn(self.bw, self.gamma, phase, scale, orientation)
                    print(current_filter.shape)
                    self.filter[gabor_id, :, :] = current_filter

    def visualize_image(self, filtered_img, abs_values=True):
        """
        Take the image and try to produce an unfiltered version
        :param filtered_img: image of shape (n_layers, dim, dim), n_layers is the number of gabor filters in the input (each filter has its own input layer)
        :return: image of shape (dim, dim) with 'unfiltered' values
        """
        n_layers = filtered_img.shape[0]
        assert(n_layers == self.filter.shape[0])
        result = np.zeros_like(filtered_img)

        for i in range(n_layers):
            current_filter = self.filter[i, ::-1, ::-1] # invert the order since we are doing transposed convolution

            if abs_values:
                current_filter = np.abs(current_filter)

            result[i, :, :] = sp.convolve2d(filtered_img[i, :, :], current_filter, mode="same")

        return np.sum(result, 0)





    @staticmethod
    def gabor_fn(bw, gamma, psi, lamb, theta):
        # (bw,gamma,psi(p),scale(s),orient(o)))

        sigma = lamb / np.pi * np.sqrt(np.log(2) / 2) * (2 ** bw + 1) / (2 ** bw - 1)
        sigma_x = sigma
        sigma_y = sigma / gamma

        sz = np.fix(8 * max(sigma_y, sigma_x))
        if np.mod(sz, 2) == 0:
            sz = sz + 1

        # % alternatively, use
        # a
        # fixed
        # size
        # % sz = 60;

        x, y = np.meshgrid(np.arange(-np.fix(sz / 2), np.fix(sz / 2)+1), np.arange(np.fix(sz / 2), np.fix(-sz / 2)-1, -1))
        # % x(right +)
        # % y(up +)

        # % Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gb = np.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
            2 * np.pi / lamb * x_theta + psi)

        gb = gb - np.mean(gb)
        return gb


