#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import plot_opts, utils
plt.rcParams.update(plot_opts.params)


def get_curves(img_file_path, min_val):
    data = np.asarray(Image.open(img_file_path)).copy()
    img = data[:, :, 0]
    # pad boundaries
    img[:, 0] = 255
    img[:, -1] = 255
    img[0, :] = 255
    img[-1, :] = 255
    curves = np.argwhere(np.greater_equal(img, min_val))

    return curves


if __name__ == '__main__':
    curves = get_curves("050.tif", 163)
    fig, ax = plt.subplots()
    ax.scatter(curves[:, 0], curves[:, 1], )
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()
