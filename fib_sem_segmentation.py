#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, PolygonSelector, Slider, TextBox
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.path as mpath

import plot_opts, utils
plt.rcParams.update(plot_opts.params)


input_dir = "output/segmentation/raw"
output_dir = "output/segmentation/polygons"
utils.make_dir_if_missing(output_dir)


class Segmentation:
    def __init__(self, ax):
        self._ax = ax
        self._img_id = 1
        self._polygons = []
        self._image = np.zeros((500, 500))

    @property
    def ax(self):
        return self._ax

    @property
    def img_id(self):
        return self._img_id

    @property
    def polygons(self):
        return self._polygons

    @property
    def image(self):
        return self._image

    def add_polygons(self, nodes):
        self._polygons.append(nodes)

    def write_to_file(self, event):
        with open(os.path.join(output_dir, f"{str(self.img_id).zfill(3)}.json"), "w", encoding='utf-8') as f:
            json.dump(self.polygons, f, ensure_ascii=False, indent=4)
        self._polygons.clear()

    def update(self, val):
        self._img_id = int(val)
        self._image = plt.imread(os.path.join(input_dir, f"{str(self.img_id).zfill(3)}.tif"))
        ax.imshow(self.image, "gray")
        fig.canvas.draw_idle()
        polygons.clear()


if __name__ == "__main__":
    fig, ax = plt.subplots()
    segmentor = Segmentation(ax)
    ax_txtbx = fig.add_axes([0.5, 0.025, 0.1, 0.03])
    text_box = TextBox(ax_txtbx, "Image Id", textalignment="center")
    text_box.set_val(str(segmentor.img_id))
    text_box.on_submit(segmentor.update)

    ax_save = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    save_button = Button(ax_save, 'Save', hovercolor='0.975')
    save_button.on_clicked(segmentor.write_to_file)
    segmentor.ax.imshow(segmentor.image, "gray")
    selector = PolygonSelector(segmentor.ax, onselect=segmentor.add_polygons)
    plt.show()
