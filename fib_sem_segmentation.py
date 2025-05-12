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

fig, ax = plt.subplots()
Path = mpath.Path

polygons = []

def plot_bezier_curve(nodes):
    polygons.append(nodes)

img_ids = list(range(1, 203))
img_id = 1
image = plt.imread(os.path.join(input_dir, f"{str(img_id).zfill(3)}.tif"))
output_json = os.path.join(output_dir, f"{str(img_id).zfill(3)}.json")

ax_file = fig.add_axes([0.5, 0.025, 0.1, 0.03])

text_box = TextBox(ax_file, "Image Id", textalignment="center")

def write_to_file(event):
    print(polygons)
    output_json = os.path.join(output_dir, f"{str(img_id).zfill(3)}.json")
    with open(output_json, "w", encoding='utf-8') as f:
        json.dump(polygons, f, ensure_ascii=False, indent=4)
    polygons.clear()


def update(val):
    img_id = int(val)
    image = plt.imread(os.path.join(input_dir, f"{str(img_id).zfill(3)}.tif"))
    ax.imshow(image, "gray")
    fig.canvas.draw_idle()
    polygons.clear()

text_box.on_submit(update)
text_box.set_val("1")

ax_save = fig.add_axes([0.8, 0.025, 0.1, 0.04])
save_button = Button(ax_save, 'Save', hovercolor='0.975')
save_button.on_clicked(write_to_file)

if __name__ == "__main__":
    ax.imshow(image, "gray")
    selector = PolygonSelector(ax, onselect=plot_bezier_curve)
    plt.show()
