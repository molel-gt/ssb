#!/usr/bin/env python3
import cv2 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage as ski
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon

import grapher, plot_opts, utils
plt.rcParams.update(plot_opts.params)


def get_curves(img_file_path, min_val):
    data = np.asarray(Image.open(img_file_path)).copy()
    img = data[:, :, 0]
    nx, ny = img.shape
    # pad boundaries
    img[:, 0] = 255
    img[:, -1] = 255
    img[0, :] = 255
    img[-1, :] = 255
    curves = np.where(np.greater_equal(img, min_val))

    return curves, nx, ny


if __name__ == '__main__':
    curves, nx, ny = get_curves("050.tif", 255)
    img_1 = np.zeros((nx, ny))
    img_1[curves] = 255
    img_2 = Image.fromarray(img_1)
    img_2.save("loops.tif")

    # Find contours at a constant value of 0.8
    contours = ski.measure.find_contours(img_1, 0.8)
    print(f"There are {len(contours)} loops")

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(img_1, cmap=plt.cm.gray)
    count = 0
    for contour in contours:
        if contour.shape[0] <= 20:
              continue
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        count += 1
    print(f"There are {count} big loops")

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


    # points = {}
    # for (x, y) in curves:
    #     idx = ny * x + y
    #     points[idx] = (x, y) #=  idx
    # print(f"{len(points.keys())}")
    # fig, ax = plt.subplots()
    # ax.imshow(img, "gray")
    # # ax.scatter(curves[:, 0], curves[:, 1], s=0.1)
    # ax.set_box_aspect(1)
    # plt.tight_layout()
    # plt.show()
    # graph = grapher.PixelGraph(points)
    # graph.build_graph()
    # graph.get_graph_pieces()
    # print(f"Graph has {graph.n_pieces} pieces")
    # print(len(graph.pieces[-2]))
    # for p in graph.pieces[:50]:
    #     print(len(p))
