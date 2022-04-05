#! /usr/bin/env python3

import os

import argparse
import numpy as np

from collections import OrderedDict
from PIL import Image

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create phases files from segmented image")
    parser.add_argument('--phases', help='phase_name1=image_value1,phase_name2=image_value2,...',
                        required=True)
    parser.add_argument("--img_sub_dir", required=True)
    args = parser.parse_args()
    phases = OrderedDict([(phase.split("=")[0], phase.split("=")[1]) for phase in args.phases.split(",")])
    image_files = sorted([os.path.join(os.path.abspath(os.path.dirname(__file__)), args.img_sub_dir, f) for f in os.listdir(args.img_sub_dir)
                   if f.endswith(".tif")])
    for img_file in image_files:
        file_number = int(img_file.split("/")[-1].split(".")[0])
        img = np.array(Image.open(img_file), dtype=np.uint8)
        for phase_name, phase_value in phases.items():
            phase_img = np.zeros(img.shape, dtype=np.uint8)
            phase_img[np.where(img == int(phase_value))] = 255
            phase_img = Image.fromarray(phase_img)
            phase_file_name = os.path.join(os.path.dirname(img_file), phase_name, f"SegIm{file_number}.bmp")
            utils.make_dir_if_missing(os.path.dirname(phase_file_name))
            phase_img.save(phase_file_name, format='bmp')
    