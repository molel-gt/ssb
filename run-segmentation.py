#!/usr/bin/env python3
import os
import time

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

import segmentation

from PIL import Image 
from scipy import stats
from sklearn.metrics import confusion_matrix


def resample(img_id, output_dir):
    phases = plt.imread(os.path.join("segmented", str(img_id).zfill(3) + '.tif'))
    with open(os.path.join(output_dir, 'clusters', str(img_id).zfill(3)), 'rb') as fp:
        clusters = pickle.load(fp)
    new_img = phases.copy()
    for v in np.unique(clusters):
        if v < 0:
            continue
        coords = np.where(np.isclose(clusters, v))
        p = phases[coords]
        mode = stats.mode(p)
        count = coords[0].shape[0]
        print(v, mode.mode[0], mode.count[0]/count)#, np.mean(p), np.std(p))
        if mode.count[0]/count > 0.7:
            new_img[coords] = mode.mode[0]
    p0 = np.where(np.isclose(new_img, 0))
    p1 = np.where(np.isclose(new_img, 1))
    p2 = np.where(np.isclose(new_img, 2))
    new_img[p1] = 128
    new_img[p2] = 255
    img_2 = Image.fromarray(new_img)
    img_2.save(f'segmented/resampled/{str(img_id).zfill(3)}.tif', format='TIFF')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument('--output_dir', help='working directory for output, e.g. YYYY-mm-dd', default='2023-03-05')

    args = parser.parse_args()
    start = time.time()

    # segmentation from training data
    # ob = segmentation.StackSeg(training_images=np.linspace(0, 200, num=41), output_dir=args.output_dir)
    # ob.build_features_matrix()
    # ob.train()
    # ob.test()
    # cf_matrix = confusion_matrix(ob.y_test, ob.y_test_pred, normalize='all')
    # sns.heatmap(cf_matrix, annot=True)
    # ob.create_output()

    # resample
    for i in range(0, 203, 5):
        resample(i, args.output_dir)
    
    stop = time.time()
    print("Took: %s minutes" % str(int((stop - start)/60)))
    # plt.show()