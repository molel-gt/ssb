#!/usr/bin/env python3
import time

import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import segmentation

from sklearn.metrics import confusion_matrix
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument('--output_dir', help='working directory for output, e.g. YYYY-mm-dd', default='2023-03-05')

    args = parser.parse_args()
    start = time.time()
    ob = segmentation.StackSeg(training_images=np.linspace(0, 200, num=41), output_dir=args.output_dir)
    # ob.build_features_matrix()
    # ob.train()
    # ob.test()
    # cf_matrix = confusion_matrix(ob.y_test, ob.y_test_pred, normalize='all')
    # sns.heatmap(cf_matrix, annot=True)
    ob.create_output()
    stop = time.time()
    print("Took: %s minutes" % str(int((stop - start)/60)))
    # plt.show()