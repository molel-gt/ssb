#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from skimage import feature


# Generate noisy image of a square
image = plt.imread("unsegmented/000.tif") / 255

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(image)
edges2 = feature.canny(image, sigma=3)
edges3 = feature.canny(image, sigma=0.75)

# display results
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 3))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original', fontsize=20)

ax[1].imshow(edges3, cmap='gray')
ax[1].set_title(r'Canny filter, $\sigma=0.75$', fontsize=20)

ax[2].imshow(edges1, cmap='gray')
ax[2].set_title(r'Canny filter, $\sigma=1$', fontsize=20)

ax[3].imshow(edges2, cmap='gray')
ax[3].set_title(r'Canny filter, $\sigma=3$', fontsize=20)

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()