import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage import (
    filters,
    segmentation,
    )


def H_lo(D, D0, n):
    return 1 / (1 + (D/D0) ** (2 * n))


def H_hi(D, D0, n):
    return 1 / (1 + (D0/D) ** (2 * n))


img = plt.imread("unsegmented/000.tif")
F = np.fft.fft2(img)
Fshift = np.fft.fftshift(F)
M, N = img.shape
H = np.zeros((M, N), dtype=np.float32)
n = 1
D0 = 10

for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M/2) ** 2 + (v - N/2) ** 2)
        H[u, v] = H_lo(D, D0, n)

Gshift = Fshift * H
G = np.fft.ifftshift(Gshift)
g = np.abs(np.fft.ifft2(G))

img_meijering = filters.meijering(g)
thresh = filters.threshold_multiotsu(img)
img_edges = np.digitize(img, bins=thresh)

fig, ax = plt.subplots(1, 4)
ax[0].imshow(img, cmap="gray")
ax[0].set_title("Original")
ax[1].imshow(g, cmap="gray")
ax[1].set_title(f"Filtered, cut off freq = {D0}")
ax[2].imshow(img_meijering, cmap="gray")
ax[2].set_title(f"Sobel filter")
# ax[3].hist(g.ravel(), bins=255, density=True)
# ax[3].set_title('Image Histogram')
ax[3].imshow(img_edges, cmap="gray")
ax[3].set_title("Edges")
plt.show()
