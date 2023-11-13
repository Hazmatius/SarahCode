import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import os
from PIL import Image

from scipy.ndimage import gaussian_filter as gf
from scipy.ndimage import sobel, median_filter
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
from skimage.filters import rank


def gaussdiff(img, σ_1, σ_2):
	return gf(img, σ_1, mode='mirror') - gf(img, σ_2, mode='mirror')


def prepare_nd_gaussdiff(img, scales, factor):
	gaussdiffs = [normalize(np.expand_dims(gaussdiff(img, scale, scale*factor), -1)) for scale in scales]
	return np.concatenate(gaussdiffs, axis=-1)


def normalize(v):
	min_v, max_v = np.min(v), np.max(v)
	return (v - min_v) / (max_v - min_v)


def gradient(v):
	sobel_h = sobel(v, 0)
	sobel_v = sobel(v, 1)
	magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
	magnitude -= np.min(magnitude)
	magnitude /= np.max(magnitude)
	magnitude = np.expand_dims(magnitude, -1)
	angle = np.expand_dims(np.arctan2(sobel_h, sobel_v) + np.pi, -1)
	return magnitude, angle, sobel_h, sobel_v


def hsv_from_angle(angle, magnitude):
	h = angle / (2 * np.pi)
	s = np.ones_like(h)
	if magnitude is None:
		magnitude = np.ones_like(h)
	return hsv_to_rgb(np.concatenate([h, s, normalize(magnitude)], axis=-1))


def get_segmentation(img):
	ls = morphological_geodesic_active_contour(img, num_iter=5, init_level_set='checkerboard', smoothing=1, balloon=-1, threshold=0.01)
	return ls
	

main_dir = '/home/alex/Sarah_data/EV_Data_EM'
trial = 'Trial_1'
subdir = 'Control1_Div7'
filename = '10k_02.tif'

filepath = os.path.join(main_dir, trial, subdir, filename)

pilimg = Image.open(filepath)
img = np.asarray(pilimg, dtype=float)
crop = 120
img = img[0:img.shape[0]-crop, :]
ax1 = plt.subplot(1, 3, 1)
nimg = img / (gf(img, 100) + 1)
plt.imshow(nimg)
# gdiff_nimg = gaussdiff(nimg, 5, 50) ** 2
# gdiff_nimg = gf(nimg, 10, mode='mirror')
# mag, angle, _, _ = gradient(gdiff_nimg)
# mag[mag < 0.05] = 0
# gdiff_nimg = gf(nimg, 40, mode='mirror')

# mag /= gf(mag, 5, mode='mirror')
# mag[mag < 0.5] = 0

ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
# plt.imshow(prepare_nd_gaussdiff(nimg, [1, 5, 15], 5))
# plt.imshow(hsv_from_angle(angle, mag))
# gimage = inverse_gaussian_gradient(nimg)
# ls = get_segmentation(gimage)
# plt.imshow(ls)
mimg = median_filter(gf(nimg, 2, mode='mirror'), size=(11, 11), mode='mirror')
gdiff_nimg = gaussdiff(mimg, 20, 40)
mag, angle, _, _ = gradient(gdiff_nimg)
# mag /= gf(mag, 5, mode='mirror')
plt.imshow(np.log(np.abs(gdiff_nimg) * mag[:, :, 0]))

# gradient = sobel(gdiff_nimg)
segments_watershed = watershed(-gdiff_nimg, markers=None, compactness=0.1)

ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
plt.imshow(mark_boundaries(normalize(nimg), segments_watershed))
plt.title('watershed')

plt.show()

# ndimg = prepare_nd_gaussdiff(nimg, [5, 15, 45], 5)
# plt.imshow(ndimg)

# plt.imshow(gdiff_nimg)

# h = np.expand_dims(angle / (2 * np.pi), -1)
# hsv = np.concatenate([h, np.ones_like(h), magnitude], axis=-1)
# rgb = hsv_to_rgb(hsv)



# ax4 = plt.subplot(2, 2, 1, sharex=ax1, sharey=ax1)
# plt.imshow(mark_boundaries(gdiff_nimg, segments_fz))
# plt.title('felzenszwalb')
#
# ax5 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
# plt.imshow(mark_boundaries(gdiff_nimg, segments_slic))
# plt.title('slic')
#
# ax6 = plt.subplot(2, 2, 3, sharex=ax1, sharey=ax1)
# plt.imshow(mark_boundaries(gdiff_nimg, segments_quick))
# plt.title('quick')

# ax7 = plt.subplot(2, 2, 4, sharex=ax1, sharey=ax1)