import numpy as np
import os, tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter as gf


def normalize(v):
	min_v, max_v = np.min(v), np.max(v)
	return (v - min_v) / (max_v - min_v)


def process_tif(filepath, new_filepath):
	if not os.path.exists(new_filepath):
		pilimg = Image.open(filepath)
		img = np.asarray(pilimg, dtype=float)
		nimg = img / (gf(img, 100) + 1)
		nimg = gf(nimg, 5, mode='mirror')
		nimg = normalize(nimg) * 255
		nimg = nimg.astype(np.int16)
		tifffile.imsave(new_filepath, nimg)
	# simg = Image.fromarray((nimg * 255).astype(np.int8))
	# simg.save(new_filepath)


def process_folder(input_folder, output_folder, subpath):
	items = os.listdir(os.path.join(input_folder, subpath))
	for item in items:
		if item.endswith('.tif'):
			os.makedirs(os.path.join(output_folder, subpath), exist_ok=True)
			filepath = os.path.join(input_folder, subpath, item)
			new_filepath = os.path.join(output_folder, subpath, item)
			process_tif(filepath, new_filepath)
		elif os.path.isdir(os.path.join(input_folder, subpath, item)):
			process_folder(input_folder, output_folder, os.path.join(subpath, item))


main_folder = '/home/alex/Sarah_data/EM_Data_Mod'
output_folder = '/home/alex/Sarah_data/cleaned_2'

process_folder(main_folder, output_folder, '')