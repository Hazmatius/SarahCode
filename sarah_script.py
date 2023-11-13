import subprocess, os, cv2, pickle, time, datetime
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib.patches import Ellipse
import random


with open('/home/alex/Documents/GitHub/cnru_rotation/xterm_rgb.pkl', 'rb') as f:
	xterm_rgb = pickle.load(f)

rgb_codes = list(xterm_rgb.keys())

color_codes = {
	'r': (255, 0, 0),
	'g': (0, 255, 0),
	'b': (0, 0, 255),
	'c': (0, 255, 255),
	'y': (255, 255, 0),
	'm': (255, 0, 255),
	'w': (255, 255, 255),
	'o': (255, 135, 0),
	'k': (0, 0, 0)
}


def euc(tuple1, tuple2):
	return np.sqrt(np.sum((np.array(tuple1) - np.array(tuple2))**2))


def get_min_verbose(rgb, rgb_codes):
	dists = [euc(rgb, rgb2) for rgb2 in rgb_codes]
	return rgb_codes[np.argmin(dists)], np.min(dists)


def get_min(rgb, rgb_codes):
	dists = [euc(rgb, rgb2) for rgb2 in rgb_codes]
	return rgb_codes[np.argmin(dists)]


def rgb_to_xterm_verbose(rgb):
	nearest_rgb, min_dist = get_min_verbose(rgb, rgb_codes)
	return xterm_rgb[nearest_rgb], nearest_rgb, min_dist


def rgb_to_xterm(rgb):
	return xterm_rgb[get_min(rgb, rgb_codes)]


def color_string(color, string):
	return (f'\033[38;5;{color}m{string}\033[0;0m')


def rgb_string(rgb, string):
	return color_string(rgb_to_xterm(rgb), string)


def cstring(c, string):
	return rgb_string(color_codes[c], string)


def extract_paths(folder):
	paths = list()
	items = os.listdir(folder)
	for item in items:
		item_path = os.path.join(folder, item)
		if item.endswith('.tif'):
			paths.append(item_path)
		elif os.path.isdir(item_path):
			paths.extend(extract_paths(item_path))
	return paths


def get_ellipse_equation(f1, f2, a, b, phi):
	c_x, c_y = (f1[0] + f2[0]) / 2, (f1[1] + f2[1]) / 2
	# phi = np.arctan2(f2[0] - f1[0], f2[1] - f1[1])
	
	def func(theta):
		x = np.cos(phi) * b * np.cos(theta) - np.sin(phi) * a * np.sin(theta) + c_x
		y = np.sin(phi) * b * np.cos(theta) + np.cos(phi) * a * np.sin(theta) + c_y
		return x, y
	
	return func


def make_submask(mask):
	min_i, max_i = np.min(mask[0]), np.max(mask[0])
	min_j, max_j = np.min(mask[1]), np.max(mask[1])
	bin_img = np.zeros((max_i - min_i + 3, max_j - min_j + 3), dtype=np.uint8)
	bin_img[mask[0] - min_i + 1, mask[1] - min_j + 1] = 255
	return bin_img


def fit_ellipse_to_mask(binary_mask):
	# plt.figure()
	# plt.imshow(binary_mask)
	# plt.title('contours...')
	# plt.show()
	contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# print(len(contours))
	if len(contours) > 0:
		largest_contour = max(contours, key=cv2.contourArea)
		# print('\t{}'.format(len(largest_contour)))
		# print(largest_contour)
		ellipse = cv2.fitEllipse(largest_contour)
		return ellipse
	return None


def process_masks(mask_path):
	img = np.asarray(Image.open(mask_path))
	idxs = list(np.unique(img))
	idxs.remove(0)
	kernel = np.ones((3, 3), np.uint8)
	masks = dict()
	for idx in idxs:
		try:
			idxs = np.where(img == idx)
			submask = make_submask(idxs)
			# print(np.max(submask))
			boundary_mask = submask * (255 - cv2.erode(submask, kernel, iterations=1))
			boundary_idxs = np.where(boundary_mask == 1)
			i, j = boundary_idxs
			mean_i, mean_j = np.mean(i), np.mean(j)
			radii = np.sqrt((i - mean_i) ** 2 + (j - mean_j) ** 2)
			theta = np.arctan2(i - mean_i, j - mean_j)
			radii_mean = np.mean(radii)
			radii_stddev = np.std(radii)
			radii_cov = radii_stddev / radii_mean
			# print(boundary_idxs)
			# plt.imshow(submask)
			ellipse_params = fit_ellipse_to_mask(submask)
			center, axes, angle = ellipse_params
			major_axis, minor_axis = axes
			min_diameter = np.minimum(major_axis, minor_axis)
			max_diameter = np.maximum(major_axis, minor_axis)
			e_score = max_diameter / min_diameter - 1
			area = float(len(idxs[0]))
			avg_diameter = 2 * np.sqrt(area / np.pi)
			
			a = max_diameter / 2
			b = min_diameter / 2
			c = np.sqrt(a ** 2 - b ** 2)
			cx, cy = center
			rangle = np.radians(angle + 90)
			f1 = (cx + c * np.cos(rangle), cy + c * np.sin(rangle))
			f2 = (cx - c * np.cos(rangle), cy - c * np.sin(rangle))
			
			d1 = np.sqrt((i - f1[1]) ** 2 + (j - f1[0]) ** 2)
			d2 = np.sqrt((i - f2[1]) ** 2 + (j - f2[0]) ** 2)
			el_radii = d1 + d2
			ridxs = np.argsort(theta)
			
			# plt.plot(el_radii[ridxs])
			# plt.show()
			el_radii_stddev = np.sqrt(np.mean((el_radii - max_diameter) ** 2))
			el_radii_cov = el_radii_stddev / max_diameter * 100
			
			ellipse_func = get_ellipse_equation(f1, f2, min_diameter/2, max_diameter/2, np.radians(angle + 90))

			theta = np.arctan2(j - cx, i - cy)
			order = np.argsort(theta)
			theta = theta[order]
			# print(order)
			j, i = j[order], i[order]
			# theta = np.arctan2(j - cx, i - cy)
			_radii = np.sqrt((j - cx) ** 2 + (i - cy) ** 2)
			ex, ey = ellipse_func(theta + np.radians(angle))
			radii = np.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
			badness = np.mean((_radii - radii) ** 2) / (np.sqrt(min_diameter/2 * max_diameter/2))
			
			mask = {
				'idxs': idxs,
				'e_score': e_score,
				'area': area,
				'max_diameter': max_diameter,
				'min_diameter': min_diameter,
				'avg_diameter': avg_diameter,
				'img.shape': img.shape,
				'radii_cov': radii_cov,
				'el_radii_cov': el_radii_cov,
				'badness': badness,
				'ellipse': {'f1': f1, 'f2': f2, 'a': max_diameter/2, 'b': min_diameter/2, 'phi': np.radians(angle + 90)}
			}
			masks[idx] = mask
			
			# fig, axs = plt.subplots(1, 2, figsize=(12, 5))
			# axs[0].imshow(submask)
			# plt.title('Badness: {}'.format(mask['badness']))
			# ellipse = Ellipse(xy=center, width=major_axis, height=minor_axis, angle=angle, edgecolor='red', facecolor='none')
			# axs[0].scatter(f1[0], f1[1], color='red')
			# axs[0].scatter(f2[0], f2[1], color='red')
			# axs[0].plot(ex, ey, color='red')
			# axs[0].add_patch(ellipse)
			# axs[1].plot(theta, _radii)
			# axs[1].plot(theta, radii)
			# plt.axis('off')
			# plt.show()
		except Exception as e:
			print(e)
	return masks

# mask_folder = '/home/alex/Sarah_data/masks/Trial_2/Control2_Div14/10k_22'
# items = os.listdir(mask_folder)
# for item in items:
# 	if item.endswith('.png'):
# 		# plt.figure()
# 		masks = process_masks(os.path.join(mask_folder, item))
# 		for idx_1, mask_1 in masks.items():
# 			for idx_2, mask_2 in masks.items():
# 				if idx_1 != idx_2:
# 					p1, p2 = compare_masks(mask_1, mask_2)
# 					print(p1, p2)
# 		# plt.imshow(img)
# plt.show()
#
# exit()


# path = '/home/alex/Sarah_data/masks/Trial_2/Control2_Div14/10k_22/10k_22_cp_masks.png'
# img = np.asarray(Image.open(path))
# plt.imshow(img != 0)
# plt.show()
# exit()

cmd = 'python -m cellpose --image_path {image_path} --pretrained_model cyto2 --use_gpu --chan 0 --savedir {output_directory} --save_png --diameter {diameter}'

main_folder = '/home/alex/Sarah_data'
input_folder = os.path.join(main_folder, 'cleaned_2')
output_folder = os.path.join(main_folder, 'masks_folder')

tif_paths = extract_paths(input_folder)
# tif_paths = [path for path in tif_paths if 'Trial_1' in path]
# tif_paths = [path for path in tif_paths if 'Control1' in path]
# tif_paths = [path for path in tif_paths if '10k_01' in path]
# random.shuffle(tif_paths)

# for path in tif_paths:
# 	if 'Trial_1' in path and 'Control2' in path:
# 		print(path)
#
# exit()

t = time.time()
for i, path in enumerate(tif_paths):
	output_path = path.replace(input_folder, output_folder).replace('.tif', '')
	# print(output_path)
	name = os.path.basename(output_path)
	base_filepath = os.path.join(output_path, name)
	os.makedirs(output_path, exist_ok=True)
	# print(base_filepath + '[masks].pkl')
	# exit()
	if not os.path.exists(base_filepath + '[masks].pkl'):
		print(path)
		masks = dict()
		for diameter in [100, 200, 300, 400, 500]:
			subprocess.run(cmd.format(image_path=path, output_directory=output_path, diameter=diameter), shell=True)
			png_filepath = base_filepath + '_cp_masks.png'
			if os.path.exists(png_filepath):
				masks[diameter] = process_masks(png_filepath)
				os.rename(png_filepath, base_filepath + '[{}].png'.format(diameter))
			else:
				masks[diameter] = None
		with open(base_filepath + '[masks].pkl', 'wb') as f:
			pickle.dump(masks, f)
	ETA = (len(tif_paths) / (i+1) - 1) * (time.time() - t)
	print(cstring('g', datetime.timedelta(seconds=ETA)))

# for path in tif_paths:
# 	print(path.replace(input_folder, output_folder))