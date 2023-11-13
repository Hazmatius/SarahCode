import os, pickle, cv2, sys, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image, ImageDraw
from matplotlib.patches import Polygon


def get_number_from_path(path):
	basename = os.path.basename(path.replace('[masks].pkl', ''))
	basename = basename.replace('-', '_')
	basename = basename.replace('.', '')
	try:
		return int(basename.split('_')[1])
	except:
		return 0


def extract_paths(folder):
	paths = list()
	items = os.listdir(folder)
	pkl_items = [item for item in items if item.endswith('.pkl')]
	# print(folder)
	# print([int(item.replace('[masks].pkl', '').split('_')[1]) for item in pkl_items])
	# numbers = [int(item.replace('[masks].pkl', '').split('_')[1]) for item in pkl_items]
	# pkl_items = [pkl_items[i] for i in np.argsort(numbers)]
	dir_items = [item for item in items if os.path.isdir(os.path.join(folder, item))]
	pkl_paths = [os.path.join(folder, item) for item in pkl_items]
	dir_paths = [os.path.join(folder, item) for item in dir_items]
	numbers = [get_number_from_path(item) for item in dir_paths]
	dir_paths = [dir_paths[i] for i in np.argsort(numbers)]
	
	paths.extend(pkl_paths)
	for dir_path in dir_paths:
		paths.extend(extract_paths(dir_path))
	return paths


def get_coord_set(mask):
	i, j = list(mask['idxs'][0]), list(mask['idxs'][1])
	return set(zip(i, j))


def compare_masks(mask_1, mask_2):
	mask_1_coords, mask_2_coords = mask_1['coord_set'], mask_2['coord_set']
	overlap_set = mask_1_coords.intersection(mask_2_coords)
	overlap = float(len(overlap_set))
	p1 = overlap / float(len(mask_1['idxs'][0]))
	p2 = overlap / float(len(mask_2['idxs'][0]))
	return p1, p2


def _isgood(mask, badness_thresh):
	if 'Sarah' in mask.keys():
		return mask['Sarah']
	else:
		return mask['badness'] <= badness_thresh


def mask_is_good(mask, level, levels, masks, badness_thresh, overlap_thresh):
	# print(mask['badness'])
	if 'Sarah' in mask.keys():
		return mask['Sarah']
	if mask['badness'] > badness_thresh:
		return False
	else:
		super_levels = [l for l in levels if l < level]
		for l in super_levels:
			for idx in masks[l].keys():
				overmask = masks[l][idx]
				p1, p2 = compare_masks(mask, overmask)
				# if p1 > overlap_thresh and overmask['e_score'] <= circ_thresh and overmask['radii_cov'] <= radii_cov_thresh:
				if p2 > overlap_thresh and _isgood(overmask, badness_thresh):
					return False
		return True


def get_good_masks(masks, badness_thresh, overlap_thresh):
	"""
	For any mask, we retain the mask if:
		1) the mask is sufficiently circular
		2) the mask has less than X% overlap with any other mask which is also sufficiently circular
	"""
	levels = list(masks.keys())
	levels.sort()
	levels = [level for level in levels if masks[level] is not None]
	good_masks = list()
	reasons = dict()
	for level in levels:
		reasons[level] = dict()
		for idx, mask in masks[level].items():
			if mask_is_good(mask, level, levels, masks, badness_thresh, overlap_thresh):
				mask['good'] = True
				good_masks.append(mask)
				reasons[level][idx] = {'good': True}
			else:
				mask['good'] = False
				reasons[level][idx] = {'good': False}
	return good_masks, reasons


# def prepare_image(path, good_masks):
# 	img = np.zeros(good_masks[0]['img.shape'])
# 	for i, mask in enumerate(good_masks):
# 		img[mask['idxs']] += 1
# 	return img


def imgpath_from_pklpath(pklpath):
	img_path = pklpath.replace('.pkl', '')
	img_path = img_path.replace('/' + os.path.basename(img_path), '')
	img_path = img_path.replace('masks_folder', 'cleaned_2') + '.tif'
	return img_path


def pklpath_from_imgpath(imgpath):
	pkl_path = imgpath.replace('.tif', '')
	pkl_path = os.path.join(pkl_path, os.path.basename(pkl_path) + '[masks]')
	pkl_path = pkl_path.replace('cleaned_2', 'masks_folder')
	pkl_path = pkl_path + '.pkl'
	return pkl_path


def get_binarymask(img, mask):
	binary_mask = np.zeros(img.shape[0:2])
	binary_mask[mask['idxs']] = 1
	return binary_mask


def get_patch(kernel, mask):
	min_i, max_i = np.min(mask['idxs'][0]), np.max(mask['idxs'][0])
	min_j, max_j = np.min(mask['idxs'][1]), np.max(mask['idxs'][1])
	bin_img = np.zeros((max_i - min_i + 3, max_j - min_j + 3), dtype=np.uint8)
	bin_img[mask['idxs'][0] - min_i + 1, mask['idxs'][1] - min_j + 1] = 1
	bin_img = bin_img * (1 - cv2.erode(bin_img, kernel, iterations=1))
	boundary_idxs = np.where(bin_img)
	i, j = boundary_idxs
	mean_i, mean_j = np.mean(i), np.mean(j)
	theta = np.arctan2(i - mean_i, j - mean_j)
	order = np.argsort(theta)
	points = list(zip(j[order] + min_j - 1, i[order] + min_i - 1))
	patch = Polygon(points, closed=True)
	patch.set_facecolor('none')
	patch.set_linewidth(2)
	patch.set_edgecolor((1, 1, 1))
	return points, patch


def overlay_masks(ax, mask_list):
	good_patches = dict()
	for mask in mask_list:
		color = hsv_to_rgb((np.random.rand(), 1, 1))
		patch = Polygon(mask['points'], closed=True)
		patch.set_facecolor('none')
		patch.set_linewidth(2)
		patch.set_edgecolor(color)
		good_patches[(mask['level'], mask['idx'])] = patch
		ax.add_patch(patch)
	ax.set_title('Final')
	ax.axis('off')
	return good_patches
	
	
	# image = Image.open(image_path)
	# image_array = np.array(image)
	# r = image_array.copy()
	# g = image_array.copy()
	# b = image_array.copy()
	# kernel = np.ones((7, 7), np.uint8)
	#
	# for i in range(len(mask_list)):
	# 	color = (np.array(hsv_to_rgb((np.random.rand(), 1, 1))) * 255).astype(np.uint8)
	# 	binary_mask = get_binarymask(image_array, mask_list[i])
	# 	binary_mask = binary_mask * (1 - cv2.erode(binary_mask, kernel, iterations=1))
	# 	r[np.where(binary_mask)] = color[0]
	# 	g[np.where(binary_mask)] = color[1]
	# 	b[np.where(binary_mask)] = color[2]
	# image_array = np.stack((r, g, b), axis=-1)
	# image_array = image_array.astype(np.uint8)
	#
	# return image_array


def display_masks(ax, image, masks, reasons, level):
	ax.imshow(image)
	if masks[level] is not None:
		for idx, mask in masks[level].items():
			if reasons[level][idx]['good']:
				color = (0, 1, 0)
			else:
				color = (1, 0, 0)
			masks[level][idx]['patch'].set_edgecolor(color)
			ax.add_patch(masks[level][idx]['patch'])
	ax.axis('off')
	ax.set_title('{}'.format(level))


def add_patches_to_image(image, patches):
	thickness = 10
	pil_image = Image.fromarray(image.astype(np.uint8))
	draw = ImageDraw.Draw(pil_image)
	for patch in patches:
		color = tuple(list((np.array(hsv_to_rgb((np.random.rand(), 1, 1))) * 255).astype(np.uint8)))
		patch_vertices = patch.get_path().vertices
		pillow_vertices = [(int(x), int(y)) for x, y in patch_vertices]
		# draw.polygon(pillow_vertices, outline=color)
		for i in range(len(pillow_vertices) - 1):
			draw.line([pillow_vertices[i], pillow_vertices[i + 1]], fill=color, width=thickness)
		draw.line([pillow_vertices[-1], pillow_vertices[0]], fill=color, width=thickness)

	return pil_image


def click_img(event):
	global good_masks, good_patches, reasons
	for i in range(len(axes)):
		for j in range(len(axes[i])):
			if axes[i][j] == event.inaxes:
				level = ax_to_level[(i, j)]
				mask_dict = masks[level]
				for idx, mask in mask_dict.items():
					mask['patch'].set_edgecolor((1, 1, 0))
					if mask['patch'].contains_point(axes[i][j].transData.transform((event.xdata, event.ydata))):
						if event.button == 1:  # left click
							mask['Sarah'] = True
						elif event.button == 3:  # right click
							mask['Sarah'] = False
				good_masks, reasons = get_good_masks(masks, badness_thresh, overlap_thresh)
				for mask in all_masks:
					if mask['good']:
						mask['patch'].set_edgecolor((0, 1, 0))
					else:
						mask['patch'].set_edgecolor((1, 0, 0))
				for patch in good_patches.values():
					patch.remove()
				good_patches.clear()
				good_patches = overlay_masks(axes[1, 2], good_masks)
				fig.canvas.draw()


def on_key(event):
	global save
	if event.key == 'r':
		save = False


ax_to_level = {
	(0, 0): 100,
	(0, 1): 200,
	(0, 2): 300,
	(1, 0): 400,
	(1, 1): 500,
	(1, 2): 'Final'
}


main_folder = '/home/alex/Sarah_data'
input_folder = os.path.join(main_folder, 'masks_folder')

# circ_thresh = 0.1
overlap_thresh = 0.8
# radii_cov_thresh = 0.1
badness_thresh = 1

# print(len(sys.argv))

if len(sys.argv) == 2:
	img_path = sys.argv[1]
	pklpath = pklpath_from_imgpath(img_path)
	pkl_paths = [pklpath]
else:
	pkl_paths = extract_paths(input_folder)

for i, path in enumerate(pkl_paths):
	save = True
	new_path = path.replace('masks_folder', 'good_masks_folder')
	filename = os.path.basename(new_path)
	new_folder = new_path.replace(filename, '')
	new_path = os.path.join(new_folder, filename)
	if not os.path.exists(new_folder):
		img_path = imgpath_from_pklpath(path)
		print('{}/{} : {}'.format(i, len(pkl_paths), img_path))
		img_gray = np.asarray(Image.open(img_path))
		img_rgb = np.stack((img_gray, img_gray, img_gray), axis=-1)
		with open(path, 'rb') as f:
			masks = pickle.load(f)
		kernel = np.array([[0, 1, 0],
						   [1, 1, 1],
						   [0, 1, 0]], dtype=np.uint8)
		all_masks = list()
		for level in masks.keys():
			if masks[level] is not None:
				for idx in masks[level]:
					masks[level][idx]['points'], masks[level][idx]['patch'] = get_patch(kernel, masks[level][idx])
					masks[level][idx]['coord_set'] = get_coord_set(masks[level][idx])
					masks[level][idx]['level'] = level
					masks[level][idx]['idx'] = idx
					masks[level][idx]['Sarah'] = False
					all_masks.append(masks[level][idx])
		# for level in masks.keys():
		# 	if masks[level] is not None:
		# 		for idx in masks[level]:
		# 			masks[level][idx]['coord_set'] = get_coord_set(masks[level][idx])
		
		# print('{} / {}'.format(i+1, len(pkl_paths)))
		# print(new_folder)
		# print(new_path)
		# exit()
		t = time.time()
		good_masks, reasons = get_good_masks(masks, badness_thresh, overlap_thresh)
		fig, axes = plt.subplots(2, 3, sharex='all', sharey='all', figsize=(20, 12))
		display_masks(axes[0, 0], img_rgb, masks, reasons, 100)
		display_masks(axes[0, 1], img_rgb, masks, reasons, 200)
		display_masks(axes[0, 2], img_rgb, masks, reasons, 300)
		display_masks(axes[1, 0], img_rgb, masks, reasons, 400)
		display_masks(axes[1, 1], img_rgb, masks, reasons, 500)
	
		axes[1, 2].imshow(img_rgb)
		good_patches = overlay_masks(axes[1, 2], good_masks)
		fig.tight_layout()
		fig.canvas.mpl_connect('button_press_event', click_img)
		fig.canvas.mpl_connect('key_press_event', on_key)
		plt.show()
		
		os.makedirs(new_folder, exist_ok=True)
		if save:
			with open(new_path, 'wb') as f:
				pickle.dump(good_masks, f)
			pilimg = add_patches_to_image(img_rgb, [mask['patch'] for mask in good_masks])
			pilimg.save(new_path.replace('.pkl', '.png'))
	
	# exit()