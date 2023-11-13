import os, shutil


def extract_paths(folder):
	paths = list()
	items = os.listdir(folder)
	for item in items:
		item_path = os.path.join(folder, item)
		if item.endswith('.pkl'):
			paths.append(item_path)
		elif os.path.isdir(item_path):
			paths.extend(extract_paths(item_path))
	return paths


main_folder = '/home/alex/Sarah_data/good_masks_folder'
pkl_paths = extract_paths(main_folder)

for i, pkl_path in enumerate(pkl_paths):
	old_img_path = pkl_path.replace('.pkl', '.png')
	new_img_path = old_img_path.replace('good_masks_folder', 'masks_images_folder')
	basename = os.path.basename(new_img_path)
	new_folder = new_img_path.replace('/' + basename, '')
	new_folder = new_folder.replace('/' + os.path.basename(new_folder), '')
	new_img_path = os.path.join(new_folder, basename).replace('[masks]', '')
	os.makedirs(new_folder, exist_ok=True)
	shutil.copyfile(old_img_path, new_img_path)
	print('{}/{}'.format(i+1, len(pkl_paths)))