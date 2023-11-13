import os, pickle
import matplotlib.spines
from matplotlib.patches import Polygon


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


def get_info_from_path(path):
	basename = os.path.basename(path).replace('[masks].pkl', '')
	resolution = int(basename.split('_')[0][0:2]) * 1000
	subpath = path.replace('/home/alex/Sarah_data/good_masks_folder/', '').replace(os.path.basename(path), '')[0:-1]
	items = subpath.split('/')
	trial = items[0]
	condition = items[1].split('_')[0]
	day = items[1].split('_')[1]
	sample = basename
	info = {
		'trial': trial,
		'condition': condition,
		'day': day,
		'sample': sample,
		'resolution': resolution
	}
	return info


csv_filepath = '/home/alex/Sarah_data/diameters.csv'
main_folder = '/home/alex/Sarah_data/good_masks_folder'
pkl_paths = extract_paths(main_folder)

# with open(csv_filepath, 'w') as f:
# 	f.write('trial,condition,day,sample,resolution,idx,avg_diameter,min_diameter,max_diameter,e_score,area,radii_cov,el_radii_cov,badness\n')


for i, path in enumerate(pkl_paths):
	info = get_info_from_path(path)
	try:
		with open(path, 'rb') as f:
			masks = pickle.load(f)
		with open(csv_filepath, 'a') as f:
			for mask in masks:
				for key in ['idx', 'avg_diameter', 'min_diameter', 'max_diameter', 'e_score', 'area', 'radii_cov', 'el_radii_cov', 'badness']:
					info[key] = mask[key]
				f.write('{trial},{condition},{day},{sample},{resolution},{idx},{avg_diameter},{min_diameter},{max_diameter},{e_score},{area},{radii_cov},{el_radii_cov},{badness}\n'.format(**info))
	except Exception as e:
		pass
	print('{}/{}'.format(i+1, len(pkl_paths)))