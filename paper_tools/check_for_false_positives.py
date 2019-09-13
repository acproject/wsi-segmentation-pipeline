'''
some of wsi have no
tumor bed in them.
finding them accurately (so
pathologist don't waste time
by looking at them)
is significant clinically.
'''

from PIL import Image
import cv2
import numpy as np
import openslide
from myargs import args
import os
import glob
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

cancer_thresh = 0.0  # only accept images > this percentage with tumor in them as cancerous

args.val_save_pth = '/home/ozan/remoteDir/Tumor Bed Detection Results/Cellularity_ozan'
args.val_save_pth = '/home/ozan/remoteDir/Tumor Bed Detection Results/Ynet_segmentation_ozan'
args.val_save_pth = '/home/ozan/remoteDir/Tumor Bed Detection Results/Lowres_segmentation_ozan'
args.raw_val_pth = '/home/ozan/remoteDir/'


annotations_list = glob.glob('{}/**/sedeen-Sherine**/*.xml'.format(args.raw_val_pth))
annotations_filename_list = [int(os.path.basename(p).replace('.session.xml', '')) for p in annotations_list]

svs_list = glob.glob('{}/Case*/*.svs'.format(args.raw_val_pth))
svs_filename_list = [(int(os.path.basename(p).replace('.svs', '')), p) for p in svs_list]
svs_ids = [p[0] for p in svs_filename_list]

benigns = [101332, 101333, 101358, 101359, 101361,
101362, 101363, 101364,101366,101372,
101376,101381,101382,101488,101492,101497,
101498,101510,99189,99190,99191,99192,
99204,99205,99206,99207,99916]

preds, gts = [], []

for sj, (svs, svs_path) in tqdm(enumerate(svs_filename_list)):

	cancer_gt = int(svs in annotations_filename_list and int(svs) not in benigns)

	image_id = svs

	scan = openslide.OpenSlide(svs_path)
	if scan.level_count < 3:
		continue

	heatmap_path = glob.glob('{}/**/*{}*heatmap*'.format(args.val_save_pth, image_id))
	heatmap_path = heatmap_path[0]

	im = Image.open(heatmap_path).convert('L')
	x, y = im.size

	heatmap = np.array(im)

	im = np.uint8(heatmap >= 0.99 * 255)  # heatmap is uint8, so 254/255 is about 0.99

	im = cv2.morphologyEx(
		im,
		cv2.MORPH_OPEN,
		kernel=np.ones((50, 50))
	)

	cancer_pred = int(np.float(np.count_nonzero(im))/im.size > cancer_thresh)

	# print('# {}: {}'.format(sj, (cancer_gt, cancer_pred)))

	gts.append(cancer_gt)
	preds.append(cancer_pred)


gts = np.array(gts)
preds = np.array(preds)

print(
	'acc. {:.2f}, '
	'f1 {:.2f}, '
	'prc {:.2f}, '
	'rec {:.2f}, '
	'auc {:.2f}, '
	'cfs {} '.format(
		np.mean(gts == preds),
		f1_score(gts, preds),
		precision_score(gts, preds, average='micro'),
		recall_score(gts, preds, average='micro'),
		roc_auc_score(gts, preds),
		confusion_matrix(gts, preds),
))
