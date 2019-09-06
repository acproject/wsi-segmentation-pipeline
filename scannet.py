# import the necessary packages
from skimage.segmentation import slic, quickshift, felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import openslide
from utils import regiontools
from mahotas import bwperim
from utils import preprocessing
from utils.dataset_hr import GenerateIterator_eval
from myargs import args
import utils.networks as networktools
from models import optimizers
import resnets_shift
import os
import torch
import cv2
from scipy.ndimage.morphology import binary_fill_holes
import utils.dataset_hr as dhr
scan_level = 2

us_kmeans = 4
us = 4

# load the image and convert it to a floating point data type
svspth = '/home/ozan/ICIAR2018_BACH_Challenge/WSI/A01.svs'
scan = openslide.OpenSlide(svspth)
x, y = scan.level_dimensions[-1]

wsi = scan.read_region((0, 0), 2, scan.level_dimensions[-1]).convert('RGB')

wsi = wsi.resize((x//us, y//us))
wsi = np.asarray(wsi)

'generate color thresholded wsi mask'
wsi_mask = preprocessing.find_nuclei(wsi)

mask = Image.open('/home/ozan/ICIAR2018_BACH_Challenge/WSI/gt_thumbnails/{}'.format(os.path.basename(svspth).replace('.svs', '.png'))).convert('L')
mask = mask.resize(scan.level_dimensions[-1])
mask = np.asarray(mask)

kernel_size = 50
'''
kernel = np.ones((kernel_size, kernel_size), np.uint8)
labels = binary_fill_holes(mask > 0)
labels = cv2.morphologyEx(labels.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
labels = binary_fill_holes(labels)
labels = cv2.morphologyEx(labels.astype(np.uint8), cv2.MORPH_OPEN, kernel)
labels = binary_fill_holes(labels)
'''

_, labels, _, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))

image = Image.fromarray(wsi)
image = image.resize((x, y))
image = np.asarray(image)

wsi_mask = Image.fromarray(wsi_mask)
wsi_mask = wsi_mask.resize((x, y))
wsi_mask = np.asarray(wsi_mask)

metadata = {}

patch_id = 0
for tile_id in range(labels.max()):

    label_patch = labels == tile_id

    area = np.count_nonzero(label_patch)
    num_clusters = 2 + int(area / (0.01 * labels.size))

    n, center_pts, out_image, foreground_indices = regiontools.get_key_points(label_patch, us_kmeans, num_clusters, num_clusters)

    'get width & height'
    indices = np.where(label_patch)
    h = 1 + indices[0].max() - indices[0].min()
    w = 1 + indices[1].max() - indices[1].min()

    if n is not None and (w * h) / labels.size <= 0.05:
        perim = bwperim(label_patch)
        perim_coords = np.transpose(np.where(perim))[:, ::-1]  # (x,y) pairs
        skip = np.maximum(2, perim_coords.shape[0] // dhr.HR_NUM_PERIM_SAMPLES)
        perim_coords = perim_coords[::skip, :]

        metadata[patch_id] = {
            'cnt_xy': center_pts,
            'perim_xy': perim_coords,
            'wsipath': svspth,
            'scan_level': scan_level,
            'foreground_indices': foreground_indices,
            'tile_id': patch_id,
        }
        patch_id = patch_id + 1

    elif n is not None:

        for r_id in range(1, n+1):

            sub_patch = (out_image == r_id)

            min_center_large = int((w * h) / (wsi_mask.size * 0.005))
            # when the region is large, split more to achieve uniform regions
            min_center_large = num_clusters

            sub_n, sub_center_pts, _, sub_foreground_indices = regiontools.get_key_points(sub_patch, us_kmeans, min_center_large)

            if sub_n is None or (
                    tile_id == 0 and
                    np.count_nonzero(wsi_mask[sub_foreground_indices]) / (sub_foreground_indices[0].shape[0]) < 0.5):
                continue

            sub_perim_coords = np.transpose(np.where(bwperim(sub_patch)))[:, ::-1]  # (x,y) pairs
            skip = np.maximum(2, sub_perim_coords.shape[0]//dhr.HR_NUM_PERIM_SAMPLES)
            sub_perim_coords = sub_perim_coords[::skip, :]

            metadata[patch_id] = {
                'cnt_xy': sub_center_pts,
                'perim_xy': sub_perim_coords,
                'wsipath': svspth,
                'scan_level': scan_level,
                'foreground_indices': sub_foreground_indices,
                'tile_id': patch_id,
            }
            patch_id = patch_id + 1

'''
evaluation stage
'''

# load model
model = resnets_shift.resnet18(True).cuda()
optimizer = optimizers.optimfn(args.optim, model)  # unused
model, _, _ = networktools.continue_train(model, optimizer,
                                                            args.eval_model_pth, True)
model.eval()

# generate dataset from points
iterator_val = GenerateIterator_eval(metadata)
# pass through dataset
pred_mask = np.zeros_like(labels)

with torch.no_grad():
    for batch_it, (images, tile_ids) in enumerate(iterator_val):
        images = images.cuda()
        pred_ensemble = model(images)
        pred_ensemble = torch.softmax(pred_ensemble, 0)
        for cj in range(args.num_classes):
            pred_ensemble[pred_ensemble[:, cj] < args.class_probs[cj], cj] = 0

        pred_ensemble = torch.argmax(pred_ensemble, 1).cpu().numpy()
        for tj, tile_id in enumerate(tile_ids.numpy()):
            pred_mask[metadata[tile_id]['foreground_indices']] = pred_ensemble[tj]

pred_mask_rgb = np.eye(4)[pred_mask][..., 1:]
pred_mask_rgb = Image.fromarray(pred_mask_rgb.astype(np.uint8) * 255)
pred_mask_rgb = pred_mask_rgb.resize((x//us, y//us))
pred_mask_rgb.save('scannet_out_mask.png')

slic_out = mark_boundaries(image, labels, color=(0, 0, 0))
Image.fromarray((255*slic_out).astype(np.uint8)).save('scannet_out.png')

