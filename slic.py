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
import torch
import utils.dataset_hr as dhr

scan_level = 2

numSegments = 200
compactness = 20
sigma = 5

us_kmeans = 4
us = 4

# load the image and convert it to a floating point data type
svspth = '/home/ozan/ICIAR2018_BACH_Challenge/WSI/A10.svs'
scan = openslide.OpenSlide(svspth)
x, y = scan.level_dimensions[-1]

wsi = scan.read_region((0, 0), 2, scan.level_dimensions[-1]).convert('RGB')

wsi = wsi.resize((x//us, y//us))
wsi = np.asarray(wsi)

'generate color thresholded wsi mask'
wsi_mask = preprocessing.find_nuclei(wsi)

labels = slic(img_as_float(wsi), n_segments=numSegments, enforce_connectivity=False, sigma=sigma, min_size_factor=0.0, compactness=compactness)

image = Image.fromarray(wsi)
labels = Image.fromarray(labels.astype(np.uint16))

image = image.resize((x, y))
labels = labels.resize((x, y))

image = np.asarray(image)
labels = np.asarray(labels)

metadata = {}

for tile_id in range(labels.max()):

    label_patch = labels == tile_id
    n, center_pts, out_image, foreground_indices = regiontools.get_key_points(label_patch, us_kmeans, dhr.HR_NUM_CNT_SAMPLES, dhr.HR_NUM_CNT_SAMPLES)

    perim_coords = np.zeros([0, 2])
    if dhr.HR_NUM_PERIM_SAMPLES > 0:
        perim = bwperim(label_patch)
        perim_coords = np.transpose(np.where(perim))[:, ::-1]  # (x,y) pairs
        skip = np.maximum(2, perim_coords.shape[0] // dhr.HR_NUM_PERIM_SAMPLES)
        perim_coords = perim_coords[::skip, :]

    metadata[tile_id] = {
        'cnt_xy': center_pts,
        'perim_xy': perim_coords,
        'wsipath': svspth,
        'scan_level': scan_level,
        'foreground_indices': foreground_indices,
        'tile_id': tile_id,
    }

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
        pred_ensemble = torch.argmax(pred_ensemble, 1).cpu().numpy()
        for tj, tile_id in enumerate(tile_ids.numpy()):
            pred_mask[metadata[tile_id]['foreground_indices']] = pred_ensemble[tj]

pred_mask_rgb = np.eye(4)[pred_mask][..., 1:]
pred_mask_rgb = Image.fromarray(pred_mask_rgb.astype(np.uint8) * 255)
pred_mask_rgb = pred_mask_rgb.resize((x//us, y//us))
pred_mask_rgb.save('slic_out_mask.png')

slic_out = mark_boundaries(image, labels, color=(0, 0, 0))
Image.fromarray((255*slic_out).astype(np.uint8)).save('slic_out.png')

