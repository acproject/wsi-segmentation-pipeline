# import the necessary packages
from skimage.segmentation import slic, quickshift, felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import openslide
from utils import preprocessing
from skimage.morphology.convex_hull import convex_hull_image
import cv2
from sklearn.cluster import KMeans
from myargs import args
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import MultiPoint, Point, Polygon
from sklearn.neighbors import KDTree
from shapely.geometry import Point, Polygon
from scipy.ndimage.morphology import binary_erosion
from mahotas import bwperim
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point
from shapely.ops import polygonize,unary_union
from utils import regiontools
from skimage.morphology import binary_erosion
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import os
from utils.read_xml import getGT
import glob
from tqdm import tqdm
import time
from openslide.lowlevel import *
import utils.dataset_hr as dhr
from concave_hull.ConcaveHull import concaveHull
from contour_ordering import evenly_spaced_points_on_a_contour as esp

# load the image and convert it to a floating point data type
impath = '/home/ozan/PycharmProjects/wsi_segmentation_pipeline/data/val/out2/2/A01.svs_256.png'
impath = '/home/ozan/ICIAR2018_BACH_Challenge/WSI/gt_thumbnails/A01.png'

args.raw_train_pth = '/home/ozan/ICIAR2018_BACH_Challenge/WSI/'
us = 1
us_kmeans = 8

wsipaths = sorted(glob.glob('{}/A*.svs'.format(args.raw_train_pth)))

print(wsipaths)

wsipath = wsipaths[0]
scan = openslide.OpenSlide(wsipath)
filename = os.path.basename(wsipath)
'get actual mask, i.e. the ground truth'
xmlpath = '{}/{}.xml'.format(args.raw_train_pth, filename.split('.svs')[0])
gt = getGT(xmlpath, scan, level=args.scan_level)
gt_rgb = np.eye(4)[gt][..., 1:]

wsi = scan.read_region((0, 0), scan.level_count-1, scan.level_dimensions[scan.level_count-1]).convert('RGB')
x_max, y_max = scan.level_dimensions[-1]
wsi_mask = preprocessing.find_nuclei(wsi)

n_labels, labels, stats, centers = cv2.connectedComponentsWithStats((gt > 0).astype(np.uint8))
centers = centers.astype(np.int)

fig = plt.figure()

'''
sorted_indices = stats[:, 4].argsort()[::-1]  # based on area, largest to smallest
stats = stats[sorted_indices, :]
centers = centers[sorted_indices, :]
'''

ms = 0.5

for tile_id in tqdm(range(1, n_labels)):

    l, u = stats[tile_id, [0, 1]]
    w, h = stats[tile_id, [2, 3]]
    area = stats[tile_id, 4]
    cx, cy = centers[tile_id, :]

    label_patch = labels == tile_id
    area = np.count_nonzero(label_patch)
    num_clusters = dhr.HR_NUM_CNT_SAMPLES  #6 + int(area / (0.01 * gt.size))

    if (w * h)/gt.size <= 0.005:

        n, cnt_pts, out_image, foreground_indices = regiontools.get_key_points(label_patch, us_kmeans, dhr.HR_NUM_CNT_SAMPLES, dhr.HR_NUM_CNT_SAMPLES)

        if n is not None:

            label_patch = Image.fromarray(label_patch.astype(np.uint8))
            x, y = label_patch.size
            label_patch = label_patch.resize((x // us_kmeans, y // us_kmeans))
            perim = bwperim(np.asarray(label_patch))
            coords_ = np.transpose(np.where(perim))[:, ::-1]  # (x,y) pairs
            cvh = concaveHull(coords_, 3)
            coords = esp(cvh, dhr.HR_NUM_PERIM_SAMPLES) * us_kmeans

            plt.plot(cnt_pts[:, 0], cnt_pts[:, 1], 'wo', ms=ms)
            plt.plot(coords[:, 0], coords[:, 1], 'ko', ms=ms)

    else:

        min_center_large = int((w * h)/(gt.size * 0.01))
        min_center_large = np.maximum(min_center_large, 5)
        min_center_large = num_clusters

        n, cnt_pts, out_image, foreground_indices = regiontools.get_key_points(label_patch, us_kmeans, min_center_large, min_center_large)

        if n is not None:

            for r_id in range(1, n+1):

                sub_patch = (out_image == r_id)

                n2, cnt_pts2, out_image2, foreground_indices = regiontools.get_key_points(sub_patch, us_kmeans, dhr.HR_NUM_CNT_SAMPLES)

                if n2 is None:
                    continue

                if tile_id == 0:  #  and np.count_nonzero(wsi_mask[foreground_indices]) / (foreground_indices[0].shape[0]) < 0.9:
                    continue

                labels[foreground_indices] = n_labels
                n_labels += 1

                #perim = bwperim(sub_patch)
                '''
                perim = bwperim(convex_hull_image(sub_patch))  # perim = bwperim(label_patch)
                coords = np.transpose(np.where(perim))[:, ::-1]  # (x,y) pairs
                skip = np.maximum(2, coords.shape[0]//num_perim_points)
                plt.plot(coords[::skip, 0], coords[::skip, 1], 'ko', ms=ms)
                '''
                sub_patch = Image.fromarray(sub_patch.astype(np.uint8))
                x, y = sub_patch.size
                sub_patch = sub_patch.resize((x // us_kmeans, y // us_kmeans))
                perim = bwperim(np.asarray(sub_patch))
                coords_ = np.transpose(np.where(perim))[:, ::-1]  # (x,y) pairs
                cvh = concaveHull(coords_, 3)
                coords = esp(cvh, dhr.HR_NUM_PERIM_SAMPLES) * us_kmeans

                plt.plot(coords[:, 0], coords[:, 1], 'ko', ms=ms)
                plt.plot(cnt_pts2[:, 0], cnt_pts2[:, 1], 'wo', ms=ms)


label_codes = np.unique(labels)[1:]
np.random.shuffle(label_codes)
for lj, l_code in enumerate(np.unique(labels)[1:]):
    labels[labels == l_code] = label_codes[lj]

plt.imshow(labels, cmap='jet')
plt.show()
fig.savefig('temp.png', dpi = 800)