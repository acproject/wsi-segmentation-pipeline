'''
overlay the tumor bed output
using heatmap, convex hull
on the heatmap
'''

from PIL import Image
import cv2
from mahotas import bwperim
import numpy as np
from skimage.morphology.convex_hull import convex_hull_image as chull
import openslide


scan = openslide.OpenSlide('/home/ozan/remoteDir/Case 8/101323.svs')
wsi = scan.read_region((0, 0), 2, scan.level_dimensions[2]).convert('RGB')


im = Image.open('/home/ozan/101323.svs_128_heatmap.png').convert('L')
x, y = im.size

wsi = wsi.resize((x, y))
wsi = np.array(wsi)


heatmap = np.array(im)

im = np.uint8(heatmap/255 >= 0.9)

im = cv2.morphologyEx(
	im,
	cv2.MORPH_OPEN,
	kernel=np.ones((30, 30))
)

heatmap_orig = heatmap

heatmap = heatmap * im
heatmap = np.repeat(heatmap[..., np.newaxis], 3, 2)


tb_perim = cv2.morphologyEx(
	bwperim(chull(im)).astype(np.uint8),
	cv2.MORPH_DILATE,
	kernel=np.ones((20, 20))
)

overlay = 0.65 * wsi + 0.35 * heatmap
yy, xx = np.where(tb_perim)
overlay[yy,xx,...]=0
overlay = np.uint8(overlay)

overlay_image = Image.fromarray(overlay).resize((x//4, y//4))
overlay_image.save('overlay_tumor_bed.png')


wsi = Image.fromarray(wsi).resize((x//4, y//4))
wsi.save('wsi.png')

tumor_bed_perim = Image.fromarray(np.uint8(255 * tb_perim)).resize((x//4, y//4))
tumor_bed_perim.save('tumor_bed_perim.png')

heatmap_orig = Image.fromarray(heatmap_orig).resize((x//4, y//4))
heatmap_orig.save('heatmap.png')