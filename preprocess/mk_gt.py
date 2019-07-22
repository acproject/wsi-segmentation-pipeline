'''
make gt mask (for validation)
generates an image where the
pixel represent classes from
the xml file (not rgb, just class
codes)
'''
import openslide
import os
import numpy as np
from myargs import args
import glob
from utils.read_xml import getGT
from PIL import Image


wsipaths = glob.glob('../{}/*.svs'.format(args.raw_val1_pth))

for wsipath in sorted(wsipaths):
    'read scan'
    scan = openslide.OpenSlide(wsipath)
    filename = os.path.basename(wsipath)
    'get actual mask, i.e. the ground truth'
    xmlpath = '../{}/{}.xml'.format(args.raw_val1_pth, filename.split('.svs')[0])
    gt = getGT(xmlpath, scan, level=args.scan_level)
    gt = Image.fromarray(gt.astype(np.uint8))

    if args.scan_resize != 1:
        x_rs, y_rs = int(gt.size[0] / args.scan_resize), int(gt.size[1] / args.scan_resize)
        gt = gt.resize((x_rs, y_rs))

    gt.save('../{}/{}_mask.png'.format(args.raw_val1_pth, filename))
