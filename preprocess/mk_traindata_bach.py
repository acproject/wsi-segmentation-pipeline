import openslide
'''
preprocess train images
whole slide
``deprecated`` in favor of centered
patch extractor
'''
import os
import numpy as np
from myargs import args
import glob
import utils.preprocessing as preprocessing
from utils.read_xml import getGT
import utils.filesystem as ufs
from tqdm import tqdm


if __name__ == '__main__':

    ufs.make_folder('../' + args.train_image_pth, True)
    wsipaths = glob.glob('../{}/*.svs'.format(args.raw_train_pth))

    ' check if metadata gt.npy already exists to append to it '
    metadata_pth = '../{}/gt.npy'.format(args.train_image_pth)
    metadata = ufs.fetch_metadata(metadata_pth)

    numpixels = args.tile_w*args.tile_h

    for wsipath in tqdm(sorted(wsipaths)):
        'read scan and get metadata'
        scan = openslide.OpenSlide(wsipath)
        filename = os.path.basename(wsipath)
        metadata[filename] = {}

        params = {
            'ih': scan.level_dimensions[args.scan_level][1],
            'iw': scan.level_dimensions[args.scan_level][0],
            'ph': args.tile_h,
            'pw': args.tile_w,
            'sh': args.tile_stride_h,
            'sw': args.tile_stride_w,
        }

        'read in image'
        wsi = scan.read_region((0, 0), args.scan_level, scan.level_dimensions[args.scan_level]).convert('RGB')

        'get actual mask, i.e. the ground truth'
        xmlpath = '../{}/{}.xml'.format(args.raw_train_pth, filename.split('.svs')[0])
        gt = getGT(xmlpath, scan, sample=4 ** args.scan_level, level=args.scan_level)

        'get low res. nuclei image/foreground mask'
        mask = np.zeros(wsi.size[::-1], dtype=np.uint8)  # preprocessing.find_nuclei(wsi)

        'get tiles for masks and wsi'
        for tile_id, (tile_w, tile_g, tile_m) in enumerate(zip(
                preprocessing.tile_image(wsi, params),
                preprocessing.tile_image(gt, params),
                preprocessing.tile_image(mask, params))):

            'skip background patches'
            m_ = (np.asarray(tile_m[-1]) > 0).astype(np.uint8)
            g_ = (np.asarray(tile_g[-1]) > 0).astype(np.uint8)

            if np.count_nonzero(m_)/numpixels < 0.5 and np.count_nonzero(g_)/numpixels < 0.01:
                continue

            tilepth_w = '{}/w_{}_{}.png'.format(args.train_image_pth, filename, tile_id)
            tilepth_g = '{}/g_{}_{}.png'.format(args.train_image_pth, filename, tile_id)
            tilepth_m = '{}/m_{}_{}.png'.format(args.train_image_pth, filename, tile_id)

            ' save metadata '
            metadata[filename][tile_id] = {
                'wsi': tilepth_w,
                'label': tilepth_g,
                'mask': tilepth_m,
            }

            ' save images '
            tile_w[-1].save('../' + tilepth_w)
            tile_g[-1].save('../' + tilepth_g)
            tile_m[-1].save('../' + tilepth_m)

    np.save(metadata_pth, metadata)
