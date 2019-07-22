'''
preprocess train images
whole slide
with sunnybrook images
the big difference from other images,
since we don't have any 'normal'
(i.e. not labeled) we ignore patches that
are only normal
args.raw_train_pth
``deprecated`` in favor of centered
patch extractor

'''
import openslide
import os
import numpy as np
from myargs import args
import glob
import utils.preprocessing as preprocessing
from utils.read_xml_sunnybrook import getGT
import utils.filesystem as ufs


if __name__ == '__main__':

    args.raw_train_pth = 'data/sunnybrook/WSI'

    ufs.make_folder('../' + args.train_image_pth, False)
    wsipaths = glob.glob('../{}/*.svs'.format(args.raw_train_pth))

    ' check if metadata gt.npy already exists to append to it '
    metadata_pth = '../{}/gt.npy'.format(args.train_image_pth)
    metadata = ufs.fetch_metadata(metadata_pth)

    for wsipath in sorted(wsipaths):
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
        gt = getGT(xmlpath, scan, level=args.scan_level)

        'get low res. nuclei image/foreground mask'
        mask = preprocessing.find_nuclei(wsi)

        'get tiles for masks and wsi'
        for tile_id, (tile_w, tile_g, tile_m) in enumerate(zip(
                preprocessing.tile_image(wsi, params),
                preprocessing.tile_image(gt, params),
                preprocessing.tile_image(mask, params))):

            '''
            sunnybrook normals are unreliable, if only
            normal in patch, ignore the patch.
            '''
            numsamples = np.bincount(np.asarray(tile_g[-1]).reshape(-1))

            if numsamples[0]/numsamples.sum() >= 0.75:
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
