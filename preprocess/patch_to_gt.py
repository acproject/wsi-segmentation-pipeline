'''
original preprocessing is
done on wsi's (svs)
given patches, this generates
gt mask.
this script is specifically written
for bach iciar 2018 dataset
(part a images)
run by: python patch_to_gt.py --patch_folder your_path/
args.patch_folder = '/home/ozan/ICIAR2018_BACH_Challenge/Photos'

'''
import os
import numpy as np
from myargs import args
import glob
import utils.preprocessing as preprocessing
import utils.filesystem as ufs
from PIL import Image

args.patch_folder = '/home/ozan/ICIAR2018_BACH_Challenge/Photos'

if __name__ == '__main__':

    ufs.make_folder('../' + args.train_image_pth, False)

    ' map class names to codes '
    cls_codes = {
        'Normal': 0,
        'Benign': 1,
        'InSitu': 2,
        'Invasive': 3
    }

    ' check if metadata gt.npy already exists to append to it '
    metadata_pth = '../{}/gt.npy'.format(args.train_image_pth)
    metadata = ufs.fetch_metadata(metadata_pth)

    cls_folders = glob.glob('{}/*/'.format(args.patch_folder))

    for cls_folder in cls_folders:
        cls_name = cls_folder.split('/')[-2]
        cls_code = cls_codes[cls_name]

        'gt will be a single number over the patch since preds are on patch level (not pixel) '
        gt = cls_code*np.ones((args.tile_h, args.tile_w), dtype=np.uint8)
        gt = Image.fromarray(gt)

        image_paths = sorted(glob.glob('{}*.tif'.format(cls_folder)))
        for tile_id, image_path in enumerate(image_paths):

            filename = os.path.basename(image_path)
            metadata[filename] = {}

            image = Image.open(image_path).convert('RGB')
            image = image.resize((args.tile_w, args.tile_h))

            'get low res. nuclei image/foreground mask'
            mask = preprocessing.find_nuclei(image)
            mask = Image.fromarray(mask.astype(np.uint8))


            'save everything'
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
            image.save('../' + tilepth_w)
            gt.save('../' + tilepth_g)
            mask.save('../' + tilepth_m)

    np.save('../{}/gt.npy'.format(args.train_image_pth), metadata)
