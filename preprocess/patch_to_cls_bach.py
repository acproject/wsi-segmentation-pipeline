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
from tqdm import tqdm
import os
import numpy as np
from myargs import args
import glob
import utils.preprocessing as preprocessing
import utils.filesystem as ufs
from PIL import Image

args.patch_folder = '../data/bach/patches/Photos'

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

    for cls_folder in tqdm(cls_folders):
        cls_name = cls_folder.split('/')[-2]
        cls_code = cls_codes[cls_name]

        image_paths = sorted(glob.glob('{}*.png'.format(cls_folder)))
        for image_path in image_paths:

            filename = os.path.basename(image_path)
            metadata[filename] = {}

            image = Image.open(image_path).convert('RGB')
            image = image.resize((args.tile_h, args.tile_w))

            'save paths'
            tilepth_w = '{}/w_{}_{}.png'.format(args.train_image_pth, filename, 0)

            ' save metadata '
            metadata[filename][0] = {
                'wsi': tilepth_w,
                'label': cls_code,
            }
            ' save images '
            image.save('../' + tilepth_w)

    np.save('../{}/gt.npy'.format(args.train_image_pth), metadata)
