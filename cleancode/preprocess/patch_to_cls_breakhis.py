'''
breakhis dataset patches
'''
from tqdm import tqdm
import os
import numpy as np
from myargs import args
import glob
import utils.preprocessing as preprocessing
import utils.filesystem as ufs
from PIL import Image

args.patch_folder = '../data/breakhis/malignant/'

if __name__ == '__main__':

    ufs.make_folder('../' + args.train_image_pth, False)

    ' check if metadata gt.npy already exists to append to it '
    metadata_pth = '../{}/gt.npy'.format(args.train_image_pth)
    metadata = ufs.fetch_metadata(metadata_pth)

    num_images = 0
    for root, dirs, files in tqdm(os.walk(args.patch_folder, topdown=False)):
        for name in files:

            if '.png' not in name \
                    or '/40X' not in root:
                continue

            image_path = os.path.join(root, name)
            filename = os.path.basename(image_path)

            num_images += 1

            #metadata.pop(filename, None)
            #continue

            dcis = '/ductal_carcinoma/' in root
            cls_code = 2 if dcis else 3

            'gt will be a single number over the patch since preds are on patch level (not pixel) '
            gt = cls_code*np.ones((args.tile_h, args.tile_w), dtype=np.uint8)
            gt = Image.fromarray(gt)
            mask = np.ones((args.tile_h, args.tile_w), dtype=np.uint8)
            mask = Image.fromarray(mask)

            metadata[filename] = {}

            image = Image.open(image_path).convert('RGB')
            image = image.resize((args.tile_h, args.tile_w))

            'save paths'
            tilepth_w = '{}/w_{}_{}.png'.format(args.train_image_pth, filename, 0)
            tilepth_g = '{}/g_{}_{}.png'.format(args.train_image_pth, filename, 0)
            tilepth_m = '{}/m_{}_{}.png'.format(args.train_image_pth, filename, 0)

            ' save metadata '
            metadata[filename][0] = {
                'wsi': tilepth_w,
                'label': tilepth_g,
                'mask': tilepth_m,
                'is_cls': 1,
            }
            ' save images '
            image.save('../' + tilepth_w)
            mask.save('../' + tilepth_m)
            gt.save('../' + tilepth_g)


    print('{} images have been processed.'.format(num_images))
    np.save('../{}/gt.npy'.format(args.train_image_pth), metadata)
