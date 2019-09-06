'''
breakhis dataset patches
'''
from tqdm import tqdm
import os
import numpy as np
from myargs import args
import utils.filesystem as ufs
from PIL import Image
import csv
import glob
from utils import preprocessing

is_spie = True  # spie challenge or  wsi pipeline?
val = True

if val:
    args.patch_folder = '/home/ozan/Downloads/breastpathq/datasets/validation'
    args.label_csv_path = '/home/ozan/Downloads/breastpathq-test/val_labels.csv'
    #args.patch_folder = '/home/ozan/Downloads/breastpathq/datasets (copy)/validation'
    #args.label_csv_path = '/home/ozan/Downloads/breastpathq/datasets (copy)/val_labels.csv'
    savepath = args.val_image_pth
else:
    args.patch_folder = '/home/ozan/Downloads/breastpathq/datasets/train'
    args.label_csv_path = '/home/ozan/Downloads/breastpathq/datasets/train_labels.csv'
    #args.patch_folder = '/home/ozan/Downloads/breastpathq/datasets (copy)/train'
    #args.label_csv_path = '/home/ozan/Downloads/breastpathq/datasets (copy)/train_labels.csv'
    savepath = args.train_image_pth


if __name__ == '__main__':


    'train'
    ufs.make_folder('../' + savepath, is_spie)
    metadata_pth_train = '../{}/gt.npy'.format(savepath)
    metadata = ufs.fetch_metadata(metadata_pth_train)

    raw_gt = {}

    cc = []

    with open('{}'.format(args.label_csv_path)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            image_id = int(row[0])
            region_id = int(row[1])
            cellularity = float(row[2])
            if image_id not in raw_gt:
                raw_gt[image_id] = {}
            raw_gt[image_id][region_id] = cellularity
            cc.append(cellularity)

    cc = np.array(cc)
    cc = np.unique(cc)
    print(cc.shape, cc)

    for num_images, image_path in tqdm(enumerate(glob.glob('{}/*.tif'.format(args.patch_folder)))):

        image_id, region_id = os.path.basename(image_path).split('_')
        region_id = region_id.replace('.tif', '')
        image_id, region_id = int(image_id), int(region_id)

        cellularity = raw_gt[image_id][region_id]
        if is_spie:
            cls_code = float(cellularity)  # for regression use probabilities. not classification
            cls_code = int(cellularity > 0)
        else:
            cls_code = int(cellularity > 0)

        image = Image.open(image_path).convert('RGB')
        image = image.resize((args.tile_h, args.tile_w))
        image = preprocessing.quantize_image(image)
        'save paths'
        tilepth_w = '{}/w_{}_{}.png'.format(savepath, image_id, region_id)
        ' save images '
        image.save('../' + tilepth_w)

        if image_id not in metadata:
            metadata[image_id] = {}

        metadata[image_id][region_id] = {
            'wsi': tilepth_w,
            'label': cls_code,
        }

    np.save(metadata_pth_train, metadata)
