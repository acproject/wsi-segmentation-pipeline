from torch.utils import data
import numpy as np
from PIL import Image
import torch
import utils.preprocessing as preprocessing
import openslide
from myargs import args
import glob
import os
from tqdm import tqdm


class Dataset(data.Dataset):
    def __init__(self, impth, eval, duplicate_dataset):

        self.eval = eval
        ' augmentation settings '
        self.image_aug = preprocessing.standard_augmentor(eval)

        ' build the dataset '
        self.datalist = []
        gt = np.load('{}/gt.npy'.format(impth), allow_pickle=True).flatten()[0]
        for key in gt:
            self.datalist.append([{
                'wsi': gt[key][tile_id]['wsi'],
                'label': gt[key][tile_id]['label'],
            } for tile_id in gt[key]])
        self.datalist = [item for sublist in self.datalist for item in sublist]

        if not self.eval:
            from itertools import chain
            self.datalist = list(chain(*[[i] * duplicate_dataset for i in self.datalist]))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        'read in image&labels'
        image = Image.open(self.datalist[index]['wsi'])

        if isinstance(self.datalist[index]['label'], str):
            label = Image.open(self.datalist[index]['label'])
        else:
            label = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8))

        if not self.eval:
            'rotate image by 90*'
            degree = int(torch.randint(0, 4, (1, ))) * 90

            image = image.rotate(degree, expand=True)
            label = label.rotate(degree, expand=True)

            image = image.resize((args.tile_w, args.tile_h))
            label = label.resize((args.tile_w, args.tile_h))

        label = np.asarray(label)

        image = self.image_aug(image)
        label = torch.from_numpy(label.astype(np.uint8)).long()

        is_cls = isinstance(self.datalist[index]['label'], int)
        is_reg = isinstance(self.datalist[index]['label'], float)
        is_seg = isinstance(self.datalist[index]['label'], str)

        cls_code = self.datalist[index]['label'] if not is_seg else -1

        return image, label, is_cls, is_reg, is_seg, cls_code


def GenerateIterator(impth, eval=False, duplicate_dataset=1):

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }

    return data.DataLoader(Dataset(impth, eval=eval, duplicate_dataset=duplicate_dataset), **params)


class Dataset_wsis:
    ' all validation wsis '
    def __init__(self, svs_pth, params, bs=args.batch_size):

        self.params = preprocessing.DotDict(params)
        self.wsis = {}

        wsipaths = glob.glob('{}/**/*.svs'.format(svs_pth))
        with tqdm(enumerate(sorted(wsipaths))) as t:
            for wj, wsipath in t:
                t.set_description('Loading wsis.. {:d}/{:d}'.format(1 + wj, len(wsipaths)))

                filename = os.path.basename(wsipath)
                scan = openslide.OpenSlide(wsipath)
                itr = GenerateIterator_wsi(wsipath, self.params, bs)

                msk_pth = '{}/{}.png'.format(args.wsi_mask_pth, filename)

                if itr is not None:
                    self.wsis[filename] = {
                        'iterator': itr,
                        'wsipath': wsipath,
                        'scan': scan,
                        'maskpath': msk_pth
                    }


class Dataset_wsi(data.Dataset):
    ' use a wsi image to create the dataset '
    def __init__(self, wsipth, params):

        self.params = params

        ' build the dataset '
        self.datalist = []

        'read the wsi scan'
        filename = os.path.basename(wsipth)
        self.scan = openslide.OpenSlide(wsipth)

        ' if a slide has less levels than our desired scan level, ignore the slide'
        if len(self.scan.level_dimensions) - 1 >= args.scan_level:

            self.params.iw, self.params.ih = self.scan.level_dimensions[args.scan_level]

            'gt mask'
            'find nuclei is slow, hence save masks from preprocessing' \
            'for later use'
            msk_pth = '{}/{}.png'.format(args.wsi_mask_pth, filename)
            if not os.path.exists(msk_pth):
                thmb = self.scan.read_region((0, 0), self.scan.level_count - 1, self.scan.level_dimensions[-1]).convert('RGB')
                mask = preprocessing.find_nuclei(thmb)
                Image.fromarray(mask.astype(np.uint8)).save(msk_pth)
            else:
                mask = Image.open(msk_pth).convert('L')
                mask = np.asarray(mask)

            ' augmentation settings '
            self.image_aug = preprocessing.standard_augmentor(True)

            'downsample multiplier'
            m = self.scan.level_downsamples[args.scan_level]/self.scan.level_downsamples[-1]
            dx, dy = int(self.params.pw*m), int(self.params.ph*m)

            for ypos in range(1, self.params.ih - 1 - self.params.ph, self.params.sh):
                for xpos in range(1, self.params.iw - 1 - self.params.pw, self.params.sw):
                    yp, xp = int(ypos * m), int(xpos * m)
                    if not preprocessing.isforeground(mask[yp:yp+dy, xp:xp+dx]):
                        continue
                    self.datalist.append((xpos, ypos))

            xpos = self.params.iw - 1 - self.params.pw
            for ypos in range(1, self.params.ih - 1 - self.params.ph, self.params.sh):
                yp, xp = int(ypos * m), int(xpos * m)
                if not preprocessing.isforeground(mask[yp:yp + dy, xp:xp + dx]):
                    continue
                self.datalist.append((xpos, ypos))

            ypos = self.params.ih - 1 - self.params.ph
            for xpos in range(1, self.params.iw - 1 - self.params.pw, self.params.sw):
                yp, xp = int(ypos * m), int(xpos * m)
                if not preprocessing.isforeground(mask[yp:yp + dy, xp:xp + dx]):
                    continue
                self.datalist.append((xpos, ypos))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        'get top left corner'
        x, y = self.datalist[index]
        _x, _y = int(self.scan.level_downsamples[args.scan_level] * x), int(self.scan.level_downsamples[args.scan_level] * y)

        'read in image'
        image = self.scan.read_region((_x, _y), args.scan_level, (self.params.pw, self.params.ph)).convert('RGB')

        if args.scan_resize != 1:
            image = image.resize((args.tile_w, args.tile_h))

        image = self.image_aug(image)

        return float(x), float(y), image


def GenerateIterator_wsi(wsipth, p, bs):

    params = {
        'batch_size': bs,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }

    dataset = Dataset_wsi(wsipth, p)
    if len(dataset) > 0:
        return data.DataLoader(dataset, **params)

    return None
