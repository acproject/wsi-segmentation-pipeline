from torch.utils import data
import numpy as np
from PIL import Image
import torch
import utils.preprocessing as preprocessing
import openslide
from myargs import args
import glob
import os


class Dataset(data.Dataset):
    def __init__(self, impth, eval):

        self.eval = eval
        ' augmentation settings '
        self.image_aug = preprocessing.standard_augmentor()

        ' build the dataset '
        self.datalist = []
        gt = np.load('{}/gt.npy'.format(impth), allow_pickle=True).flatten()[0]
        for key in gt:
            self.datalist.append([{
                'wsi': gt[key][tile_id]['wsi'],
                'label': gt[key][tile_id]['label'],
                'mask': gt[key][tile_id]['mask'],
            } for tile_id in gt[key]])
        self.datalist = [item for sublist in self.datalist for item in sublist]

        if not self.eval:
            from itertools import chain
            self.datalist = list(chain(*[[i] * 1 for i in self.datalist]))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        'read in image&labels'
        image = Image.open(self.datalist[index]['wsi'])
        label = Image.open(self.datalist[index]['label'])
        mask = Image.open(self.datalist[index]['mask'])

        'rotate image by 90*'
        degree = int(torch.randint(0, 4, (1, ))) * 90
        image = image.rotate(degree)
        label = label.rotate(degree)
        mask = mask.rotate(degree)

        image = self.image_aug(image)
        label = torch.from_numpy(np.asarray(label)).long()
        mask = torch.from_numpy(np.asarray(mask)).long()

        return image, label, mask


def GenerateIterator(impth, eval=False):

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }

    return data.DataLoader(Dataset(impth, eval=eval), **params)


class Dataset_wsis:
    ' all validation wsis '
    def __init__(self, svs_pth, params):

        self.params = preprocessing.DotDict(params)
        self.wsis = {}

        wsipaths = glob.glob('{}/*.svs'.format(svs_pth))
        for wsipath in sorted(wsipaths):
            filename = os.path.basename(wsipath)
            scan = openslide.OpenSlide(wsipath)
            self.wsis[filename] = {
                'iterator': GenerateIterator_wsi(wsipath, self.params),
                'wsipath': wsipath,
                'scan': scan
            }


class Dataset_wsi(data.Dataset):
    ' use a wsi image to create the dataset '
    def __init__(self, wsipth, params):

        self.params = params

        'read the wsi scan'
        self.scan = openslide.OpenSlide(wsipth)
        self.params.iw, self.params.ih = self.scan.level_dimensions[args.scan_level]

        ' augmentation settings '
        self.image_aug = preprocessing.standard_augmentor(True)

        ' build the dataset '
        self.datalist = []

        for ypos in range(0, self.params.ih - 1 - self.params.ph, self.params.sh):
            for xpos in range(0, self.params.iw - 1 - self.params.pw, self.params.sw):
                self.datalist.append((xpos, ypos))

        xpos = self.params.iw - 1 - self.params.pw
        for ypos in range(0, self.params.ih - 1 - self.params.ph, self.params.sh):
            self.datalist.append((xpos, ypos))

        ypos = self.params.ih - 1 - self.params.ph
        for xpos in range(0, self.params.iw - 1 - self.params.pw, self.params.sw):
            self.datalist.append((xpos, ypos))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        'get top left corner'
        x, y = self.datalist[index]
        _x, _y = (4**args.scan_level)*x, (4**args.scan_level)*y

        'read in image'
        image = self.scan.read_region((_x, _y), args.scan_level, (self.params.pw, self.params.ph)).convert('RGB')

        image = self.image_aug(image)

        return x, y, image


def GenerateIterator_wsi(wsipth, p):

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }

    return data.DataLoader(Dataset_wsi(wsipth, p), **params)


