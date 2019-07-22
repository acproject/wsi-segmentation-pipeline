from torch.utils import data
import numpy as np
from PIL import Image
import torch
import utils.preprocessing as preprocessing
import openslide
from myargs import args
import glob
import os
from torchvision import transforms


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

        'rescale'
        if args.scan_resize != 0:
            x_rs, y_rs = int(image.size[0] / args.scan_resize), int(image.size[1] / args.scan_resize)
            image = image.resize((x_rs, y_rs))
            label = label.resize((x_rs, y_rs))
            mask = mask.resize((x_rs, y_rs))

        'random crop'
        if args.tile_w < image.size[0] or args.tile_h < image.size[1]:
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(args.tile_w, args.tile_h))
            image = transforms.functional.crop(image, i, j, h, w)
            label = transforms.functional.crop(label, i, j, h, w)
            mask = transforms.functional.crop(mask, i, j, h, w)

        'rotate image by 90*'
        degree = int(torch.randint(0, 4, (1, ))) * 90
        image = image.rotate(degree)
        label = label.rotate(degree)
        mask = mask.rotate(degree)

        label = np.asarray(label)
        mask = torch.from_numpy(np.asarray(mask)).float()

        '''
        'idea/to do: instead of using class weights,
        'can we randomly drop pixels & loss associated with them
        'proportional to the gt class of that pixel
        '''
        'loss mask'
        '''
        label_tensor = torch.from_numpy(label)
        mask = torch.zeros(image.size[::-1]).byte()
        for ch in range(args.num_classes):
            m = label_tensor == ch
            mask[m] = torch.rand(np.count_nonzero(m)) >= args.cls_ratios[ch]
        '''

        image = self.image_aug(image)
        label = torch.from_numpy(label).long()
        mask = mask.float()

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
    def __init__(self, svs_pth, params, bs=args.batch_size):

        self.params = preprocessing.DotDict(params)
        self.wsis = {}

        wsipaths = glob.glob('{}/*.svs'.format(svs_pth))
        for wsipath in sorted(wsipaths):
            filename = os.path.basename(wsipath)
            scan = openslide.OpenSlide(wsipath)

            self.wsis[filename] = {
                'iterator': GenerateIterator_wsi(wsipath, self.params, bs),
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

        'gt mask'
        thmb = self.scan.get_thumbnail(self.scan.level_dimensions[-1])
        mask = preprocessing.find_nuclei(thmb)
        mask = Image.fromarray(mask).resize(self.scan.level_dimensions[args.scan_level])
        mask = np.asarray(mask)

        ' augmentation settings '
        self.image_aug = preprocessing.standard_augmentor(True)

        ' build the dataset '
        self.datalist = []

        for ypos in range(1, self.params.ih - 1 - self.params.ph, self.params.sh):
            for xpos in range(1, self.params.iw - 1 - self.params.pw, self.params.sw):
                if not preprocessing.isforeground(mask[ypos:ypos+self.params.ph, xpos:xpos+self.params.pw]):
                    continue
                self.datalist.append((xpos, ypos))

        xpos = self.params.iw - 1 - self.params.pw
        for ypos in range(1, self.params.ih - 1 - self.params.ph, self.params.sh):
            if not preprocessing.isforeground(mask[ypos:ypos + self.params.ph, xpos:xpos + self.params.pw]):
                continue
            self.datalist.append((xpos, ypos))

        ypos = self.params.ih - 1 - self.params.ph
        for xpos in range(1, self.params.iw - 1 - self.params.pw, self.params.sw):
            if not preprocessing.isforeground(mask[ypos:ypos + self.params.ph, xpos:xpos + self.params.pw]):
                continue
            self.datalist.append((xpos, ypos))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        'get top left corner'
        x, y = self.datalist[index]
        _x, _y = (4**args.scan_level)*x, (4**args.scan_level)*y

        'read in image'
        image = self.scan.read_region((_x, _y), args.scan_level, (self.params.pw, self.params.ph)).convert('RGB')

        if args.scan_resize != 1:
            x_rs, y_rs = int(image.size[0] / args.scan_resize), int(image.size[1] / args.scan_resize)
            image = image.resize((x_rs, y_rs))

        image = self.image_aug(image)

        return x, y, image


def GenerateIterator_wsi(wsipth, p, bs):

    params = {
        'batch_size': bs,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }

    return data.DataLoader(Dataset_wsi(wsipth, p), **params)


class Dataset_wsiwgt(data.Dataset):
    ' use a wsi image + gt to create the dataset '
    def __init__(self, wsipth, params):

        self.params = params

        'read the wsi scan'
        self.scan = openslide.OpenSlide(wsipth)
        self.params.iw, self.params.ih = self.scan.level_dimensions[args.scan_level]

        'read the gt image'
        filename = os.path.basename(wsipth)
        mask_pth = '{}/{}_mask.png'.format(os.path.dirname(wsipth), filename)
        self.gt = Image.open(mask_pth) if os.path.exists(mask_pth) else \
            np.zeros(self.scan.level_dimensions[args.scan_level][::-1], dtype=np.uint8)
        self.gt = np.asarray(self.gt)

        ' augmentation settings '
        self.image_aug = preprocessing.standard_augmentor(True)
        ' build the dataset '
        self.datalist = []

        for ypos in range(1, self.params.ih - 1 - self.params.ph, self.params.sh):
            for xpos in range(1, self.params.iw - 1 - self.params.pw, self.params.sw):
                self.datalist.append((xpos, ypos))

        xpos = self.params.iw - 1 - self.params.pw
        for ypos in range(1, self.params.ih - 1 - self.params.ph, self.params.sh):
            self.datalist.append((xpos, ypos))

        ypos = self.params.ih - 1 - self.params.ph
        for xpos in range(1, self.params.iw - 1 - self.params.pw, self.params.sw):
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

        label = self.gt[y:y+self.params.ph, x:x+self.params.pw]
        label = torch.from_numpy(label).long()

        return x, y, image, label


def GenerateIterator_wsiwgt(wsipth, p):

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }

    return data.DataLoader(Dataset_wsiwgt(wsipth, p), **params)


class Dataset_wsiswgt:
    ' all validation wsis with gt '
    def __init__(self, svs_pth, params):

        self.params = preprocessing.DotDict(params)
        self.wsis = {}

        wsipaths = glob.glob('{}/*.svs'.format(svs_pth))
        for wsipath in sorted(wsipaths):
            filename = os.path.basename(wsipath)
            scan = openslide.OpenSlide(wsipath)
            self.wsis[filename] = {
                'iterator': GenerateIterator_wsiwgt(wsipath, self.params),
                'wsipath': wsipath,
                'scan': scan
            }
