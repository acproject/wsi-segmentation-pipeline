from skimage import color
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms
from myargs import args
import torch
from mahotas import bwperim
import utils.filesystem as ufs
from scipy.ndimage.morphology import binary_fill_holes
import gc


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __setitem__(self, key, value):
        self.__dict__.update({key: value})


def isforeground(arr, thresh=0.25):
    """
    isforeground: check if patch
    has more than thresh% of

    (args)
    arr: np array
    thresh: % of foreground required
    (out)
    bool
    """
    return np.count_nonzero(arr) / arr.size >= thresh


def find_nuclei(wsi):
    """
    find nuclei: preprocessing fn.
    removes pink and white regions.
    filters nuclei (purplish regions)

    (args)
    wsi: pil image object
    (out)
    mask: filtered wsi
    """
    np.seterr(divide='ignore')

    ' lab threshold to remove white'
    lab = color.rgb2lab(np.asarray(wsi))
    mu = np.mean(lab[..., 1])
    lab = lab[..., 1] > (1+0.1)*mu

    ' hsv threshold to remove pink '
    hsv = color.rgb2hsv(np.asarray(wsi)) * 360
    hsv = (hsv[..., 0] > 270) * (hsv[..., 0] < 300)

    mask = lab+hsv

    ' dilate/close whatever '
    mask = binary_fill_holes(mask)

    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    mask = mask.astype(np.uint8)

    return mask


def tile_image(image, params):
    """
    tile image: given an image,
    tiles it wrt tile dims
    &step sizes
    and returns parts
    piece by piece

    (args)
    params: pil image object
    args: {image width,height; patch w,h; step w,h},
    (out)
    yields all tiles as image objects
    along with top left coordinates of
    tile
    in form:
    top x, top y, tile image
    """

    if type(image) == np.ndarray:
        image = Image.fromarray(image.astype(np.uint8))

    params = DotDict(params)

    if (params.ih - 1 - params.ph) <= 0 or (params.iw - 1 - params.pw) <= 0:
        xpos = 0
        ypos = 0
        yield xpos, ypos, image.crop((xpos, ypos, xpos+params.pw, ypos+params.ph))
        return

    for ypos in range(0, params.ih - 1 - params.ph, params.sh):
        for xpos in range(0, params.iw - 1 - params.pw, params.sw):
            yield xpos, ypos, image.crop((xpos, ypos, xpos+params.pw, ypos+params.ph))

    xpos = params.iw - 1 - params.pw
    for ypos in range(0, params.ih - 1 - params.ph, params.sh):
        yield xpos, ypos, image.crop((xpos, ypos, xpos + params.pw, ypos + params.ph))

    ypos = params.ih - 1 - params.ph
    for xpos in range(0, params.iw - 1 - params.pw, params.sw):
        yield xpos, ypos, image.crop((xpos, ypos, xpos + params.pw, ypos + params.ph))


def pred_to_mask(pred, wsi=None, perim=False):
    """
    given a prediction logit
    of size [# classes, width, height]
    & corresponding wsi image
    return the mask embedded onto
    this wsi (as np array)
    """

    str_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    '''
    threshold probs: if network is inclined to guess
    the same class, increase threshold for that class 
    '''
    pred = torch.softmax(torch.from_numpy(pred), dim=0)
    for cj in range(args.num_classes):
        pred[cj, pred[cj, ...] < args.class_probs[cj]] = 0
    pred = torch.argmax(pred, dim=0)
    pred = pred.numpy()

    'save image'
    pred = 255 * (np.eye(args.num_classes)[pred][..., 1:]).astype(np.uint8)
    wsi = np.zeros_like(pred) if wsi is None else np.array(wsi.copy())
    for cj in range(args.num_classes - 1):
        rgbcolor = [0, 0, 0]
        rgbcolor[cj] = 255

        if perim:
            pred[..., cj] = bwperim(pred[..., cj])
            pred[..., cj] = cv2.dilate(pred[..., cj], str_elem, iterations=1)

        wsi[pred[..., cj] > 0, :] = rgbcolor

    del pred

    return wsi


def standard_augmentor(eval=False):

    if eval:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.dataset_mean, args.dataset_std),
        ])

    return transforms.Compose([
        transforms.ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize(args.dataset_mean, args.dataset_std),
    ])


def cls_ratios(pth=args.train_image_pth, ignore_index=None):
    '''
    given gt.npy,
    calculates class distributions
    of images
    '''
    metadata_pth = '{}/gt.npy'.format(pth)
    metadata = ufs.fetch_metadata(metadata_pth)

    numsamples = np.zeros((args.num_classes, ))

    for _, item in metadata.items():
        for _, subitem in item.items():
            l = Image.open(subitem['label'])
            l = np.asarray(l)
            n = np.bincount(l.reshape(-1), minlength=args.num_classes)

            numsamples = numsamples + n

    if ignore_index is not None:
        numsamples[ignore_index] = 0

    return numsamples/numsamples.sum()

