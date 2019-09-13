from skimage import color
import numpy as np
from PIL import Image
import cv2
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from myargs import args
import torch
from mahotas import bwperim
import utils.filesystem as ufs
from scipy.ndimage.morphology import binary_fill_holes
import gc
import openslide
from utils import regiontools
import glob
from sklearn.cluster import MiniBatchKMeans as KMeans


def display_tensor_images_on_grid(image):

    def make_image(img):
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        return Image.fromarray((255 * npimg).astype(np.uint8))

    rev_norm = NormalizeInverse(args.dataset_mean, args.dataset_std)

    out = image.clone()
    for ij in range(image.size(0)):
        out[ij, ...] = rev_norm(image[ij, ...])

    return make_image(make_grid(out.cpu()))


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __setitem__(self, key, value):
        self.__dict__.update({key: value})


def isforeground(arr, thresh=0.05):
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


def find_nuclei(wsi, mu_percent=0.1, mode='hsv', fill_mask=False):
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
    if mode is 'lab':
        lab = color.rgb2lab(np.asarray(wsi))
        mu = np.mean(lab[..., 1])
        lab = lab[..., 1] > (1+mu_percent)*mu
        mask = lab.astype(np.uint8)

    ' hsv threshold to remove pink '
    if mode is 'hsv':
        hsv = color.rgb2hsv(np.asarray(wsi))
        hsv = hsv[..., 1] > mu_percent
        mask = hsv.astype(np.uint8)

    ' dilate/close '
    if fill_mask:
        mask = binary_fill_holes(mask)

        kernel_size = 10
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

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


def threshold_probs(pred):
    '''
    threshold probs: if network is inclined to guess
    the same class, increase threshold for that class
    '''

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)

    pred = torch.softmax(pred, dim=0)

    for cj in range(args.num_classes):
        pred[cj, pred[cj, ...] < args.class_probs[cj]] = 0

    pred = pred.numpy()

    return np.argmax(pred, axis=0).astype(np.uint8), pred


def pred_to_mask(pred, wsi=None, perim=False):
    """
    given a prediction logit
    of size [# classes, width, height]
    & corresponding wsi image
    return the mask embedded onto
    this wsi (as np array)
    """

    str_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    pred = threshold_probs(pred)

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


def nextpow2(x):
    x = int(x)
    return 1 << (x-1).bit_length()


def cls_weights(pth=args.train_image_pth, ignore_index=None, ignore_cls=False, ignore_seg=False):
    '''
    given gt.npy,
    calculates class distributions
    of images and returns inverse
    '''
    metadata_pth = '{}/gt.npy'.format(pth)
    metadata = ufs.fetch_metadata(metadata_pth)

    numsamples_cls = np.zeros((args.num_classes, ), dtype=np.int64)
    numsamples_seg = np.zeros((args.num_classes, ), dtype=np.int64)

    for _, item in metadata.items():
        for _, subitem in item.items():

            if not ignore_seg and isinstance(subitem['label'], str):
                l = Image.open(subitem['label'])
                l = np.array(l)
                n = np.bincount(l.reshape(-1), minlength=args.num_classes)
                numsamples_seg += n

            if not ignore_cls and isinstance(subitem['label'], int):
                numsamples_cls[int(subitem['label'])] += 1

    if ignore_index is not None:
        numsamples_seg[ignore_index] = 0
        numsamples_cls[ignore_index] = 0

    ratios_cls, ratios_seg = numsamples_cls/(args.epsilon+numsamples_cls.sum()), numsamples_seg/(args.epsilon+numsamples_seg.sum())

    ' find classes with sample count > 0'
    nonzero_cls = np.nonzero(numsamples_cls)
    nonzero_seg = np.nonzero(numsamples_seg)

    'inverse ratios (i.e. weights)'
    ratios_cls = 1.0 / ratios_cls[nonzero_cls]
    ratios_seg = 1.0 / ratios_seg[nonzero_seg]

    'placeholder for class weights'
    cls_weights_cls = np.zeros((args.num_classes, ))
    cls_weights_seg = np.zeros((args.num_classes, ))

    'normalize max weight to 1'
    if ~(ratios_cls == []):
        ratios_cls /= (args.epsilon+ratios_cls.max())
        cls_weights_cls[nonzero_cls] = ratios_cls
    if ~(ratios_seg == []):
        ratios_seg /= (args.epsilon+ratios_seg.max())
        cls_weights_seg[nonzero_seg] = ratios_seg

    return cls_weights_cls, cls_weights_seg


def cls_ratios_ssr(pth=args.train_image_pth, ignore_index=None, option='segmentation'):
    '''
    given image paths,
    calculates class distributions
    of images
    '''

    numsamples = np.zeros((args.num_classes,))

    if option == 'classification':
        gt = np.load('{}/gt.npy'.format(pth), allow_pickle=True).flatten()[0]
        for key in gt:
            for tile_id in gt[key]:
                numsamples[gt[key][tile_id]['label']] += 1

    elif option == 'segmentation':
        metadata_pth = glob.glob('{}/*_gt.png'.format(pth))
        for pth in metadata_pth:
                l = Image.open(pth)
                l = np.asarray(l)
                l = np.concatenate((np.zeros(shape=(*l.shape[:2], 1)), l), -1)
                l = np.argmax(l, -1)

                n = np.bincount(l.reshape(-1), minlength=args.num_classes)

                numsamples = numsamples + n

    if ignore_index is not None:
        numsamples[ignore_index] = 0

    return numsamples/numsamples.sum()


def cls_ratios_hr(pth=args.train_hr_image_pth, ignore_index=None):

    import utils.dataset_hr as ds

    metadata_pth = ufs.fix_path('../{}/gt.npy'.format(pth))
    metadata = ufs.fetch_metadata(metadata_pth)

    ' build the datalist '
    cls = np.zeros(args.num_classes, )
    for filename in metadata:
        first_region_id = list(metadata[filename].keys())[0]
        first_sub_region_id = list(metadata[filename][first_region_id].keys())[0]
        pth = metadata[filename][first_region_id][first_sub_region_id]['wsipath']
        pth = ufs.fix_path(pth)

        _wsi_ = openslide.OpenSlide(pth)

        params = {
            'iw': _wsi_.level_dimensions[0][0],
            'ih': _wsi_.level_dimensions[0][1],
            'tile_w': ds.HR_PATCH_W,
            'tile_h': ds.HR_PATCH_H,
            'scan_level': ds.HR_SCAN_LEVEL
        }
        params = DotDict(params)

        for conncomp in metadata[filename]:
            for id in metadata[filename][conncomp]:

                region_obj = metadata[filename][conncomp][id].copy()

                'which/are the points valid for this patch size, scan level combo?'
                region_obj['cnt_xy'], num_cnt_pts = regiontools.map_points(region_obj['cnt_xy'], params)
                region_obj['perim_xy'], num_perim_pts = regiontools.map_points(region_obj['perim_xy'], params)

                if num_cnt_pts >= ds.HR_NUM_CNT_SAMPLES and \
                        num_perim_pts >= ds.HR_NUM_PERIM_SAMPLES:
                    cls[region_obj['label']] += 1

    cls = np.array(cls)
    print(cls, cls.sum())
    cls = cls/cls.sum()

    return cls


def quantize_image(image, n_colors=0):

    if n_colors < 2:
        return image

    image = np.asarray(image)
    arr = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    q = centers[labels].reshape(image.shape).astype('uint8')

    return Image.fromarray(q)


def nextpow2(x):
    x = int(x)
    return 1 << (x-1).bit_length()


def ispow2(x):
    x = int(x)
    return x > 0 and (x & (x - 1))
