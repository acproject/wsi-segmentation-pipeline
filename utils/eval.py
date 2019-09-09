import torch
import torch.nn.functional as F
import os
from myargs import args
from utils import preprocessing
import numpy as np
from PIL import Image
import cv2
import gc
import openslide
from mahotas import bwperim
from skimage.morphology.convex_hull import convex_hull_image as chull
import glob
import torchvision
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def predict_wsis(model, dataset, ep):
    '''
	given directory svs_path,
	current model goes through each
	wsi (svs) and generates a
	prediction mask embedded onto
	wsi.
	sequential: tiles images and
	generates batches of size 1
	parallel: uses preallocated
	dataset and batches>>1
	'''

    os.makedirs('{}/{}'.format(args.val_save_pth, ep), exist_ok=True)

    model.eval()

    with torch.no_grad():
        ' go through each svs and make a pred. mask'
        ious_tb = 0
        for key in dataset.wsis:
            'create prediction template'
            pred = np.zeros(
                (args.num_classes, *dataset.wsis[key]['iterator'].dataset.scan.level_dimensions[args.scan_level][::-1]),
                dtype=np.float)
            'slide over wsi'
            for batch_x, batch_y, batch_image in dataset.wsis[key]['iterator']:
                batch_image = batch_image.cuda()

                pred_src = model(batch_image)
                if args.scan_resize != 1:
                    pred_src = torch.nn.functional.interpolate(pred_src,
                                                               (args.tile_h * args.scan_resize,
                                                                args.tile_w * args.scan_resize))
                pred_src = pred_src.cpu().numpy()

                for bj in range(batch_image.size(0)):
                    tile_x, tile_y = int(batch_x[bj]), int(batch_y[bj])
                    pred[:, tile_y:tile_y + dataset.params.ph, tile_x:tile_x + dataset.params.pw] += pred_src[bj]

            'post process wsi (throw out non-foreground tissue)'
            scan = openslide.OpenSlide(dataset.wsis[key]['wsipath'])
            mask = Image.open(dataset.wsis[key]['wsipath'] + '_find_nuclei.png')
            mask = np.asarray(mask)

            'downsample pred'
            pred_ = np.zeros((args.num_classes, *scan.level_dimensions[2][::-1]))
            for ij in range(args.num_classes):
                pred_[ij, ...] = cv2.resize(pred[ij, ...], scan.level_dimensions[2])
            pred = pred_
            del pred_

            'calculate score'
            if os.path.exists(dataset.wsis[key]['wsipath'] + '_mask.png'):

                gt = Image.open(dataset.wsis[key]['wsipath'] + '_mask.png')
                gt = gt.resize(pred.shape[1:][::-1])
                gt = np.array(gt)

                p = np.argmax(pred, 0)

                '''
				get tumor bed
				erosion: to remove small possibly
				miss-regions
				
				'''
                # tb = (p.astype(np.uint8) == 3).astype(np.uint8)
                tb = (p.astype(np.uint8) >= 2).astype(np.uint8)
                tb = cv2.morphologyEx(tb, cv2.MORPH_OPEN, np.ones((20, 20), dtype=np.uint8))
                tb_pred = chull(tb)

                tb = bwperim(tb_pred).astype(np.uint8)
                tb = cv2.dilate(tb, np.ones((20, 20), dtype=np.uint8), iterations=1)
                tb = np.nonzero(tb)
                '''
				use gt tumor bed
				'''
                tb_pth = dataset.wsis[key]['wsipath'] + '_tumor_bed.png'
                if os.path.exists(tb_pth):
                    tb_gt = Image.open(tb_pth).convert('L')
                    tb_gt = (np.array(tb_gt) > 0).astype(np.uint8)

                    iou_tb = (tb_gt * tb_pred).sum() / (args.epsilon + (tb_gt | tb_pred).sum())
                    ious_tb += iou_tb

                acc = (p == gt)
                acc = acc[gt > 0]
                acc = np.mean(acc)
                s = 1 - np.sum(np.abs(p - gt)) / np.sum(
                    np.maximum(np.abs(gt - 0), np.abs(gt - 3.0)) * (1 - (1 - (p > 0)) * (1 - gt > 0)))

                p = mask * p
                acc_masked = (p == gt)
                acc_masked = acc_masked[gt > 0]
                acc_masked = np.mean(acc_masked)
                s_masked = 1 - np.sum(np.abs(p - gt)) / np.sum(
                    np.maximum(np.abs(gt - 0), np.abs(gt - 3.0)) * (1 - (1 - (p > 0)) * (1 - gt > 0)))

                'only detect foreground vs background'
                iou_fg = ((p > 0) * (gt > 0)).sum() / (args.epsilon + ((p > 0) | (gt > 0)).sum())

                print('{}, '
                      '{:.3f}({:.3f}),'
                      ' {:.3f}({:.3f}),'
                      ' {:.3f},'
                      ' tb iou: {:.3f} '.format(
                    dataset.wsis[key]['wsipath'].split('/')[-1],
                    s_masked, s,
                    acc_masked, acc,
                    iou_fg,
                    iou_tb if os.path.exists(tb_pth) else -1
                )
                )
                del p
                del gt

            'save color mask'
            pred_image = np.expand_dims(mask, -1) * preprocessing.pred_to_mask(pred)
            pred_image[tb] = [255, 255, 255]
            pred_image = Image.fromarray(pred_image)
            pred_image.resize((dataset.wsis[key]['scan'].level_dimensions[2][0] // 2,
                               dataset.wsis[key]['scan'].level_dimensions[2][1] // 2)). \
                save('{}/{}/{}_{}.png'.format(args.val_save_pth, ep, key, args.tile_stride_w))

            del pred
            del pred_image

        print('Average tb iou: {:.3f}'.format(ious_tb / len(dataset.wsis)))

    model.train()


def predict_tumorbed(model, dataset, ep, mode='seg'):
    '''
    given directory svs_path,
    current model goes through each
    wsi (svs) and generates a
    tumor bed.
    '''

    os.makedirs('{}/{}'.format(args.val_save_pth, ep), exist_ok=True)

    ious_tb = 0

    model.eval()
    model.regressor.eval()
    model.classifier.eval()
    model.decoder.eval()

    with torch.no_grad():
        ' go through each svs and make a pred. mask'
        with tqdm(enumerate(dataset.wsis)) as t:
            for wj, key in t:

                t.set_description('Generating heatmaps of tumor beds from wsis.. {:d}/{:d}'.format(1 + wj, len(dataset.wsis)))

                scan = dataset.wsis[key]['scan']

                'create prediction template'
                wsi_yx_dims = scan.level_dimensions[2][::-1]
                pred = np.zeros((args.num_classes, *wsi_yx_dims), dtype=np.float)
                'downsample multiplier'
                m = scan.level_downsamples[args.scan_level]/scan.level_downsamples[2]
                dx, dy = int(m * dataset.params.pw), int(m * dataset.params.ph)

                'slide over wsi'

                for batch_x, batch_y, batch_image in dataset.wsis[key]['iterator']:

                    batch_image = batch_image.cuda()

                    # preprocessing.display_tensor_images_on_grid(batch_image).save('{}.png'.format(np.random.randint(0, 1000)))

                    encoding = model.encoder(batch_image)
                    if mode == 'cls':
                        pred_src = model.classifier(encoding[0])
                    if mode == 'seg':
                        pred_src = model.decoder(encoding)

                    if args.scan_resize != 1:
                        pred_src = F.interpolate(
                            pred_src,
                            (args.tile_h * args.scan_resize, args.tile_w * args.scan_resize)
                        )

                    pred_src = pred_src.cpu().numpy()

                    while pred.ndim >= pred_src.ndim:
                        pred_src = np.expand_dims(pred_src, -1)

                    for bj in range(batch_image.size(0)):
                        tile_x, tile_y = int(m * batch_x[bj]), int(m * batch_y[bj])
                        pred[:, tile_y:tile_y + dy, tile_x:tile_x + dx] += pred_src[bj]

                pred_classes, pred_heatmap = preprocessing.threshold_probs(pred)

                mask = np.array(Image.open(dataset.wsis[key]['maskpath']))
                if mode == 'cls':
                    pred_heatmap = pred_heatmap[1, ...]
                if mode == 'seg':
                    pred_heatmap = pred_heatmap[2, ...]+pred_heatmap[3, ...]  # (pred_heatmap[2, ...] + pred_heatmap[3, ...])

                pred_heatmap = mask * pred_heatmap

                'save heatmap'
                pred_heatmap = np.uint8(255 * pred_heatmap)
                Image.fromarray(pred_heatmap).save('{}/{}/{}_{}_heatmap.png'.format(args.val_save_pth, ep, key, args.tile_stride_w))

                '''
                get tumor bed
                erosion: to remove small possibly
                miss-regions
                '''

                '''
                pred_classes = dataset.wsis[key]['mask'] * pred_classes
    
                tb = (pred_classes.astype(np.uint8) > 0).astype(np.uint8)
                tb = cv2.morphologyEx(tb, cv2.MORPH_OPEN, np.ones((20, 20), dtype=np.uint8))
                tb_pred = chull(tb)
    
                tb = bwperim(tb_pred).astype(np.uint8)
                tb = cv2.dilate(tb, np.ones((20, 20), dtype=np.uint8), iterations=1)
                tb = np.nonzero(tb)
                '''

                'save color mask'
                '''
                pred_image = pred_classes[..., np.newaxis].repeat(axis=-1, repeats=3)
                pred_image = pred_image.astype(np.uint8)
                pred_image[tb] = [255, 255, 255]
                pred_image = Image.fromarray(pred_image)
                pred_image.resize((scan.level_dimensions[-1][0] // 2,
                                   scan.level_dimensions[-1][1] // 2)). \
                    save('{}/{}/{}_{}_tb.png'.format(args.val_save_pth, ep, key, args.tile_stride_w))
                '''


                'save overlay mask'
                pred_image = scan.read_region((0, 0), 2, scan.level_dimensions[2]).convert('RGB')
                pred_image = np.asarray(pred_image).astype(np.uint8)

                pred_image = pred_image*0.75 + 255 * np.repeat(np.expand_dims(pred_heatmap > 255 * 0.99, -1), repeats=3, axis=-1)*0.25
                pred_image = Image.fromarray(np.uint8(pred_image))
                pred_image.save('{}/{}/{}_{}_overlay.png'.format(args.val_save_pth, ep, key, args.tile_stride_w))



                '''
                pred_image[tb] = [0, 0, 0]
                pred_image = Image.fromarray(pred_image)
                pred_image.resize((scan.level_dimensions[-1][0] // 2,
                                   scan.level_dimensions[-1][1] // 2)). \
                    save('{}/{}/{}_{}_overlay.png'.format(args.val_save_pth, ep, key, args.tile_stride_w))
                '''

                del pred
                del pred_image
                del mask
                dataset.wsis[key] = None

        print('Average tb iou: {:.3f}'.format(ious_tb / len(dataset.wsis)))

    model.train()


def predict_reg(model, dataset, ep):
    rev_norm = preprocessing.NormalizeInverse(args.dataset_mean, args.dataset_std)

    model.eval()
    model.regressor.eval()
    model.classifier.eval()
    model.decoder.eval()

    with torch.no_grad():

        preds, gts = [], []

        image_num = 0

        for batch_it, (image, label, is_cls, is_reg, is_seg, cls_code) in enumerate(dataset):

            image = image.cuda()

            pred_cls_ = None
            augmented_set = [
                image,
                image.transpose(2, 3),
                image.flip(2),
                image.transpose(2, 3).flip(3)
            ]
            for image_ in augmented_set:
                encoding = model.encoder(image_)
                pred_cls = model.regressor(encoding[0])
                pred_seg = model.decoder(encoding)
                if pred_cls_ is None:
                    pred_cls_ = pred_cls.view(-1)
                else:
                    pred_cls_ += pred_cls.view(-1)

            pred_seg = torch.argmax(pred_seg, 1).cpu().numpy()
            image_rev = image_
            for ij in range(image_rev.size(0)):
                patch = rev_norm(image_rev[ij, ...])
                patch = patch.cpu().numpy().transpose(1, 2, 0)
                mask = np.expand_dims(pred_seg[ij], -1).repeat(3, -1)
                mask[..., [0, 2]] = 0
                patch = patch * 0.75 + mask * 0.25
                patch = Image.fromarray((patch * 255).astype(np.uint8))

                image_num += 1
                patch.save('data/cell_seg/{}.png'.format(image_num))

            pred_cls = pred_cls_ / len(augmented_set)

            preds.extend(pred_cls.cpu().numpy())
            gts.extend(cls_code.numpy())

    preds, gts = np.asarray(preds), np.asarray(gts)

    print('Ep. {}, '
          'l1 {:.3f},'
          ' mse {:.3f}, '.format(
        ep,
        np.mean(np.abs(preds - gts)),
        np.mean((preds - gts) ** 2),
    ))

    model.train()


def predict_breastpathq(model, ep, dataset_path, label_csv_path):
    import csv

    image_aug = preprocessing.standard_augmentor(True)
    image_resize = torchvision.transforms.Resize((args.tile_h, args.tile_w))

    model.eval()
    model.regressor.eval()
    model.classifier.eval()
    model.decoder.eval()

    with torch.no_grad():

        with open('Ozan_Results_{}.csv'.format(ep), 'w', newline='') as csv_write:

            fieldnames = ['slide', 'rid', 'p']
            writer = csv.DictWriter(csv_write, fieldnames=fieldnames)
            writer.writeheader()

            with open('{}'.format(label_csv_path)) as csv_file:

                csv_reader = csv.reader(csv_file, delimiter=',')
                next(csv_reader)

                for row in csv_reader:

                    image_id = int(row[0])
                    region_id = int(row[1])

                    pth = '{}/{}_{}.tif'.format(dataset_path, image_id, region_id)

                    image = Image.open(pth).convert('RGB')
                    image = image_resize(image)
                    image = image_aug(image)
                    image = image.cuda()
                    image = image.unsqueeze_(0)

                    pred_cls_ = None
                    augmented_set = [
                        image,
                        image.transpose(2, 3),
                        image.flip(2),
                        image.transpose(2, 3).flip(3)
                    ]
                    for image_ in augmented_set:
                        encoding = model.encoder(image_)
                        pred_cls = model.regressor(encoding[0])
                        if pred_cls_ is None:
                            pred_cls_ = pred_cls.view(-1)
                        else:
                            pred_cls_ += pred_cls.view(-1)

                    pred_cls = pred_cls_ / len(augmented_set)

                    pred_cls = pred_cls.cpu().numpy()[0]
                    pred_cls = np.maximum(pred_cls, 0.0)
                    pred_cls = np.minimum(pred_cls, 1.0)

                    writer.writerow({fieldnames[0]: image_id, fieldnames[1]: region_id, fieldnames[2]: pred_cls})


def predict_cls(model, dataset, ep):
    model.eval()
    model.regressor.eval()
    model.classifier.eval()
    model.decoder.eval()

    with torch.no_grad():

        preds, gts = [], []

        for batch_it, (image, label, is_cls, is_reg, is_seg, cls_code) in enumerate(dataset):

            image = image.cuda()
            is_cls = is_cls.type(torch.bool).cuda()

            encoding = model.encoder(image)

            if torch.nonzero(is_cls).size(0) > 0:
                pred_cls = model.classifier(encoding[0][is_cls])
                pred_cls = torch.argmax(pred_cls, 1)

            preds.extend(pred_cls.cpu().numpy())
            gts.extend(cls_code.numpy())

    preds, gts = np.asarray(preds), np.asarray(gts)

    print('Ep. {}, '
          'acc {:.3f},'
          'f1 {:.3f}'.format(
        ep,
        np.mean(preds == gts),
        f1_score(gts, preds),
    ))

    model.train()
