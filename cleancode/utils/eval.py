import torch
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


Image.MAX_IMAGE_PIXELS = None


def predict_wsis(model, dataset, ep, bach=False):
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
            pred = np.zeros((args.num_classes, *dataset.wsis[key]['iterator'].dataset.scan.level_dimensions[args.scan_level][::-1]), dtype=np.float)
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
            pred_ = np.zeros((args.num_classes, *scan.level_dimensions[-1][::-1]))
            for ij in range(args.num_classes):
                pred_[ij, ...] = cv2.resize(pred[ij, ...], scan.level_dimensions[-1])
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
                tb = (p.astype(np.uint8) == 3).astype(np.uint8)
                tb = cv2.morphologyEx(tb, cv2.MORPH_OPEN, np.ones((10, 10), dtype=np.uint8))
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

                    iou_tb = (tb_gt * tb_pred).sum()/(args.epsilon+(tb_gt | tb_pred).sum())
                    ious_tb += iou_tb

                acc = (p == gt)
                acc = acc[gt > 0]
                acc = np.mean(acc)
                s = 1 - np.sum(np.abs(p - gt)) / np.sum(
                    np.maximum(np.abs(gt - 0), np.abs(gt - 3.0)) * (1 - (1 - (p > 0)) * (1 - gt > 0)))

                p = mask*p
                acc_masked = (p == gt)
                acc_masked = acc_masked[gt > 0]
                acc_masked = np.mean(acc_masked)
                s_masked = 1 - np.sum(np.abs(p - gt)) / np.sum(
                    np.maximum(np.abs(gt - 0), np.abs(gt - 3.0)) * (1 - (1 - (p > 0)) * (1 - gt > 0)))

                'only detect foreground vs background'
                iou_fg = ((p > 0) * (gt > 0)).sum()/(args.epsilon+((p > 0) | (gt > 0)).sum())

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
            pred_image = np.expand_dims(mask, -1)*preprocessing.pred_to_mask(pred)
            pred_image[tb] = [255, 255, 255]
            pred_image = Image.fromarray(pred_image)
            pred_image.resize((dataset.wsis[key]['scan'].level_dimensions[-1][0]//2,
                         dataset.wsis[key]['scan'].level_dimensions[-1][1]//2)).\
                save('{}/{}/{}_{}.png'.format(args.val_save_pth, ep, key, args.tile_stride_w))

            del pred
            del pred_image

        print('Average tb iou: {:.3f}'.format(ious_tb/len(dataset.wsis)))

    model.train()
