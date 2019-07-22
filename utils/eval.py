import torch
import os
from myargs import args
from utils import preprocessing
import numpy as np
from PIL import Image
from torch import nn
import cv2
import gc
import openslide

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

    ' go through each svs and make a pred. mask'
    for key in dataset.wsis:
        'create prediction template'
        pred = np.zeros((args.num_classes, *dataset.wsis[key]['iterator'].dataset.scan.level_dimensions[args.scan_level][::-1]), dtype=np.float)
        'slide over wsi'
        for batch_x, batch_y, batch_image in dataset.wsis[key]['iterator']:

            if torch.cuda.is_available():
                batch_image = batch_image.cuda()

            pred_src = model(batch_image)
            if pred_src.size(2) != args.tile_h or pred_src.size(3) != args.tile_w:
                pred_src = torch.nn.functional.interpolate(pred_src, (args.tile_h, args.tile_w))
            pred_src = pred_src.cpu().numpy()

            for bj in range(batch_image.size(0)):
                tile_x, tile_y = int(batch_x[bj]), int(batch_y[bj])
                pred[:, tile_y:tile_y + dataset.params.ph, tile_x:tile_x + dataset.params.pw] += pred_src[bj]

        if bach:
            submitdims = {
                1: (10663, 9398),
                2: (13782, 11223),
                3: (8798, 8344),
                4: (15738, 8531),
                5: (14262, 9062),
                6: (12711, 6993),
                7: (14096, 10396),
                8: (9995, 7497),
                9: (12906, 10935),
                10: (14483, 10928)
            }
            import re
            test_id = int(re.findall('\d+', key)[0])

            '''
            threshold probs: if network is inclined to guess
            the same class, increase threshold for that class 
            '''
            pred_ = torch.softmax(torch.from_numpy(pred), dim=0).numpy()
            for cj in range(args.num_classes):
                pred_[cj, pred_[cj, ...] < args.class_probs[cj]] = 0
            pred_ = np.argmax(pred_, axis=0)

            pred__ = np.zeros(submitdims[test_id][::-1], dtype=np.uint8)
            pred__[:pred_.shape[0], :pred_.shape[1]] = pred_

            pred__ = Image.fromarray(pred__)
            pred__.save('{}/{}/{}.png'.format(args.val_save_pth, ep, test_id))

        'post process wsi (throw out non-foreground tissue)'
        scan = openslide.OpenSlide(dataset.wsis[key]['wsipath'])
        wsi = scan.get_thumbnail(scan.level_dimensions[-1])
        mask = preprocessing.find_nuclei(wsi)

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
            gt = np.asarray(gt)

            p = np.argmax(pred, 0)

            acc = (p == gt)
            acc = acc[gt > 0]
            acc = np.mean(acc)
            f1 = 0  # f1_score(m.flatten(), p.flatten(), average='weighted')
            s = 1 - np.sum(np.abs(p - gt)) / np.sum(
                np.maximum(np.abs(gt - 0), np.abs(gt - 3.0)) * (1 - (1 - (p > 0)) * (1 - gt > 0)))

            p = mask*p
            acc_masked = (p == gt)
            acc_masked = acc_masked[gt > 0]
            acc_masked = np.mean(acc_masked)
            f1_masked = 0  # f1_score(m.flatten(), p.flatten(), average='weighted')
            s_masked = 1 - np.sum(np.abs(p - gt)) / np.sum(
                np.maximum(np.abs(gt - 0), np.abs(gt - 3.0)) * (1 - (1 - (p > 0)) * (1 - gt > 0)))

            print('{}, {:.3f}({:.3f}), {:.3f}({:.3f}), {:.3f}({:.3f})'.format(dataset.wsis[key]['wsipath'].split('/')[-1],
                                              s_masked, s,
                                              f1_masked, f1,
                                              acc_masked, acc))
            del p
            del gt

        'save color mask'
        pred = preprocessing.pred_to_mask(pred)
        pred = Image.fromarray(np.expand_dims(mask, -1)*pred)
        pred.resize((dataset.wsis[key]['scan'].level_dimensions[-1][0]//2,
                     dataset.wsis[key]['scan'].level_dimensions[-1][1]//2)).\
            save('{}/{}/{}_{}.png'.format(args.val_save_pth, ep, key, args.tile_stride_w))

        del pred


def gen_heatmap(model, dataset, ep):
    '''
    idea: make network focus on
    not confident predictions.
    the problem: damn thing is sure about
    everything, right or wrong.
    '''

    os.makedirs('{}_heatmap/{}'.format(args.val_save_pth, ep), exist_ok=True)
    os.makedirs('{}/{}'.format(args.val_save_pth, ep), exist_ok=True)

    lossfn = nn.CrossEntropyLoss(reduction='none').cuda()

    ' go through each svs and make a pred. mask'
    for key in dataset.wsis:

        'create prediction template'
        loss_map = np.zeros((dataset.wsis[key]['iterator'].dataset.scan.level_dimensions[args.scan_level][::-1]), dtype=np.float)
        pred_map = np.zeros((args.num_classes, *dataset.wsis[key]['iterator'].dataset.scan.level_dimensions[args.scan_level][::-1]), dtype=np.float)
        gt = np.zeros((dataset.wsis[key]['iterator'].dataset.scan.level_dimensions[args.scan_level][::-1]), dtype=np.float)
        'slide over wsi'
        for batch_x, batch_y, batch_image, batch_gt in dataset.wsis[key]['iterator']:

            if torch.cuda.is_available():
                batch_image = batch_image.cuda()
                batch_gt = batch_gt.cuda()
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            pred_src = model(batch_image)
            loss = lossfn(pred_src, batch_gt).cpu().numpy()

            for bj in range(batch_image.size(0)):
                tile_x, tile_y = int(batch_x[bj]), int(batch_y[bj])
                loss_map[tile_y:tile_y + dataset.params.ph, tile_x:tile_x + dataset.params.pw] += loss[bj]
                pred_map[:, tile_y:tile_y + dataset.params.ph, tile_x:tile_x + dataset.params.pw] += pred_src[bj].cpu().numpy()
                gt[tile_y:tile_y + dataset.params.ph, tile_x:tile_x + dataset.params.pw] = batch_gt[bj].cpu().numpy()

        'save prediction mask'
        pred_image = Image.fromarray(preprocessing.pred_to_mask(pred_map))
        pred_image.resize((dataset.wsis[key]['scan'].level_dimensions[-1][0] // 2,
                     dataset.wsis[key]['scan'].level_dimensions[-1][1] // 2)). \
            save('{}/{}/{}_color.png'.format(args.val_save_pth, ep, key))

        'save confidence mask'
        '''
        pred_probs = torch.softmax(torch.from_numpy(pred_map), dim=0).numpy()
        pred_probs = np.max(pred_probs, 0)
        pred_probs = ((-4/3*pred_probs + 4/3)*255).astype(np.uint8)
        attention_mask = Image.fromarray(pred_probs)
        attention_mask.resize((dataset.wsis[key]['scan'].level_dimensions[-1][0]//2,
                     dataset.wsis[key]['scan'].level_dimensions[-1][1]//2)).\
            save('{}_heatmap/{}/{}_color.png'.format(args.val_save_pth, ep, key))
        '''

        '''
        ''save attention mask'
        nomatch = ((np.argmax(pred_map, 0) != gt)*(gt > 0)).astype(np.uint8)
        attention_mask = nomatch * ((loss_map-loss_map.min())/(loss_map.max()-loss_map.min()) )
        attention_mask = (255*attention_mask).astype(np.uint8)
        #pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
        attention_mask = Image.fromarray(attention_mask)
        attention_mask.resize((dataset.wsis[key]['scan'].level_dimensions[-1][0]//2,
                     dataset.wsis[key]['scan'].level_dimensions[-1][1]//2)).\
            save('{}_heatmap/{}/{}_color.png'.format(args.val_save_pth, ep, key))
        '''