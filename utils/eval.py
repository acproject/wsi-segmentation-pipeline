import torch
import os
from myargs import args
from utils import preprocessing
import numpy as np
from PIL import Image


def eval_model(model, iterator):
    with torch.no_grad():
        # calculate accuracy on validation set
        model.eval()
        overlap = []
        for image, label, mask in iterator:
            if torch.cuda.is_available():
                image = image.cuda()

            pred_src = model(image)
            pred_src = torch.argmax(pred_src, 1).cpu()

            overlap.append(torch.mean((pred_src == label).float()))

        average = torch.Tensor(overlap).mean()
        print('\n Average overlap {:.3f} \n'.format(
            average,
        ))
    model.train()


def predict_wsi(model, dataset, epoch):
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

    with torch.no_grad():
        model.eval()
        gen_mask(model, dataset, epoch)

    model.train()


def gen_mask(model, dataset, ep):

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
                pred_src = pred_src.cpu().numpy()

                for bj in range(batch_image.size(0)):
                    tile_x, tile_y = int(batch_x[bj]), int(batch_y[bj])
                    pred[:, tile_y:tile_y + self.params.ph, tile_x:tile_x + self.params.pw] += pred_src[bj]

        'save color mask'
        pred = preprocessing.pred_to_mask(pred)
        pred = Image.fromarray(pred)
        pred.resize((dataset.wsis[key]['scan'].level_dimensions[-1][0]//2,
                     dataset.wsis[key]['scan'].level_dimensions[-1][1]//2)).\
            save('{}/{}/{}_color.png'.format(args.val_save_pth, ep, key))

