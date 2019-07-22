import torch
import utils.dataset as ds
from myargs import args
import os
import utils.eval as val
import utils.networks as networktools
import segmentation_models_pytorch as smp
from models import optimizers
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids


def _eval():

    args.val_save_pth = 'data/val/out_val'
    args.batch_size = 12

    ' model setup '
    def activation(x):
        x
    model = eval('smp.'+args.model_name)(
        args.arch_encoder,
        encoder_weights='imagenet',
        classes=args.num_classes,
        activation=activation,
    )

    optimizer = optimizers.optimfn(args.optim, model)  # unused
    model, optimizer, start_epoch = networktools.continue_train(model, optimizer,
                                                                args.eval_model_pth, True)
    ' datasets '
    validation_params = {
        'ph': args.tile_h,  # patch height (y)
        'pw': args.tile_w,  # patch width (x)
        'sh': args.tile_stride_h,     # slide step (dy)
        'sw': args.tile_stride_w,     # slide step (dx)
    }
    iterator_val1 = ds.Dataset_wsis(args.raw_val1_pth, validation_params)

    if torch.cuda.is_available():
        model = model.cuda()

    'zoom in stage'
    model.eval()
    with torch.no_grad():
        val.predict_wsi(model, iterator_val1, 0)

    'save preds'
    # shutil.make_archive('predictions', 'zip', '{}/0/'.format(args.val_save_pth))


if __name__ == "__main__":
    _eval()
