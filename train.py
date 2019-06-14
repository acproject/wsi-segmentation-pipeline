import torch
from torch import optim
import utils.dataset as ds
from myargs import args
from tqdm import tqdm
import os
import utils.eval as val
import utils.networks as networktools
import segmentation_models_pytorch as smp
import models.losses as losses

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids


def train():

    ' model setup '
    def activation(x):
        x
    model = eval('smp.'+args.model_name)(
        args.arch_encoder,
        encoder_weights='imagenet',
        classes=args.num_classes,
        activation=activation,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    model, optimizer, start_epoch = networktools.continue_train(model, optimizer,
                                                                args.train_model_pth, args.continue_train)
    ' losses '
    lossfn = losses.lossfn(args.loss).cuda()
    ' datasets '
    validation_params = {
        'ph': args.tile_h,  # patch height (y)
        'pw': args.tile_w,  # patch width (x)
        'sh': args.tile_stride_h,     # slide step (dy)
        'sw': args.tile_stride_w,     # slide step (dx)
    }
    iterator_train = ds.GenerateIterator(args.train_image_pth)
    iterator_val = ds.Dataset_wsis(args.raw_val_pth, validation_params)

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
        lossfn = lossfn.cuda()

    ' current run train parameters '
    print(args)

    for epoch in range(start_epoch, 1+args.num_epoch):
        sum_loss_cls = 0
        progress_bar = tqdm(iterator_train, disable=False)
        for batch_it, (image, label, mask) in enumerate(progress_bar):
            if cuda:
                image = image.cuda()
                label = label.cuda()
                mask = mask.cuda()

            # pass images through the network (cls)
            pred_src = model(image)

            loss_cls = lossfn(pred_src, label)
            loss_cls = loss_cls.mean()

            sum_loss_cls = sum_loss_cls + loss_cls.item()

            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()

            progress_bar.set_description('ep. {}, cls loss: {:.3f}'.format(epoch, sum_loss_cls/(batch_it+args.epsilon)))

        ' test model accuracy '
        if epoch % 10 == 0:
            val.predict_wsi(model, iterator_val, epoch)

        if args.save_models:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, '{}/model_{}.pt'.format(args.model_save_path, epoch))


if __name__ == "__main__":
    train()
