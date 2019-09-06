import torch
import utils.dataset_ssr as ds
import utils.dataset as ds_tr
from myargs import args
from tqdm import tqdm
import os
import utils.eval as val
import utils.networks as networktools
import segmentation_models_pytorch as smp
import models.losses as losses
from models import optimizers
from utils import preprocessing

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
    optimizer = optimizers.optimfn(args.optim, model)

    model, optimizer, start_epoch = networktools.continue_train(model, optimizer,
                                                                args.train_model_pth, args.continue_train)
    ' losses '
    args.cls_ratios = preprocessing.cls_ratios_ssr('data/same_sized_regions/train')
    cls_weights = 1.0/args.cls_ratios
    cls_weights /= cls_weights.max()

    params = {
        'reduction': 'mean',
        'alpha': torch.Tensor(cls_weights),
        'gamma': 2,
        'scale_factor': 1/8,
        'ratio': 0.25,
        'ignore_index': 0,
    }
    lossfn = losses.lossfn(args.loss, params=params).cuda()
    lossfn_dice = losses.lossfn('dice', params=params).cuda()
    ' datasets '
    iterator_train = ds.GenerateIterator('data/same_sized_regions/train')
    iterator_val = ds.GenerateIterator('data/same_sized_regions/val', eval=True)

    validation_params = {
        'ph': args.tile_h * args.scan_resize,  # patch height (y)
        'pw': args.tile_w * args.scan_resize,  # patch width (x)
        'sh': args.tile_stride_h,  # slide step (dy)
        'sw': args.tile_stride_w,  # slide step (dx)
    }
    #iterator_val = ds_tr.Dataset_wsis(args.raw_val_pth, validation_params)

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
        lossfn = lossfn.cuda()

    rev_norm = preprocessing.NormalizeInverse(args.dataset_mean, args.dataset_std)
    from torchvision.utils import make_grid
    def show(img, ep, batch_it):
        import matplotlib.pyplot as plt
        from PIL import Image
        import numpy as np
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        npimg = Image.fromarray((255*npimg).astype(np.uint8))

        os.makedirs('data/res/{}/'.format(ep), exist_ok=True)
        npimg.save('data/res/{}/{}.png'.format(ep, batch_it))

    ' current run train parameters '
    print(args)

    for epoch in range(start_epoch, 1+args.num_epoch):
        sum_loss_cls = 0
        progress_bar = tqdm(iterator_train, disable=False)

        for batch_it, (image, label) in enumerate(progress_bar):
            if cuda:
                image = image.cuda()
                label = label.cuda()

            # pass images through the network (cls)
            pred_src = model(image)

            loss_cls = lossfn(pred_src, label)  #+lossfn_dice(pred_src, label)

            sum_loss_cls = sum_loss_cls + loss_cls.item()

            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()

            progress_bar.set_description('ep. {}, cls loss: {:.3f}'.format(epoch, sum_loss_cls/(batch_it+args.epsilon)))

        ' test model accuracy '
        if 0 and epoch >= 1:  #args.validate_model > 0 and epoch % args.validate_model == 0:
            val.predict_wsis(model, iterator_val, epoch)

        if epoch >= 1:
            model.eval()
            with torch.no_grad():

                total_acc = 0
                binary_acc = 0
                for batch_it, (image, label) in enumerate(iterator_val):

                    image = image.cuda()
                    label = label.cuda()

                    pred_src = model(image)
                    pred_ = torch.argmax(pred_src, 1)

                    m = torch.cat((label, pred_), dim=-2)
                    m = torch.eye(args.num_classes)[m][..., 1:].permute(0, 3 , 1 ,2)
                    for ij in range(image.size(0)):
                        image[ij, ...] = rev_norm(image[ij, ...])
                    m = torch.cat((image, m.cuda()), dim=-2)

                    show(make_grid(m.cpu()), epoch, batch_it)

                    total_acc += torch.mean((pred_ == label).float())
                    binary_acc += torch.mean(((pred_ > 0) == (label > 0)).float())

                print('Acc {:.2f}, binary acc {:.2f}'.format(total_acc/batch_it, binary_acc/batch_it))

            model.train()

        if args.save_models > 0 and epoch % args.save_models == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': args
            }
            torch.save(state, '{}/model_{}_{}.pt'.format(args.model_save_pth, args.arch_encoder, epoch))


if __name__ == "__main__":
    train()
