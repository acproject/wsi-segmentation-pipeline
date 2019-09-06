'''
train patch level
same size region
network
'''
import numpy as np
import torch
import utils.dataset_ssr as ds
from myargs import args
from tqdm import tqdm
import os
import utils.eval as val
import utils.networks as networktools
import pretrainedmodels
import models.losses as losses
from models import optimizers
from utils import preprocessing
from sklearn.metrics import confusion_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids


def train():

    ' model setup '
    model = pretrainedmodels.__dict__[args.arch_encoder](num_classes=1000, pretrained='imagenet')
    model.last_linear = torch.nn.Linear(model.last_linear.in_features, args.num_classes)

    optimizer = optimizers.optimfn(args.optim, model)

    model, optimizer, start_epoch = networktools.continue_train(model, optimizer,
                                                                args.train_model_pth, args.continue_train)
    ' losses '
    args.cls_ratios = preprocessing.cls_ratios_ssr('data/ssr/train', option='classification')
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
    ' datasets '
    iterator_train = ds.GenerateIterator_cls('data/ssr/train')
    iterator_val = ds.GenerateIterator_cls('data/ssr/val', eval=True)

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
        lossfn = lossfn.cuda()

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

            loss_cls = lossfn(pred_src, label)

            sum_loss_cls = sum_loss_cls + loss_cls.item()

            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()

            progress_bar.set_description('ep. {}, cls loss: {:.3f}'.format(epoch, sum_loss_cls/(batch_it+args.epsilon)))

        ' test model accuracy '
        if epoch >= 1:
            model.eval()
            with torch.no_grad():
                preds, gts = [], []
                for batch_it, (image, label) in enumerate(iterator_val):

                    image = image.cuda()

                    pred_src = model(image)
                    pred_ = torch.argmax(pred_src, 1)

                    preds.extend(pred_.cpu().numpy())
                    gts.extend(label.numpy())

                preds = np.asarray(preds)
                gts = np.asarray(gts)

                total_acc = np.mean(gts == preds)

                cfs = confusion_matrix(gts, preds)
                cls_acc = np.diag(cfs / cfs.sum(1))
                cls_acc = ['{:.2f}'.format(el) for el in cls_acc]

                print('Ep. {},'
                      ' Acc {:.2f},'
                      'Classwise acc. {}'.format(
                    epoch,
                    total_acc,
                    cls_acc
                ))

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
