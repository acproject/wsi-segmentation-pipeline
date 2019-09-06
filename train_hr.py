import torch
import utils.dataset_hr as ds
from myargs import args
from tqdm import tqdm
import os
import models.losses as losses
from models import optimizers
import pretrainedmodels
from torch import nn
from utils import regiontools
from utils import preprocessing
import numpy as np
from sklearn.metrics import confusion_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids


def train():

    ' model setup '
    import resnets_shift
    model = resnets_shift.resnet18(True)

    optimizer = optimizers.optimfn(args.optim, model)

    start_epoch = 1  #model, optimizer, start_epoch = networktools.continue_train(model, optimizer,
    #                                                            args.train_model_pth, args.continue_train)
    ' datasets '
    iterator_train = ds.GenerateIterator(args.train_hr_image_pth, eval=False, duplicate_dataset=1)
    iterator_val = ds.GenerateIterator(args.val_hr_image_pth, eval=True)
    iterator_val2 = ds.GenerateIterator(args.train_hr_image_pth, eval=True)
    ' losses '
    lossfn = losses.lossfn(args.loss).cuda()

    params = {'reduction': 'mean', 'alpha': torch.Tensor([1, 1, 1, 1])}
    lossfn_s = losses.lossfn(args.loss, params=params).cuda()

    model = model.cuda()
    lossfn = lossfn.cuda()

    ' current run train parameters '
    print(args)

    for epoch in range(start_epoch, 1+args.num_epoch):

        if epoch == 2:
            iterator_train = ds.GenerateIterator(args.train_hr_image_pth, duplicate_dataset=20)

        sum_loss_cls = 0
        progress_bar = tqdm(iterator_train, disable=False)

        for batch_it, (images, label) in enumerate(progress_bar):

            images = images.cuda()
            label = label.cuda()

            # pass images through the network (cls)
            pred_singles, pred_ensemble = model(images)

            label_repeat = label.view(-1, 1).repeat((1, ds.HR_NUM_CNT_SAMPLES+ds.HR_NUM_PERIM_SAMPLES)).view(-1)

            loss_cls = lossfn(pred_ensemble, label) #+ lossfn(pred_singles, label_repeat)

            sum_loss_cls = sum_loss_cls + loss_cls.item()

            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()

            progress_bar.set_description('ep. {}, cls loss: {:.3f}'.format(epoch, sum_loss_cls/(batch_it+args.epsilon)))

        ' test model accuracy '
        if epoch>=1:  #args.validate_model > 0 and epoch % args.validate_model == 0:
           regiontools.validate_dataset(model, iterator_val, epoch)

        ' going traditional '
        if 0:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=10, max_depth=50,
                                 random_state=0)

            model.eval()

            fs, gts = [], []
            with torch.no_grad():
                for batch_it, (images, label) in enumerate(iterator_train):
                    images = images.cuda()
                    f = model(images)[0]
                    fs.extend(f.cpu().numpy())
                    gts.extend(label)

            fs = np.asarray(fs)
            gts = np.asarray(gts)
            gts[gts == 2] = 1

            clf.fit(fs, gts)

            fs, gts = [], []
            with torch.no_grad():
                for batch_it, (images, label) in enumerate(iterator_val):
                    images = images.cuda()
                    f = model(images)[0]
                    fs.extend(f.cpu().numpy())
                    gts.extend(label)

            fs = np.asarray(fs)
            gts = np.asarray(gts)
            gts[gts == 2] = 1

            preds = clf.predict(fs)
            score_cls = (np.mean(preds == gts))
            cfs = confusion_matrix(gts, preds)
            cls_acc = np.diag(cfs/cfs.sum(1))  # classwise accuracy
            print('\n Epoch {}, '
                  'Validation acc. {:.2f},'
                  'Classwise acc. ({:.2f},{:.2f},{:.2f}) \n'.format(
                epoch,
                score_cls,
                *cls_acc,
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
