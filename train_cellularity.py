import torch
import utils.dataset as ds
from myargs import args
from tqdm import tqdm
import os
import utils.eval as val
import utils.networks as networktools
import segmentation_models_pytorch as smp
import models.losses as losses
from models import optimizers
from models.models import Classifier, Regressor, ReverseLayerF
from utils import preprocessing

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids


def train():

    ' model setup '
    def activation(x):
        x
    model = eval('smp.'+args.model_name)(
        args.arch_encoder,
        encoder_weights='imagenet',
        classes=4,
        activation=activation,
    )
    model.classifier = Classifier(model.encoder.out_shapes[0], args.num_classes)
    model.regressor = Regressor(model.encoder.out_shapes[0], 1)
    optimizer = optimizers.optimfn(args.optim, model)

    model, optimizer, start_epoch = networktools.continue_train(
        model,
        optimizer,
        args.train_model_pth,
        args.continue_train
    )

    ' losses '
    cls_weights_cls, cls_weights_seg = preprocessing.cls_weights(args.train_image_pth)

    params = {
        'reduction': 'mean',
        'alpha': torch.Tensor(cls_weights_cls),
        'xent_ignore': -1,
    }
    lossfn_cls = losses.lossfn('xent', params).cuda()
    lossfn_reg = losses.lossfn('mse', params).cuda()

    params = {
        'reduction': 'mean',
        'alpha': torch.Tensor(cls_weights_seg),
        'xent_ignore': -1,
    }
    lossfn_seg = losses.lossfn('xent', params).cuda()

    ' datasets '
    iterator_train = ds.GenerateIterator(args.train_image_pth, duplicate_dataset=1)
    iterator_val = ds.GenerateIterator(args.val_image_pth, eval=True)

    model = model.cuda()

    ' current run train parameters '
    print(args)

    for epoch in range(start_epoch, 1+args.num_epoch):

        #if epoch == 2:
        #    iterator_train = ds.GenerateIterator(args.train_image_pth, duplicate_dataset=1)

        sum_loss_cls, sum_loss_reg, sum_loss_seg = 0, 0, 0

        progress_bar = tqdm(iterator_train, disable=False)

        for batch_it, (image, label, is_cls, is_reg, is_seg, cls_code) in enumerate(progress_bar):

            image = image.cuda()
            label = label.cuda()
            is_cls = is_cls.type(torch.bool).cuda()
            cls_code = cls_code.cuda()

            encoding = model.encoder(image)

            loss = 0

            if torch.nonzero(is_cls).size(0) > 0:
                pred_cls = model.classifier(encoding[0][is_cls])
                loss_cls = lossfn_cls(pred_cls, cls_code[is_cls].long())  # 154/2394*
                sum_loss_cls += loss_cls.item()
                loss += loss_cls

            if torch.nonzero(is_reg).size(0) > 0:
                pred_reg = model.regressor(encoding[0][is_reg]).view(-1)
                loss_reg = lossfn_reg(pred_reg, cls_code[is_reg].float())
                sum_loss_reg += loss_reg.item()
                loss += loss_reg

            if torch.nonzero(is_seg).size(0) > 0:
                seg_input = [x[is_seg] for x in encoding]
                pred_seg = model.decoder(seg_input)
                loss_seg = lossfn_seg(pred_seg, label[is_seg].long())
                sum_loss_seg += loss_seg.item()
                loss += loss_seg

            if not(isinstance(loss, int)) and loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            progress_bar.set_description('ep. {},'
                                         ' losses; cls: {:.2f},'
                                         ' reg: {:.2f},'
                                         ' seg: {:.2f}'.format(
                epoch,
                sum_loss_cls/(batch_it+args.epsilon),
                sum_loss_reg/(batch_it+args.epsilon),
                sum_loss_seg/(batch_it+args.epsilon)
            ))


        ' test model accuracy '
        if args.validate_model > 0 and epoch % args.validate_model == 0:
            #val.predict_reg(model, iterator_val, epoch)
            #val.predict_cls(model, iterator_val, epoch)

            dataset_path = '/home/ozan/Downloads/breastpathq-test/test_patches'
            label_csv_path = '/home/ozan/Downloads/breastpathq-test/Tony_Results.csv'
            # val.predict_breastpathq(model, epoch, dataset_path, label_csv_path)

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
