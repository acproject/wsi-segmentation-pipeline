import argparse

parser = argparse.ArgumentParser()


# ------------------------ Model related parameters ------------------------ #


parser.add_argument('--model_name', default='Unet',
                    help="FPN, PSPNet, Linknet, Unet")
parser.add_argument('--arch_encoder', default='inceptionresnetv2',
                    help="architecture of net_encoder")
parser.add_argument('--num_classes', default=4, type=int,
                    help='# of classes')
parser.add_argument('--class_probs', default=[0., 0., 0., 0.], type=list,
                    help='if prediction is below this prob, '
                         'class will not be picked')


parser.add_argument('--optim', default='adam',
                    help='optimizer to use:'
                         'adam or sgd')
parser.add_argument('--lr', default=0.0001, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=0.0001, type=float,
                    help='weight decay/weights regularizer for sgd')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='momentum for sgd, beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help=' beta2 for adam')

parser.add_argument('--num_epoch', default=10000, type=int,
                    help='epochs to train for')
parser.add_argument('--start_epoch', default=1, type=int,
                    help='epoch to start training. useful if continue from a checkpoint')

parser.add_argument('--batch_size', default=10, type=int,
                    help='input batch size')
parser.add_argument('--workers', default=10, type=int,
                    help='number of data loading workers')
parser.add_argument('--gpu_ids', default='0',
                    help='which gpus to use in train/eval')


parser.add_argument('--loss', default='xent',
                    help='loss fn to use:'
                         'cross entropy (xent)'
                         'conditional entropy+cross entropy (cent)'
                         'focal loss (focal)'
                         'online hard example mining (ohem)'
                         'dice or intersection over union (dice)'
                         'jaccard loss (jaccard)'
                         'tversky loss (tversky)')

# ------------------------ Model related parameters ------------------------ #

parser.add_argument('--eval_model_pth',
                    default='data/models/model_resnet18_15.pt')
parser.add_argument('--train_model_pth',
                    default='data/models/*.pt')
parser.add_argument('--model_save_pth',
                    default='data/models')
parser.add_argument('--continue_train', default=False, type=bool,
                    help='true if continuing from saved model '
                         '(with previous optimizer params, learning rate etc)')
parser.add_argument('--save_models', default=1, type=int,
                    help='save trained models at every n^th epoch '
                         '(n =0, do not save)')
parser.add_argument('--validate_model', default=1, type=int,
                    help='validate trained models at every n^th epoch '
                         '(n =0, do not validate)')

# ------------------------ Source data paths ------------------------ #

parser.add_argument('--raw_train_pth', default='data/bach',
                    help='wsi svs and xml files')
parser.add_argument('--raw_val_pth',
                    default='data/val')
parser.add_argument('--raw_val1_pth',
                    default='data/val')

# ------------------------ Image paths ------------------------ #

parser.add_argument('--train_image_pth',
                    default='data/train')
parser.add_argument('--train_hr_image_pth',
                    default='data/train_hr',
                    help='high res training images')
parser.add_argument('--val_hr_image_pth',
                    default='data/val_hr')
parser.add_argument('--val_save_pth',
                    default='data/val/out')

# ------------------------ Tiling parameters ------------------------ #

parser.add_argument('--tile_w', default=224, type=int,
                    help='patch size width')
parser.add_argument('--tile_h', default=224, type=int,
                    help='patch size height')
parser.add_argument('--tile_stride_w', default=56, type=int,
                    help='image crop size width dx')
parser.add_argument('--tile_stride_h', default=56, type=int,
                    help='image crop size height dy')
parser.add_argument('--scan_level', default=2, type=int,
                    help='scan pyramid level')
parser.add_argument('--scan_resize', default=2, type=int,
                    help='resize the image (given 5x, value of 2 will '
                         'make it 2.5 etc). affects '
                         '(1) patch extraction in '
                         'training image generation '
                         '(2) at eval time, patches of size'
                         'scan_resize*scan_level are resized to'
                         '(tile_w, tile_h)')


# ------------------------ Image paths ------------------------ #

parser.add_argument('--dataset_mean', default=(0.485, 0.456, 0.406), type=list,
                    help='dataset stats. mean')
parser.add_argument('--dataset_std', default=(0.229, 0.224, 0.225), type=list,
                    help='dataset stats. std')


# ------------------------ System parameters ------------------------ #

parser.add_argument('--epsilon', default=1e-8, type=float,
                    help='small epsilon to add to denominator of accuracy fns')


args = parser.parse_args()

