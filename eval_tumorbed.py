import utils.dataset as ds
from myargs import args
import os
import utils.eval as val
import utils.networks as networktools
import segmentation_models_pytorch as smp
from models import optimizers
from models.models import Classifier, Regressor

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids


def _eval():

    args.val_save_pth = '/home/ozan/remoteDir/Tumor Bed Detection Results/Cellularity_ozan'
    args.raw_val_pth = '/home/ozan/remoteDir/'

    ' model setup '
    def activation(x):
        x
    model = eval('smp.'+args.model_name)(
        args.arch_encoder,
        encoder_weights='imagenet',
        classes=args.num_classes,
        activation=activation,
    )
    model.classifier = Classifier(model.encoder.out_shapes[0], args.num_classes)
    model.regressor = Regressor(model.encoder.out_shapes[0], 1)

    model, _, _ = networktools.continue_train(
        model,
        optimizers.optimfn(args.optim, model),
        args.eval_model_pth,
        True
    )

    ' datasets '
    validation_params = {
        'ph': args.tile_h,  # patch height (y)
        'pw': args.tile_w,  # patch width (x)
        'sh': args.tile_stride_h,     # slide step (dy)
        'sw': args.tile_stride_w,     # slide step (dx)
    }
    iterator_test = ds.Dataset_wsis(args.raw_val_pth, validation_params)

    model = model.cuda()

    val.predict_tumorbed(model, iterator_test, 0)


if __name__ == "__main__":
    _eval()
