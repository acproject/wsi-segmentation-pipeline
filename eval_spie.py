import utils.eval as val
import utils.networks as networktools
import segmentation_models_pytorch as smp
from models import optimizers
from myargs import args
from models.models import Classifier, Regressor

' model setup '
def activation(x):
    x
model = eval('smp.' + args.model_name)(
    args.arch_encoder,
    encoder_weights='imagenet',
    classes=2,
    activation=activation,
)
model.classifier = Classifier(model.encoder.out_shapes[0], args.num_classes)
model.regressor = Regressor(model.encoder.out_shapes[0], 1)
optimizer = optimizers.optimfn(args.optim, model)

model, _, _ = networktools.continue_train(
    model,
    optimizer,
    args.eval_model_pth,
    True)

model = model.cuda()

dataset_path = '/home/ozan/Downloads/breastpathq-test/test_patches'
label_csv_path = '/home/ozan/Downloads/breastpathq-test/Results.csv'
val.predict_breastpathq(model, 391, dataset_path, label_csv_path)
