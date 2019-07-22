from torch import optim
from myargs import args


def optimfn(lossname, model):
    losses = {
        'adam': optim.Adam(model.parameters(), lr=args.lr,
                           betas=(args.beta1, args.beta2), weight_decay=args.weight_decay),
        'sgd': optim.SGD(model.parameters(), lr=args.lr,
                           momentum=args.beta1, weight_decay=args.weight_decay),
    }

    return losses[lossname]
