import torch
from torch import nn
from myargs import args


class Classifier(nn.Module):
    '''
    classifier module that is attached
    to the (end of the) encoder.
    '''
    def __init__(self, in_features):
        super(Classifier, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, args.num_classes)

    def forward(self, x):

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
