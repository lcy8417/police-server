import timm
from torch import nn


class DenseNet169(nn.Module):
    def __init__(self, num_classes=318, layer='169'):
        super(DenseNet169, self).__init__()

        self.backbone = timm.create_model(f'densenet{layer}', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)
