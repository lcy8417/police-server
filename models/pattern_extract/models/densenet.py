import timm
from torch import nn
import os
import torch

class DenseNet169(nn.Module):
    def __init__(self, num_classes=318, layer='169'):
        super(DenseNet169, self).__init__()
        model_name = f'densenet{layer}'
        self.root_path = r'C:\Users\piai\Desktop\police_lab_api_v3\weights\pattern'
        model_path = os.path.join(self.root_path, f"{model_name}_weights.pth")
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        self.backbone.load_state_dict(torch.load(model_path))
        # self.backbone = timm.create_model(f'densenet{layer}', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)
