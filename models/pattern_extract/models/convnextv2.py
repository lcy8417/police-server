import timm
from torch import nn
import os
import torch
from .ml_decoder import add_ml_decoder_head

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNextV2(nn.Module):
    model_type = {
        'tiny': 'convnextv2_tiny.fcmae_ft_in22k_in1k_384',
        'base': 'convnextv2_base.fcmae_ft_in22k_in1k_384',
        'large': 'convnextv2_large.fcmae_ft_in22k_in1k_384',
        'huge': 'convnextv2_huge.fcmae_ft_in22k_in1k_512'
    }

    def __init__(self, num_classes=318, layer='base'):
        super(ConvNextV2, self).__init__()
        self.root_path = r'C:\Users\piai\Desktop\police_lab_api_v3\weights\pattern'
        # self.backbone = timm.create_model(self.model_type[layer], pretrained=True, num_classes=num_classes)
        model_path = os.path.join(self.root_path,
                                  "convnextv2_base.fcmae_ft_in22k_in1k_384_weights.pth")
        self.backbone = timm.create_model(self.model_type[layer], pretrained=False, num_classes=1000)
        self.backbone.load_state_dict(torch.load(model_path))
        in_features = self.backbone.get_classifier().in_features
        self.backbone.head.fc = nn.Linear(in_features, num_classes)
        self.backbone.to("cuda")
    def forward(self, x):
        return self.backbone(x)


class ConvNextV2WithMLDecoder(nn.Module):
    model_type = {
        'tiny': 'convnextv2_tiny.fcmae_ft_in22k_in1k_384',
        'base': 'convnextv2_base.fcmae_ft_in22k_in1k_384',
        'large': 'convnextv2_large.fcmae_ft_in22k_in1k_384',
        'huge': 'convnextv2_huge.fcmae_ft_in22k_in1k_512'
    }

    def __init__(self, num_classes, layer='base', num_of_groups=30, decoder_embedding=1024, zsl=0):
        super(ConvNextV2WithMLDecoder, self).__init__()
        # self.backbone = timm.create_model(self.model_type[layer], pretrained=True, num_classes=num_classes)
        self.root_path = r'C:\Users\piai\Desktop\police_lab_api_v3\weights\pattern'
        model_path = os.path.join(self.root_path,
                                  "convnextv2_base.fcmae_ft_in22k_in1k_384_weights.pth")
        self.backbone = timm.create_model(self.model_type[layer], pretrained=False, num_classes=num_classes)
        self.backbone.load_state_dict(torch.load(model_path))

        # Remove the global pooling and fully connected layer, and add the MLDecoder head
        self.backbone = add_ml_decoder_head(self.backbone, num_classes=num_classes, num_of_groups=num_of_groups,
                                            decoder_embedding=decoder_embedding, zsl=zsl)

    def forward(self, x):
        return self.backbone(x)
