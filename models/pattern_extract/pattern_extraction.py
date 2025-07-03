import os
import torch
from torch import nn
import timm
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import pandas as pd

from .ml_decoder import add_ml_decoder_head


class PatternExtractionModel:
    def __init__(self):
        self.root_path = "./weights/pattern"
        self.cuda = torch.device("cuda")
        self.input_shape = (448, 448)
        self.parts = ["top", "mid", "bottom", "all"]  # 사용할 모델 파트 지정

        self.models = self.load_models()

        # 코드 데이터 불러오기
        pattern_path = "./data/scas_pattern_code_v2.0.xlsx"
        df = pd.read_excel(pattern_path)
        self.code_data = {i: code for i, code in enumerate(df["code"].tolist())}

        # 이미지 전처리
        self.transform = transforms.Compose(
            [
                ResizeKeepAspectRatio(self.input_shape),
                transforms.CenterCrop(self.input_shape),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load_models(self):
        models = {}
        test_list = []
        for type in ["crime", "shoes"]:
            for part in self.parts:
                model = ConvNextV2().to(self.cuda)
                weight_path = f"{self.root_path}/{type}_{part}.pt"
                # weight_path = os.path.join(self.root_path, f"{type}_{part}.pt")
                test = torch.load(weight_path, map_location=self.cuda)
                model.load_state_dict(test, strict=True)
                test_list.append(test)

                model.eval()
                models[f"{type}_{part}"] = model

        return models

    def predict(
        self, image: Image.Image, type: str = "crime", part: str = "top"
    ) -> list:
        """
        Args:
            image: PIL 이미지
            part: 사용할 모델 종류 ("top", "mid", "bottom")
        Returns:
            List[str]: 예측된 코드 리스트
        """
        request_kind = f"{type}_{part}"
        assert (
            request_kind in self.models
        ), f"{request_kind} 모델이 로드되지 않았습니다."

        image_tensor = self.transform(image).unsqueeze(0).to(self.cuda)
        with torch.no_grad():
            output = self.models[request_kind](image_tensor)
            predicted = (output > 0.5).float()[0]
            indices = torch.where(predicted > 0.5)[0].cpu().tolist()

        return [self.code_data[i] for i in indices]


class ConvNextV2(nn.Module):
    model_type = {
        "tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        "base": "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "large": "convnextv2_large.fcmae_ft_in22k_in1k_384",
        "huge": "convnextv2_huge.fcmae_ft_in22k_in1k_512",
    }

    def __init__(self, num_classes=318, layer="base", decoder=False):
        super(ConvNextV2, self).__init__()
        self.root_path = "./weights/pattern"
        model_path = os.path.join(
            self.root_path, "convnextv2_base.fcmae_ft_in22k_in1k_384_weights.pth"
        )
        self.backbone = timm.create_model(
            self.model_type[layer], pretrained=False, num_classes=1000
        )

        self.backbone.load_state_dict(torch.load(model_path))
        in_features = self.backbone.get_classifier().in_features
        self.backbone.head.fc = nn.Linear(in_features, num_classes)
        self.backbone.to("cuda")
        if decoder:
            self.backbone = add_ml_decoder_head(
                self.backbone,
                num_classes=num_classes,
                num_of_groups=150,
                decoder_embedding=1024,
                zsl=0,
            )

    def forward(self, x):
        return self.backbone(x)


class ResizeKeepAspectRatio(transforms.Resize):
    def __init__(self, max_size, padding_color=(0, 0, 0)):
        super().__init__(max_size)
        self.max_size = max_size
        self.padding_color = padding_color

    def forward(self, image):
        # 원본 이미지의 크기와 비율 계산
        original_width, original_height = image.size
        # 여기서 max_size의 순서를 높이, 너비로 가정합니다.
        ratio = min(
            self.max_size[1] / original_width, self.max_size[0] / original_height
        )

        # 비율을 유지하면서 크기 조정
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        resized_image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

        # 이미지 중앙에 위치시키기 위한 패딩 계산
        padding_left = (self.max_size[1] - new_width) // 2
        padding_top = (self.max_size[0] - new_height) // 2
        padding_right = self.max_size[1] - new_width - padding_left
        padding_bottom = self.max_size[0] - new_height - padding_top

        # 패딩 추가
        padded_image = ImageOps.expand(
            resized_image,
            border=(padding_left, padding_top, padding_right, padding_bottom),
            fill=self.padding_color,
        )

        return padded_image
