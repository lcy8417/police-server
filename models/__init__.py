from image_search import Token
from pattern_extract import PatternExtractionModel
from inpainting.inpainting import InpaintingModel
from denoising import DenoisingModel
from background import BackgroundDeletionModel


class Models:
    model_dict = {
        # image search models
        "image_search": {
            "Token": Token(
                ckpt="weights/image/checkpoints/epoch270.pth",
            ).to("cuda"),
        },
        # pattern extraction models
        "pattern_extract": PatternExtractionModel(),
        # denoising models
        "denoising": DenoisingModel(),
        # inpainting models
        "inpainting": InpaintingModel(),
        # background deletion models
        "background": BackgroundDeletionModel(),
    }

    @classmethod
    def get_all_models(cls):
        return cls.model_dict

    @classmethod
    def get_model(cls, task):
        return cls.model_dict.get(task)


models = Models.get_all_models()
