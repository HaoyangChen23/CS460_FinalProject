from enum import Enum
from piece_classifier.CNN100 import CNN_100
from piece_classifier.ResNet34 import ResNetClassifier
from piece_classifier.DenseNet import DenseNetClassifier

class ModelType(Enum):
    CNN_100 = "CNN_100"
    RESNET34 = "RESNET34"
    DENSENET = "DENSENET"


ARGPARSE_TO_TYPE = {
    "cnn": ModelType.CNN_100,
    "resnet34": ModelType.RESNET34,
    "densenet": ModelType.DENSENET,
}


def load_model(model_type: ModelType):
    if model_type == ModelType.CNN_100:
        return CNN_100()
    if model_type == ModelType.RESNET34:
        return ResNetClassifier()
    if model_type == ModelType.DENSENET:
        return DenseNetClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
