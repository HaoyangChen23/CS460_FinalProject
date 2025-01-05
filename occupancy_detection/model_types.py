from enum import Enum
from occupancy_detection.CNN100 import CNN_100
from occupancy_detection.ResNet34 import ResNetClassifier
from occupancy_detection.DenseNet import DenseNetClassifier

"""
Model types for cleaner handling in training and eval files
"""

class ModelType(Enum):
    CNN_100 = "CNN_100"
    RESNET34 = "RESNET34"
    INCEPTION = "INCEPTION"
    DENSENET = "DENSENET"
    EFFICIENTNET = "EFFICIENTNET"


ARGPARSE_TO_TYPE = {
    "cnn": ModelType.CNN_100,
    "resnet": ModelType.RESNET,
     "densenet": ModelType.DENSENET
}


def load_model(model_type: ModelType):
    """
    Loads in a base model given a specific ModelType
    """
    if model_type == ModelType.CNN_100:
        return CNN_100()
    if model_type == ModelType.RESNET34:
        return ResNetClassifier()
    if model_type == ModelType.DENSENET:
        return DenseNetClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    