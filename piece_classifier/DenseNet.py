"""
DenseNet model for piece classification
"""

import torch.nn.functional as F
import functools
import sys 
import os 
import torch
from torch import nn
from torchvision import models

class DenseNetClassifier(nn.Module):
    """DenseNet Classifier for piece classification
    
    The final layer of DenseNet's classification head is replaced with a fully-connected layer that has 12
    output units to classify each (color, piece) combination.
    """
    
    input_size = 224, 224  # DenseNet default input size
    pretrained = True

    def __init__(self, weights_path=None):
        super().__init__()
        # 首先加载带有ImageNet预训练权重的模型
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        # Replace classification head
        n = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n, 12)  # 12 classes: one for each (color, piece) combination
        self.params = {
            "head": list(self.model.classifier.parameters())
        }
        
        # 如果提供了微调权重路径，则加载微调权重
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            self.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)

