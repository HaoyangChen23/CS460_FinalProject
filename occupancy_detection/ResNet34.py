"""
ResNet for occupancy detection
"""

import torch.nn.functional as F
import functools
import sys 
import os 
import torch
from torch import nn
from torchvision import models

class ResNetClassifier(nn.Module):
    """
    Baseline ResNet pretrained model for finetuning on occupancy classification
    """

    input_size = 100, 100
    pretrained = True

    def __init__(self, weights_path=None):
        super().__init__()
        # 首先加载带有ImageNet预训练权重的模型
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, 2)   
        self.params = {
            "head": list(self.model.fc.parameters())
        }
        
        # 如果提供了微调权重路径，则加载微调权重
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            self.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)
    