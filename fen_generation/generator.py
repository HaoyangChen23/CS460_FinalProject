import sys
import os 
from fen_generation.utils import crop_square, warp_chessboard_image, resize_image, sort_corner_points, pieces as piece_mapping
from occupancy_detection.CNN100 import CNN_100 as occupancy_cnn
from piece_classifier.CNN100 import CNN_100 as classifier_cnn
from occupancy_detection.ResNet34 import ResNetClassifier as occupancy_resnet
from piece_classifier.ResNet34 import ResNetClassifier as piece_resnet
from occupancy_detection.DenseNet import DenseNetClassifier as occupancy_densenet
from piece_classifier.DenseNet import DenseNetClassifier as piece_densenet
from board_detection.detect_board import find_corners
from PIL import Image
import chess
import torch
import torchvision.transforms as transforms
import numpy as np
import functools


class Generator:
    def __init__(self, model_type="resnet", occupancy_weights_path=None, classifier_weights_path=None):
        """
        初始化Generator类
        
        Args:
            model_type: 使用的模型类型，可选 "resnet", "densenet", "cnn"
            occupancy_weights_path: 占用检测模型的微调权重路径，如果为None则使用ImageNet预训练权重
            classifier_weights_path: 棋子分类模型的微调权重路径，如果为None则使用ImageNet预训练权重
        """
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # 根据model_type选择相应的模型类
        if model_type.lower() == "resnet":
            self.occupancy_model = occupancy_resnet(weights_path=occupancy_weights_path)
            self.classifier_model = piece_resnet(weights_path=classifier_weights_path)
        elif model_type.lower() == "densenet":
            self.occupancy_model = occupancy_densenet(weights_path=occupancy_weights_path)
            self.classifier_model = piece_densenet(weights_path=classifier_weights_path)
        elif model_type.lower() == "cnn":
            self.occupancy_model = occupancy_cnn()
            self.classifier_model = classifier_cnn()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        self.occupancy_model.to(self.device)
        self.classifier_model.to(self.device)
        self.occupancy_model.eval()
        self.classifier_model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
    def _load_model(self, model_path, model_class):
        model = model_class()
        model = torch.load(model_path, map_location=self.device)
        model.to(self.device)
        model.eval() 
        return model
        
    def classify_occupancy(self, img, turn, corners):
        warped = warp_chessboard_image(img, corners)
        squares = list(chess.SQUARES)
        square_imgs = map(functools.partial(
            crop_square, warped, turn=turn, mode="OCCUPANCY"), squares)
        square_imgs = map(lambda img: self.transform(img), square_imgs)
        square_imgs = list(square_imgs)
        square_imgs = torch.stack(square_imgs).to(self.device)
        occupancy = self.occupancy_model(square_imgs)
        occupancy = occupancy.argmax(axis=-1) == 1
        occupancy = occupancy.cpu().numpy()
        return occupancy
    
    def classify_pieces(self, img, turn, corners, occupancy: np.ndarray):
        squares = list(chess.SQUARES)
        occupied_squares = np.array(squares)[occupancy]
        warped = warp_chessboard_image(img, corners)
        piece_imgs = map(functools.partial(
            crop_square, warped, turn=turn, mode="PIECE"), occupied_squares)
        piece_imgs = map(lambda img: self.transform(img), piece_imgs)
        piece_imgs = list(piece_imgs)
        piece_imgs = torch.stack(piece_imgs).to(self.device)
        
        pieces = self.classifier_model(piece_imgs)
        pieces = pieces.argmax(axis=-1).cpu().numpy()
        
        pieces = piece_mapping[pieces]
        all_pieces = np.full(64, None, dtype=object)
        all_pieces[occupancy] = pieces
        return all_pieces
    
    def predict(self, img, turn=chess.WHITE):
        squares = list(chess.SQUARES)
        with torch.no_grad():
            img, img_scale = resize_image(img)
            corners = find_corners(img)
            occupancy = self.classify_occupancy(img, turn, corners)
            pieces = self.classify_pieces(img, turn, corners, occupancy)

            board = chess.Board()
            board.clear_board()
            for square, piece in zip(squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            return board
        
    