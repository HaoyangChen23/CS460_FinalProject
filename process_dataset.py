import os
import glob
import json
import shutil
import numpy as np
from piece_classifier.prepare_dataset import get_all_squares_labels
from sklearn.model_selection import train_test_split

def process_dataset(dataset_dir, train_ratio=0.8):
    """处理数据集，将其分割为训练集和验证集
    
    Args:
        dataset_dir: Dataset目录的路径
        train_ratio: 训练集比例
    """
    # 获取所有图片文件
    image_files = glob.glob(os.path.join(dataset_dir, "*.png"))
    
    # 分割训练集和验证集
    train_files, val_files = train_test_split(image_files, train_size=train_ratio, random_state=42)
    
    # 处理训练集
    print("Processing training set...")
    for img_file in train_files:
        json_file = img_file.replace('.png', '.json')
        if not os.path.exists(json_file):
            print(f"Warning: No JSON file found for {img_file}")
            continue
            
        # 处理棋子分类数据
        get_all_squares_labels(
            output_dir="data/pieces/train",
            img_path=img_file,
            json_path=json_file
        )
        
        # 处理占用检测数据
        get_all_squares_labels(
            output_dir="data/occupancy/train",
            img_path=img_file,
            json_path=json_file
        )
    
    # 处理验证集
    print("Processing validation set...")
    for img_file in val_files:
        json_file = img_file.replace('.png', '.json')
        if not os.path.exists(json_file):
            print(f"Warning: No JSON file found for {img_file}")
            continue
            
        # 处理棋子分类数据
        get_all_squares_labels(
            output_dir="data/pieces/val",
            img_path=img_file,
            json_path=json_file
        )
        
        # 处理占用检测数据
        get_all_squares_labels(
            output_dir="data/occupancy/val",
            img_path=img_file,
            json_path=json_file
        )

if __name__ == "__main__":
    process_dataset("Dataset") 