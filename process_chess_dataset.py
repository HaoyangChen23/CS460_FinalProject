import os
import glob
import json
from sklearn.model_selection import train_test_split
from piece_classifier.prepare_dataset import get_all_squares_labels
from piece_classifier.utils import load_dataset
from piece_classifier.model_types import ModelType

def create_directory_structure():
    """创建必要的目录结构"""
    directories = [
        "data/pieces/train",
        "data/pieces/val",
        "data/occupancy/train",
        "data/occupancy/val",
        "data/transfer_learning/occupancy/train",
        "data/test"
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def process_dataset(dataset_dir: str, train_ratio: float = 0.8):
    """处理棋盘数据集
    
    Args:
        dataset_dir: 包含原始图片和JSON文件的目录
        train_ratio: 训练集比例
    """
    # 1. 创建目录结构
    create_directory_structure()
    
    # 2. 获取所有图片文件
    image_files = glob.glob(os.path.join(dataset_dir, "*.png"))
    print(f"Found {len(image_files)} images in {dataset_dir}")
    
    # 3. 分割训练集和验证集
    train_files, val_files = train_test_split(
        image_files, 
        train_size=train_ratio, 
        random_state=42
    )
    print(f"Split into {len(train_files)} training and {len(val_files)} validation images")
    
    # 4. 处理训练集
    print("\nProcessing training set...")
    for img_file in train_files:
        json_file = img_file.replace('.png', '.json')
        if not os.path.exists(json_file):
            print(f"Warning: No JSON file found for {img_file}")
            continue
            
        print(f"Processing {os.path.basename(img_file)}")
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
    
    # 5. 处理验证集
    print("\nProcessing validation set...")
    for img_file in val_files:
        json_file = img_file.replace('.png', '.json')
        if not os.path.exists(json_file):
            print(f"Warning: No JSON file found for {img_file}")
            continue
            
        print(f"Processing {os.path.basename(img_file)}")
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

def verify_dataset():
    """验证处理后的数据集"""
    # 加载并验证训练集
    print("\nVerifying processed dataset...")
    
    # 验证棋子分类数据
    pieces_train = load_dataset(ModelType.CNN, "data/pieces/train", batch_size=1)
    pieces_val = load_dataset(ModelType.CNN, "data/pieces/val", batch_size=1)
    print(f"Pieces classification dataset:")
    print(f"- Training samples: {len(pieces_train.dataset)}")
    print(f"- Validation samples: {len(pieces_val.dataset)}")
    
    # 验证占用检测数据
    occupancy_train = load_dataset(ModelType.CNN, "data/occupancy/train", batch_size=1)
    occupancy_val = load_dataset(ModelType.CNN, "data/occupancy/val", batch_size=1)
    print(f"Occupancy detection dataset:")
    print(f"- Training samples: {len(occupancy_train.dataset)}")
    print(f"- Validation samples: {len(occupancy_val.dataset)}")

def main():
    # 处理数据集
    process_dataset("Dataset")
    
    # 验证处理后的数据集
    verify_dataset()

if __name__ == "__main__":
    main() 