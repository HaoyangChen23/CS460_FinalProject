import argparse
import cv2
import chess
from fen_generation.generator import Generator
from board_detection.detect_board import find_corners
import os

def main():
    parser = argparse.ArgumentParser(description="Chess board recognition system")
    parser.add_argument("image_path", type=str, help="Path to the chess board image")
    parser.add_argument("--occupancy_model", type=str, default="resnet",
                        choices=["resnet", "densenet", "cnn"],
                        help="Model type for occupancy detection")
    parser.add_argument("--piece_model", type=str, default="resnet34",
                        choices=["resnet34", "densenet", "cnn"],
                        help="Model type for piece classification")
    parser.add_argument("--white_turn", action="store_true",
                        help="Set if it's white's turn (default: True)")
    args = parser.parse_args()

    # 检查图片文件是否存在
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} does not exist")
        return

    # 读取图片
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Could not read image file {args.image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 初始化生成器
    try:
        generator = Generator(
            occupancy_model_type=args.occupancy_model,
            classifier_model_type=args.piece_model
        )
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        return

    try:
        # 预测棋盘状态
        board = generator.predict(img, chess.WHITE if args.white_turn else chess.BLACK)
        
        # 输出FEN表示法
        print("\nResults:")
        print(f"FEN: {board.fen()}")
        print(f"Board visualization:\n{board}")
        
        # 验证FEN的合法性
        if board.is_valid():
            print("\nThe generated FEN is valid.")
        else:
            print("\nWarning: The generated FEN might not be valid!")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return

if __name__ == "__main__":
    main()
