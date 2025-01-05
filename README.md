# FEN (Forsyth–Edwards Notation) Generation System

A computer vision-based chess board recognition system that can detect piece positions and types from chess board images and generate corresponding FEN (Forsyth–Edwards Notation) notation.

## Project Structure

```
.
├── board_detection/      # Board detection module
├── occupancy_detection/  # Square occupancy detection module
├── piece_classifier/     # Chess piece classification module
├── fen_generation/      # FEN notation generation module
├── data_visualization/  # Visualization tools
├── transfer_learning/   # Transfer learning experiments
├── Dataset/            # Original dataset with images and annotations
│   ├── *.png          # Chess board images
│   └── *.json         # Corresponding annotations
├── data/              # Processed data directory
│   ├── occupancy/     # Occupancy detection data
│   │   ├── train/     # Training data
│   │   └── val/       # Validation data
│   └── pieces/        # Piece classification data
│       ├── train/     # Training data
│       └── val/       # Validation data
├── saved_models/      # Saved model weights
│   ├── occupancy/     # Occupancy detection models
│   ├── pieces/        # Piece classification models
│   ├── board/         # Board detection models
│   └── transfer_learning/  # Transfer learning models
├── process_chess_dataset.py  # Main dataset processing script
├── process_dataset.py        # Dataset processing utilities
├── run_chess_recognition.py  # Main execution script
├── environment.yml           # Conda environment configuration
└── requirements.txt          # Pip requirements file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- OpenCV
- Pillow
- numpy
- python-chess

## Environment Setup

First, clone the project:
```bash
git clone https://github.com/HaoyangChen23/CS460_FinalProject.git
cd CS460_FinalProject
```

Then choose one of the following methods to configure the environment:

### Method 1: Using Conda (Recommended)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate chess

# If GPU support is needed, install CUDA toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Method 2: Using pip

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Directory Preparation

Create necessary directory structure:
```bash
# Create data directories
mkdir -p data/occupancy/train
mkdir -p data/occupancy/val
mkdir -p data/pieces/train
mkdir -p data/pieces/val

# Create model directories
mkdir -p saved_models/occupancy/stratified_data
mkdir -p saved_models/occupancy/{cnn,densenet,resnet}
mkdir -p saved_models/pieces/stratified_data
mkdir -p saved_models/pieces/{cnn,densenet,resnet34}
mkdir -p saved_models/board
mkdir -p saved_models/transfer_learning
```

Note: You'll need the following model files in place before running evaluations:
- `saved_models/occupancy/stratified_data/resnet/model.pt`: Occupancy detection ResNet model
- `saved_models/occupancy/stratified_data/densenet/model.pt`: Occupancy detection DenseNet model
- `saved_models/occupancy/stratified_data/cnn/model.pt`: Occupancy detection CNN model
- `saved_models/pieces/stratified_data/resnet34/model.pt`: Piece classification ResNet34 model
- `saved_models/pieces/stratified_data/densenet/model.pt`: Piece classification DenseNet model
- `saved_models/pieces/stratified_data/cnn/model.pt`: Piece classification CNN model

These model files can be obtained by either:
1. Training the models yourself using the training commands below
2. Downloading pre-trained models (if available)

## Training Process

### 1. Data Preparation

Process the dataset:
```bash
# Process the chess dataset
python process_chess_dataset.py \
    --input_dir Dataset \
    --output_dir data

# Prepare piece classification dataset
python piece_classifier/prepare_dataset.py \
    --input_dir Dataset \
    --output_dir data/pieces
```

### 2. Training Occupancy Detection Model

```bash
python occupancy_detection/trainer.py \
    --model_type resnet \
    --train_path data/occupancy/train \
    --val_path data/occupancy/val \
    --save_path saved_models/occupancy/stratified_data/resnet/model.pt \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 0.001
```

### 3. Training Piece Classification Model

```bash
python piece_classifier/trainer.py \
    --model_type resnet \
    --train_path data/pieces/train \
    --val_path data/pieces/val \
    --save_path saved_models/pieces/stratified_data/resnet34/model.pt \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 0.001
```

### Individual Module Execution

1. Board Detection:
```bash
python board_detection/detect_board.py Dataset/0094.png
```

2. Occupancy Detection:
```bash
python occupancy_detection/evaluate.py \
    --model_type resnet \
    --save_name saved_models/occupancy/stratified_data/resnet/model.pt \
    --eval_path data/occupancy/val \
    --batch_size 32
```

3. Piece Classification:
```bash
python piece_classifier/evaluate.py \
    --model_type resnet \
    --save_name saved_models/pieces/stratified_data/resnet34/model.pt \
    --eval_path data/pieces/val \
    --batch_size 32
```

### Complete System Execution

The `run_chess_recognition.py` script provides a complete chess board recognition pipeline that sequentially performs board detection, square occupancy detection, and piece classification to generate the FEN notation.

Basic Usage:
```bash
# Using default settings (ResNet for occupancy detection, ResNet34 for piece classification)
python run_chess_recognition.py path/to/image.png

# Specifying different model types
python run_chess_recognition.py path/to/image.png \
    --occupancy_model densenet \
    --piece_model densenet

# Specifying black's turn (default is white's turn)
python run_chess_recognition.py path/to/image.png --white_turn
```

Parameters:
- `image_path`: Path to the input chess board image (required)
- `--occupancy_model`: Model type for occupancy detection (optional, default: resnet)
  - Available options: resnet, densenet, cnn
- `--piece_model`: Model type for piece classification (optional, default: resnet34)
  - Available options: resnet34, densenet, cnn
- `--white_turn`: Specify if it's white's turn (optional, default: True)

Output Description:
- FEN Notation: Standard FEN string representation
- Board Visualization: ASCII-formatted chess board state
- Validity Check: Verification of the generated FEN's legality 