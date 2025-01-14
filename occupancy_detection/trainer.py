import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os 
import sys 

# TODO: add this to PYTHONPATH so that this isn't an issue
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # sys path issues, this is a quick fix for now

import logging
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from occupancy_detection.CNN100 import CNN_100
from occupancy_detection.ResNet34 import ResNetClassifier
from typing import List, Mapping, Tuple, Any
from occupancy_detection.model_types import ModelType, load_model, ARGPARSE_TO_TYPE
from occupancy_detection.evaluate import evaluate_model
from occupancy_detection.utils import *

# Create and configure the logger
logger = logging.getLogger('chess_square_classifier')
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

# Create a file handler to log messages to a file
file_handler = logging.FileHandler('chess_square_classifier.log')
file_handler.setLevel(logging.DEBUG)  # Set the logging level for the file handler

# Create a console handler to log messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the logging level for the console handler

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def load_datasets(train_path: str, eval_path: str, batch_size: int = 32, train_size: int =  None, eval_size: int = None,
                  model_type: ModelType = None) -> Tuple[DataLoader, DataLoader]:
    
    """
    Generate DataLoader objects from train path and eval path. 

    Args:
        train_path (str): Path to the directory containing training files
        eval_path (str): Path to the directory containing evaluation files
        batch_size (int, optional): Number of examples per training batch. Defaults to 32.
        train_size (int, optional): Number of examples from training dataset to use as a subset. Defaults to using all examples.
        eval_size (int, optional): Number of examples from evaluation dataset to use as a subset. Defaults to using all examples.
        model_type (ModelType, optional): The ModelType class of the model that will be trained with these datasets**. 

    Returns:
        DataLoader tuple containing the training dataloader and evaluation dataloader, respectively.

    ** the model_type parameter is required because certain ModelTypes such as Inception require unique data pre-processing.
    """

    if model_type is None:
        raise ValueError(f"ModelType {model_type} is invalid.")
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if model_type == ModelType.INCEPTION:
        transform = transforms.Compose([
        transforms.Resize((299, 299)),  
        transforms.ToTensor(),
        ])  # InceptionV3 requires images to be size 299 x 299
    
    # Convert datasets to ImageFolder types
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=eval_path, transform=transform)

    # Sample if required
    if train_size is not None:
        train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
        train_dataset = Subset(train_dataset, train_indices)

    if eval_size is not None:
        test_indices = np.random.choice(len(test_dataset), eval_size, replace=False)
        test_dataset = Subset(test_dataset, test_indices)

    # Convert to dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train(num_epochs: int, model_type: ModelType, save_path: str, train_path: str, eval_path: str, batch_size: int = 32, lr: float = 0.001,
          train_size: int = None, eval_size: int = None) -> None:
    """
    Trains a piece classifier. Saves trained model to `save_path`.

    Args:
        num_epochs (int): The number of training epochs.
        model_type (ModelType): The ModelType class of the model that is being trained
        save_path (str): The outpath path to the save file for the model after finished training.
        train_path (str): The path to the training directory
        eval_path (str): the path to the eval directory
        batch_size (int, optional): Number of examples per batch, defaults to 32.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        train_size (int, optional): Number of examples to use for training as a subset. Defaults to using all training examples.
        eval_size (int, optional): Number of examples to use for evaluating as a subset. Defaults to using all eval examples.
    """
    
    # For choosing models at each epoch
    best_f1 = 0
    model_checkpoint_path = generate_checkpoint_path(save_path)
    # Load datasets
    train_loader, test_loader = load_datasets(train_path, eval_path, batch_size, train_size, eval_size, model_type)
    device = default_device()
    logger.info(f"Using device {device}.")
    
    # Init model
    model = load_model(model_type).to(device)  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = criterion.to(device)
    
    # Train loop
    for epoch in tqdm(range(num_epochs), desc="Beginning new epoch..."):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc="Training batches..."):
            # Zero the parameter gradients
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            if model_type == ModelType.INCEPTION:   # the InceptionV3 model has two loss variants that need to be combined
                primary_outputs, aux_outputs = model(inputs)
                primary_loss = criterion(primary_outputs, labels)
                aux_loss = criterion(aux_outputs, labels)

                loss = primary_loss + 0.4 * aux_loss   # https://arxiv.org/abs/1512.00567 gives reason for choosing 0.4 for the aux weight
            elif model_type == ModelType.OWL:
                labels_list = labels.tolist()
                texts = ["The square is occupied by a chess piece." if label == 1 else "The square is empty." for label in labels_list]
                outputs = model(inputs, texts)
                loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        torch.save(model, model_checkpoint_path)
        acc, macro_f1, weighted_f1 = evaluate_model(model_type, model_checkpoint_path, test_loader)

        if weighted_f1 > best_f1:
            torch.save(model, save_path)
            best_f1 = weighted_f1
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}]: New best weighted F1 {weighted_f1}. Saved model checkpoint to {save_path}.")

    logger.info(f'Finished Training. Best weighted F1 on val set: {best_f1}.')
    
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Attempted to remove {model_checkpoint_path}, but could not find file.")
    os.remove(model_checkpoint_path)  # Delete checkpoint


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str,
                    default="data/occupancy/train",
                    help="Path to training root dir")
    parser.add_argument("--val_path", type=str,
                    default="data/occupancy/val",
                    help="Path to validation root dir")
    parser.add_argument("--save_path", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models", "occupancy", "cnn.pt"),
                        help="Path to model save file")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model_type", type=str, default="cnn", help="Model architecture: ['cnn', 'resnet', ...]")
    parser.add_argument("--train_size", type=int, default=None, help="Size of training subset if needed")
    parser.add_argument("--eval_size", type=int, default=None, help="Size of val subset if needed")
    parser.add_argument("--batch_size", type=int, default=32, help="Sizes of each batch used in training")

    args = parser.parse_args()

    NUM_EPOCHS = args.num_epochs
    MODEL_TYPE = ARGPARSE_TO_TYPE.get(args.model_type)
    SAVE_NAME = args.save_path
    TRAIN_PATH = args.train_path
    EVAL_PATH = args.val_path
    LR = args.lr
    TRAIN_SIZE = args.train_size
    EVAL_SIZE = args.eval_size
    BATCH_SIZE = args.batch_size

    args = vars(args)
    logger.info("Using the following args for training: ")
    for arg, val in args.items():
        logger.info(f"{arg}: {val}")

    train(NUM_EPOCHS,
          MODEL_TYPE,
          SAVE_NAME,
          TRAIN_PATH,
          EVAL_PATH,
          train_size=TRAIN_SIZE,
          eval_size=EVAL_SIZE,
          lr=LR,
          batch_size=BATCH_SIZE)
    
if __name__ == "__main__":
    main()
