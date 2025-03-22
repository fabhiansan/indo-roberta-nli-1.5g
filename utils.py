"""
Utility functions for RoBERTa-based NLI model training.
"""
import os
import random
import numpy as np
import torch
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import config

def set_seed(seed=None):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int, optional): Random seed to set
    """
    if seed is None:
        seed = config.RANDOM_SEED
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed

def get_device():
    """
    Get the device to use (CPU or GPU).
    
    Returns:
        torch.device: Device to use
    """
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def format_time(elapsed):
    """
    Format elapsed time.
    
    Args:
        elapsed (float): Elapsed time in seconds
        
    Returns:
        str: Formatted time string
    """
    # Round to the nearest second
    elapsed_rounded = int(round(elapsed))
    
    # Format as hh:mm:ss
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_rounded))

def compute_metrics(preds, labels):
    """
    Compute evaluation metrics.
    
    Args:
        preds (numpy.ndarray): Predicted labels
        labels (numpy.ndarray): True labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }

def save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, path):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: PyTorch scheduler
        epoch (int): Current epoch
        loss (float): Current loss
        metrics (dict): Evaluation metrics
        path (str): Path to save the checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save checkpoint
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'metrics': metrics
        },
        path
    )
    
def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint.
    
    Args:
        path (str): Path to the checkpoint
        model: PyTorch model
        optimizer (optional): PyTorch optimizer
        scheduler (optional): PyTorch scheduler
        
    Returns:
        dict: Checkpoint contents
    """
    # Load checkpoint
    checkpoint = torch.load(path, map_location=get_device())
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

class EarlyStopping:
    """Early stopping implementation to prevent overfitting."""
    
    def __init__(self, patience=None, min_delta=0, mode='min'):
        """
        Initialize early stopping.
        
        Args:
            patience (int, optional): Number of epochs to wait for improvement
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'min' or 'max' depending on whether lower or higher values are better
        """
        self.patience = patience or config.EARLY_STOPPING_PATIENCE
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        Check if training should be stopped.
        
        Args:
            score (float): Current validation score
            
        Returns:
            bool: True if should stop, False otherwise
        """
        if self.best_score is None:
            # First epoch
            self.best_score = score
            return False
            
        if self.mode == 'min':
            # For metrics like loss where lower is better
            if score < self.best_score - self.min_delta:
                # Improvement
                self.best_score = score
                self.counter = 0
            else:
                # No improvement
                self.counter += 1
        else:
            # For metrics like accuracy where higher is better
            if score > self.best_score + self.min_delta:
                # Improvement
                self.best_score = score
                self.counter = 0
            else:
                # No improvement
                self.counter += 1
                
        # Check if should stop
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop
        
def count_model_parameters(model):
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def get_lr(optimizer):
    """
    Get current learning rate.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        float: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def print_gpu_memory_usage():
    """Print GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
            cached = torch.cuda.memory_reserved(device) / (1024 * 1024)
            print(f"GPU {i}: {allocated:.2f} MB allocated, {cached:.2f} MB cached")
    else:
        print("No GPU available.")
        
def calculate_warmup_steps(total_steps, warmup_ratio=None):
    """
    Calculate number of warmup steps.
    
    Args:
        total_steps (int): Total number of training steps
        warmup_ratio (float, optional): Ratio of total steps to use for warmup
        
    Returns:
        int: Number of warmup steps
    """
    if warmup_ratio is None:
        warmup_ratio = config.WARMUP_RATIO
        
    return int(total_steps * warmup_ratio) 