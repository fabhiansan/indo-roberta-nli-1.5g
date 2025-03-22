"""
Logging utilities for RoBERTa-based NLI model training.
"""
import os
import logging
import time
from datetime import datetime
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import config

def get_logger(name, log_dir=None, log_filename=None):
    """
    Get a logger with specified name and configuration.
    
    Args:
        name (str): Logger name
        log_dir (str, optional): Directory to save log files
        log_filename (str, optional): Log file name
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.getLevelName(config.LOG_LEVEL))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir and log_filename are provided)
    if log_dir and log_filename:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, log_filename)
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class TrainingLogger:
    """Logger for training progress and metrics."""
    
    def __init__(self, experiment_name=None, log_dir=None):
        """
        Initialize TrainingLogger.
        
        Args:
            experiment_name (str, optional): Name of the experiment
            log_dir (str, optional): Directory to save logs
        """
        self.experiment_name = experiment_name or config.EXPERIMENT_NAME
        self.log_dir = log_dir or config.LOG_DIR
        
        # Create experiment directory
        self.experiment_dir = os.path.join(
            self.log_dir,
            self.experiment_name
        )
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Set up file paths
        self.log_file = os.path.join(self.experiment_dir, "training.log")
        self.metrics_file = os.path.join(self.experiment_dir, "metrics.json")
        
        # Initialize logger
        self.logger = get_logger(
            name="training",
            log_dir=self.experiment_dir,
            log_filename="training.log"
        )
        
        # Initialize TensorBoard writer
        if config.USE_TENSORBOARD:
            self.tensorboard_dir = os.path.join(
                config.TENSORBOARD_DIR,
                self.experiment_name
            )
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        else:
            self.writer = None
        
        # Initialize Weights & Biases
        if config.USE_WANDB:
            try:
                import wandb
                wandb.init(
                    project=config.WANDB_PROJECT,
                    entity=config.WANDB_ENTITY,
                    name=self.experiment_name,
                    config=vars(config)
                )
                self.wandb = wandb
            except ImportError:
                self.logger.warning("Weights & Biases not installed. Skipping W&B initialization.")
                self.wandb = None
        else:
            self.wandb = None
        
        # Initialize metrics storage
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "training_time": [],
            "learning_rate": []
        }
        
        # Log configuration
        self.log_config()
        
    def log_config(self):
        """Log the configuration parameters."""
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Configuration:")
        
        # Get all uppercase attributes from config (which are configuration parameters)
        config_dict = {k: v for k, v in vars(config).items() 
                      if k.isupper() and not k.startswith('_')}
        
        # Log each configuration parameter
        for key, value in config_dict.items():
            self.logger.info(f"  {key}: {value}")
            
        # Save config to JSON
        with open(os.path.join(self.experiment_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
            
    def log_epoch(self, epoch, train_loss, val_loss, metrics, lr, epoch_time):
        """
        Log epoch results.
        
        Args:
            epoch (int): Current epoch
            train_loss (float): Training loss
            val_loss (float): Validation loss
            metrics (dict): Dictionary containing evaluation metrics
            lr (float): Current learning rate
            epoch_time (float): Time taken for the epoch
        """
        message = (f"Epoch {epoch}/{config.NUM_EPOCHS} - "
                  f"Time: {epoch_time:.2f}s - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Accuracy: {metrics['accuracy']:.4f} - "
                  f"F1: {metrics['f1']:.4f} - "
                  f"LR: {lr:.8f}")
        
        self.logger.info(message)
        
        # Record metrics
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["accuracy"].append(metrics["accuracy"])
        self.metrics["precision"].append(metrics["precision"])
        self.metrics["recall"].append(metrics["recall"])
        self.metrics["f1"].append(metrics["f1"])
        self.metrics["training_time"].append(epoch_time)
        self.metrics["learning_rate"].append(lr)
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/validation", val_loss, epoch)
            self.writer.add_scalar("Metrics/accuracy", metrics["accuracy"], epoch)
            self.writer.add_scalar("Metrics/precision", metrics["precision"], epoch)
            self.writer.add_scalar("Metrics/recall", metrics["recall"], epoch)
            self.writer.add_scalar("Metrics/f1", metrics["f1"], epoch)
            self.writer.add_scalar("Training/learning_rate", lr, epoch)
            self.writer.add_scalar("Training/epoch_time", epoch_time, epoch)
        
        # Log to Weights & Biases
        if self.wandb:
            self.wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "learning_rate": lr,
                "epoch_time": epoch_time
            })
            
        # Save metrics to JSON file
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
            
    def log_batch(self, epoch, batch, loss, lr, global_step):
        """
        Log batch results.
        
        Args:
            epoch (int): Current epoch
            batch (int): Current batch
            loss (float): Batch loss
            lr (float): Current learning rate
            global_step (int): Global training step
        """
        if global_step % config.LOGGING_STEPS == 0:
            self.logger.info(
                f"Epoch {epoch}/{config.NUM_EPOCHS} - "
                f"Batch {batch} - "
                f"Loss: {loss:.4f} - "
                f"LR: {lr:.8f}"
            )
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/train_step", loss, global_step)
                self.writer.add_scalar("Training/learning_rate_step", lr, global_step)
                
            # Log to Weights & Biases
            if self.wandb:
                self.wandb.log({
                    "batch": batch,
                    "train_loss_step": loss,
                    "learning_rate_step": lr,
                    "global_step": global_step
                })
                
    def log_model_summary(self, model):
        """
        Log model summary.
        
        Args:
            model: PyTorch model
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model Summary:")
        self.logger.info(f"  Model type: {type(model).__name__}")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        # Log to TensorBoard
        if self.writer and hasattr(model, 'config'):
            try:
                self.writer.add_text("Model/Summary", str(model.config))
            except:
                pass
            
    def log_confusion_matrix(self, cm, epoch, class_names=None):
        """
        Log confusion matrix.
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            epoch (int): Current epoch
            class_names (list, optional): List of class names
        """
        if class_names is None:
            class_names = ["entailment", "contradiction", "neutral"]
            
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        
        # Save figure
        cm_path = os.path.join(self.experiment_dir, f"confusion_matrix_epoch_{epoch}.png")
        plt.savefig(cm_path)
        plt.close()
        
        # Log to TensorBoard
        if self.writer:
            cm_img = plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - Epoch {epoch}')
            self.writer.add_figure("Confusion Matrix", cm_img, epoch)
            plt.close()
            
        # Log to Weights & Biases
        if self.wandb:
            cm_fig = plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - Epoch {epoch}')
            self.wandb.log({"confusion_matrix": wandb.Image(cm_fig)})
            plt.close()
            
    def close(self):
        """Close the logger and write final metrics."""
        # Save final metrics
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
            
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
            
        # Log final message
        self.logger.info(f"Training completed. Metrics saved to {self.metrics_file}")
        
        # Log to Weights & Biases
        if self.wandb:
            self.wandb.finish() 