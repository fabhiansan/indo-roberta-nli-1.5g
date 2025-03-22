"""
Data preprocessing utilities for RoBERTa-based NLI model training.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset
from transformers import RobertaTokenizer
import pandas as pd
import numpy as np

import config
import logging_utils
from utils import set_seed

# Initialize logger
logger = logging_utils.get_logger(name="data_preprocessing")

class NLIDataset(Dataset):
    """Custom dataset for NLI data."""
    
    def __init__(self, encodings, labels=None):
        """
        Initialize NLI dataset.
        
        Args:
            encodings (dict): Encoded inputs from tokenizer
            labels (list, optional): List of labels
        """
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        """
        Get item at index.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Item at index
        """
        # Properly create tensors to avoid warning
        item = {key: val[idx].clone().detach() if isinstance(val[idx], torch.Tensor) else torch.tensor(val[idx]) 
                for key, val in self.encodings.items()}
                
        if self.labels is not None:
            if isinstance(self.labels[idx], torch.Tensor):
                item["labels"] = self.labels[idx].clone().detach()
            else:
                item["labels"] = torch.tensor(self.labels[idx])
                
        return item
        
    def __len__(self):
        """
        Get dataset length.
        
        Returns:
            int: Dataset length
        """
        return len(self.encodings["input_ids"])

def load_nli_dataset(dataset_name=None):
    """
    Load NLI dataset.
    
    Args:
        dataset_name (str, optional): Name of the dataset to load
        
    Returns:
        dict: Dictionary containing the dataset splits
    """
    if dataset_name is None:
        dataset_name = config.DATASET_NAME
        
    logger.info(f"Loading {dataset_name} dataset...")
    
    # Set random seed for reproducibility
    set_seed()
    
    try:
        if dataset_name == "snli":
            # Load SNLI dataset
            dataset = load_dataset("snli")
        elif dataset_name == "mnli":
            # Load MultiNLI dataset
            dataset = load_dataset("multi_nli")
        elif dataset_name == "anli":
            # Load Adversarial NLI dataset
            dataset = load_dataset("anli")
        elif dataset_name == "afaji/indonli":
            # Load IndoNLI dataset
            dataset = load_dataset("afaji/indonli", cache_dir=config.DATA_DIR)
        else:
            # Attempt to load as direct dataset name
            dataset = load_dataset(dataset_name)
            
        logger.info(f"Dataset loaded successfully with splits: {list(dataset.keys())}")
        
        # Log dataset statistics
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise
        
def preprocess_dataset(dataset, tokenizer=None, max_length=None):
    """
    Preprocess dataset.
    
    Args:
        dataset: Dataset to preprocess
        tokenizer: Tokenizer to use for encoding
        max_length (int, optional): Maximum sequence length
        
    Returns:
        tuple: Preprocessed train, validation, and test datasets
    """
    if tokenizer is None:
        # Initialize tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME)
        
    if max_length is None:
        max_length = config.MAX_SEQ_LENGTH
        
    logger.info("Preprocessing dataset...")
    
    # Check if we're using IndoNLI
    using_indonli = config.DATASET_NAME == "afaji/indonli"
    
    # For IndoNLI, the mapping is:
    # entailment (0), neutral (1), contradiction (2)
    # For other datasets:
    # entailment (0), contradiction (1), neutral (2)
    if using_indonli:
        logger.info("Using IndoNLI label mapping: entailment (0), neutral (1), contradiction (2)")
        # We don't need a string-to-int mapping for IndoNLI as it already uses integers
    else:
        logger.info("Using standard NLI label mapping: entailment (0), contradiction (1), neutral (2)")
        # Label mapping for standard NLI datasets (SNLI, MNLI, etc)
        label_map = {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 2
        }
    
    # Function to tokenize and prepare features
    def prepare_features(examples):
        # Get pairs of sentences
        premises = examples["premise"]
        hypotheses = examples["hypothesis"]
        
        # Debug: Check label format
        if "label" in examples:
            logger.info(f"Label type: {type(examples['label'])}")
            logger.info(f"First 5 labels: {examples['label'][:5]}")

        # Tokenize inputs
        tokenized = tokenizer(
            premises,
            hypotheses,
            padding="max_length",
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Map labels to integers
        if "label" in examples:
            if using_indonli:
                # IndoNLI already has integer labels
                if isinstance(examples["label"], list):
                    labels = examples["label"]
                    logger.info(f"Using label list directly: {labels[:5]}")
                else:
                    labels = [examples["label"]]
                    logger.info(f"Created label list from single value: {labels}")
            else:
                # Map string labels to integers for other datasets
                if isinstance(examples["label"], list):
                    labels = [
                        label_map.get(examples["label"][i], 2) if examples["label"][i] != -1 else -1
                        for i in range(len(examples["label"]))
                    ]
                else:
                    labels = label_map.get(examples["label"], 2) if examples["label"] != -1 else -1
        else:
            labels = None
            logger.warning("No labels found in examples")
        
        # Add labels to the tokenized dictionary if they exist
        if labels is not None:
            tokenized["labels"] = torch.tensor(labels)
            
        return tokenized, labels
    
    # Process each split
    train_encodings, train_labels = None, None
    val_encodings, val_labels = None, None
    test_encodings, test_labels = None, None
    
    if "train" in dataset:
        logger.info("Processing training split...")
        train_encodings, train_labels = prepare_features(dataset["train"])
        
    if "validation" in dataset:
        logger.info("Processing validation split...")
        val_encodings, val_labels = prepare_features(dataset["validation"])
    elif "validation_matched" in dataset:
        logger.info("Processing validation_matched split...")
        val_encodings, val_labels = prepare_features(dataset["validation_matched"])
        
    if "test" in dataset:
        logger.info("Processing test split...")
        test_encodings, test_labels = prepare_features(dataset["test"])
        
    # Create dataset objects
    train_dataset = NLIDataset(train_encodings, train_labels) if train_encodings else None
    val_dataset = NLIDataset(val_encodings, val_labels) if val_encodings else None
    test_dataset = NLIDataset(test_encodings, test_labels) if test_encodings else None
    
    # Log dataset sizes
    if train_dataset:
        logger.info(f"Training examples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation examples: {len(val_dataset)}")
    if test_dataset:
        logger.info(f"Test examples: {len(test_dataset)}")
        
    return train_dataset, val_dataset, test_dataset
    
def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=None, eval_batch_size=None):
    """
    Create data loaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size (int, optional): Batch size for training
        eval_batch_size (int, optional): Batch size for evaluation
        
    Returns:
        tuple: Train, validation, and test data loaders
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    if eval_batch_size is None:
        eval_batch_size = config.EVAL_BATCH_SIZE
        
    logger.info("Creating data loaders...")
    
    # Create samplers
    train_sampler = RandomSampler(train_dataset) if train_dataset else None
    val_sampler = SequentialSampler(val_dataset) if val_dataset else None
    test_sampler = SequentialSampler(test_dataset) if test_dataset else None
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS
    ) if train_dataset else None
    
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=eval_batch_size,
        num_workers=config.NUM_WORKERS
    ) if val_dataset else None
    
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=eval_batch_size,
        num_workers=config.NUM_WORKERS
    ) if test_dataset else None
    
    # Log data loader statistics
    if train_dataloader:
        logger.info(f"Number of training batches: {len(train_dataloader)}")
    if val_dataloader:
        logger.info(f"Number of validation batches: {len(val_dataloader)}")
    if test_dataloader:
        logger.info(f"Number of test batches: {len(test_dataloader)}")
        
    return train_dataloader, val_dataloader, test_dataloader

def get_class_distribution(dataset):
    """
    Get class distribution.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        dict: Class distribution
    """
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i]["labels"].item())
        
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    return distribution

def get_tokenizer():
    """
    Get tokenizer.
    
    Returns:
        RobertaTokenizer: RoBERTa tokenizer
    """
    return RobertaTokenizer.from_pretrained(config.MODEL_NAME)

if __name__ == "__main__":
    # Load dataset
    dataset = load_nli_dataset()
    
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    
    # Preprocess dataset
    train_dataset, val_dataset, test_dataset = preprocess_dataset(dataset, tokenizer)
    
    # Get data loaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Analyze class distribution
    if train_dataset:
        train_distribution = get_class_distribution(train_dataset)
        logger.info(f"Training class distribution: {train_distribution}")
        
    if val_dataset:
        val_distribution = get_class_distribution(val_dataset)
        logger.info(f"Validation class distribution: {val_distribution}")
        
    if test_dataset:
        test_distribution = get_class_distribution(test_dataset)
        logger.info(f"Test class distribution: {test_distribution}")
    
    logger.info("Data preprocessing completed successfully.") 