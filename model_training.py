"""
Model training script for RoBERTa-based NLI model.
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
import numpy as np
from tqdm import tqdm

import config
import logging_utils
from utils import (
    set_seed, 
    get_device, 
    compute_metrics, 
    save_checkpoint, 
    format_time, 
    EarlyStopping, 
    get_lr,
    calculate_warmup_steps,
    print_gpu_memory_usage
)
from data_preprocessing import (
    load_nli_dataset, 
    preprocess_dataset, 
    get_dataloaders,
    get_tokenizer
)

# Initialize logger
logger = logging_utils.get_logger(name="model_training")

def initialize_model():
    """
    Initialize RoBERTa model for sequence classification.
    
    Returns:
        RobertaForSequenceClassification: Initialized model
    """
    logger.info(f"Initializing RoBERTa model with {config.NUM_LABELS} labels...")
    
    # Set random seed for reproducibility
    set_seed()
    
    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        output_attentions=config.OUTPUT_ATTENTIONS,
        output_hidden_states=config.OUTPUT_HIDDEN_STATES
    )
    
    return model

def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    device,
    training_logger,
    num_epochs=None
):
    """
    Train the model.
    
    Args:
        model: PyTorch model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        device: Device to use for training
        training_logger: Training logger
        num_epochs (int, optional): Number of epochs to train for
        
    Returns:
        model: Trained model
    """
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
        
    # Move model to device
    model = model.to(device)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode='min')
    
    # Log model summary
    training_logger.log_model_summary(model)
    
    # Calculate total number of training steps
    total_steps = len(train_dataloader) * num_epochs
    
    logger.info(f"Starting training on {device}...")
    logger.info(f"  Num examples: {len(train_dataloader.dataset)}")
    logger.info(f"  Num batches: {len(train_dataloader)}")
    logger.info(f"  Num epochs: {num_epochs}")
    logger.info(f"  Batch size: {config.BATCH_SIZE}")
    logger.info(f"  Total optimization steps: {total_steps}")
    
    # Initialize training variables
    global_step = 0
    best_val_loss = float('inf')
    best_model_path = os.path.join(config.MODEL_DIR, f"{config.EXPERIMENT_NAME}_best.pt")
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        
        # Measure epoch time
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        
        # Initialize batch loss
        train_loss = 0
        
        # Progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs} [Training]")
        
        # Batch training loop
        for batch_idx, batch in enumerate(progress_bar):
            # Debug: Print batch contents for the first few batches
            if batch_idx == 0:
                logger.info(f"First batch contains keys: {batch.keys()}")
                if "labels" in batch:
                    logger.info(f"Labels shape: {batch['labels'].shape}")
                    logger.info(f"Labels sample: {batch['labels'][:5]}")
                else:
                    logger.error("ERROR: 'labels' not found in batch!")
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Clear accumulated gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Handle case where loss is None
            if loss is None:
                logger.error(f"Loss is None at batch {batch_idx}. Skipping this batch.")
                continue
            
            # Apply gradient accumulation
            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
                
            # Backward pass
            loss.backward()
            
            # Track loss
            train_loss += loss.item()
            
            # Log batch progress
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{get_lr(optimizer):.8f}"
            })
            
            # Apply gradient accumulation
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                # Clip gradients
                nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                
                # Update weights
                optimizer.step()
                
                # Update learning rate
                if scheduler is not None:
                    scheduler.step()
                    
                # Log batch
                training_logger.log_batch(
                    epoch=epoch,
                    batch=batch_idx + 1,
                    loss=loss.item() * config.GRADIENT_ACCUMULATION_STEPS,
                    lr=get_lr(optimizer),
                    global_step=global_step
                )
                
                # Increment global step
                global_step += 1
                
                # Save checkpoint at save steps
                if global_step % config.SAVE_STEPS == 0:
                    logger.info(f"Saving checkpoint at step {global_step}...")
                    checkpoint_path = os.path.join(
                        config.MODEL_DIR,
                        f"{config.EXPERIMENT_NAME}_step_{global_step}.pt"
                    )
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        loss=train_loss / (batch_idx + 1),
                        metrics={"step": global_step},
                        path=checkpoint_path
                    )
                    
                # Evaluate at eval steps
                if val_dataloader is not None and global_step % config.EVAL_STEPS == 0:
                    logger.info(f"Evaluating at step {global_step}...")
                    val_loss, val_metrics = evaluate(model, val_dataloader, device)
                    
                    # Log validation results
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
                    logger.info(f"Validation F1: {val_metrics['f1']:.4f}")
                    
                    # Set model back to training mode
                    model.train()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Evaluation phase
        if val_dataloader is not None:
            logger.info("Running validation...")
            val_loss, val_metrics = evaluate(model, val_dataloader, device)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            training_logger.log_epoch(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                metrics=val_metrics,
                lr=get_lr(optimizer),
                epoch_time=epoch_time
            )
            
            # Save best model
            if val_loss < best_val_loss:
                logger.info(f"New best validation loss: {val_loss:.4f} (was {best_val_loss:.4f})")
                best_val_loss = val_loss
                
                # Save best model
                logger.info(f"Saving best model to {best_model_path}...")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    loss=val_loss,
                    metrics=val_metrics,
                    path=best_model_path
                )
                
            # Check for early stopping
            if early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        else:
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results without validation
            training_logger.log_epoch(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=0.0,
                metrics={"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                lr=get_lr(optimizer),
                epoch_time=epoch_time
            )
            
    logger.info("Training complete!")
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_DIR, f"{config.EXPERIMENT_NAME}_final.pt")
    logger.info(f"Saving final model to {final_model_path}...")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=num_epochs,
        loss=train_loss / len(train_dataloader),
        metrics={"global_step": global_step},
        path=final_model_path
    )
    
    return model

def evaluate(model, dataloader, device):
    """
    Evaluate the model.
    
    Args:
        model: PyTorch model
        dataloader: Data loader to evaluate on
        device: Device to use for evaluation
        
    Returns:
        tuple: (avg_loss, metrics)
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize evaluation variables
    eval_loss = 0
    all_preds = []
    all_labels = []
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    # Disable gradient calculation
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Track loss
            eval_loss += loss.item()
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Get labels
            labels = batch["labels"].cpu().numpy()
            
            # Track predictions and labels
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    # Calculate average loss
    avg_loss = eval_loss / len(dataloader)
    
    # Calculate metrics
    metrics = compute_metrics(all_preds, all_labels)
    
    return avg_loss, metrics

def get_optimizer_and_scheduler(model, num_training_steps):
    """
    Get optimizer and learning rate scheduler.
    
    Args:
        model: PyTorch model
        num_training_steps (int): Number of training steps
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.LEARNING_RATE,
        eps=config.ADAM_EPSILON
    )
    
    # Calculate number of warmup steps
    warmup_steps = calculate_warmup_steps(num_training_steps)
    
    # Create scheduler
    if config.LR_SCHEDULER_TYPE == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    elif config.LR_SCHEDULER_TYPE == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    elif config.LR_SCHEDULER_TYPE == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        scheduler = None
        
    return optimizer, scheduler

def save_model_to_hub(model, tokenizer):
    """
    Save the model to HuggingFace Hub.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        
    Returns:
        bool: Whether the model was successfully saved to the Hub
    """
    if not config.PUSH_TO_HUB:
        logger.info("Skipping push to HuggingFace Hub (disabled in config)")
        return False
        
    if config.HF_USERNAME is None:
        logger.warning("HF_USERNAME not set in config. Unable to push to HuggingFace Hub.")
        return False
        
    logger.info(f"Pushing model to HuggingFace Hub as {config.HF_USERNAME}/{config.HF_MODEL_NAME}...")
    
    try:
        # Set repo name
        repo_name = f"{config.HF_USERNAME}/{config.HF_MODEL_NAME}"
        
        # Create model card content
        model_card = f"""
# {config.HF_MODEL_NAME}

This model is a fine-tuned version of [{config.MODEL_NAME}](https://huggingface.co/{config.MODEL_NAME}) on the [{config.DATASET_NAME}](https://huggingface.co/{config.DATASET_NAME}) dataset for Natural Language Inference (NLI) tasks in Indonesian.

## Model Details

- **Model Type:** RoBERTa
- **Language:** Indonesian
- **Base Model:** {config.MODEL_NAME}
- **Training Dataset:** {config.DATASET_NAME}
- **Task:** Natural Language Inference (NLI)
- **Labels:** entailment, contradiction, neutral

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{config.HF_USERNAME}/{config.HF_MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained("{config.HF_USERNAME}/{config.HF_MODEL_NAME}")

# Example usage
premise = "Ibu sedang memasak di dapur."
hypothesis = "Ibu sedang berada di dapur."

# Tokenize inputs
inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)

# Make prediction
import torch
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
# Map predictions to labels
label_map = {{0: "entailment", 1: "contradiction", 2: "neutral"}}
predicted_class = torch.argmax(predictions, dim=1).item()
predicted_label = label_map[predicted_class]
print(f"Predicted label: {{predicted_label}}")
print(f"Probabilities: entailment={{predictions[0][0].item():.4f}}, contradiction={{predictions[0][1].item():.4f}}, neutral={{predictions[0][2].item():.4f}}")
```

## Training Details

- **Optimizer:** AdamW
- **Learning Rate:** {config.LEARNING_RATE}
- **Batch Size:** {config.BATCH_SIZE}
- **Epochs:** {config.NUM_EPOCHS}
- **Max Sequence Length:** {config.MAX_SEQ_LENGTH}
- **Training Framework:** PyTorch with 🤗 Transformers
        """
        
        # Push to Hub
        model.push_to_hub(
            repo_id=repo_name,
            commit_message="Add final trained model",
            private=config.HF_PRIVATE_REPO,
            use_auth_token=config.USE_AUTH_TOKEN
        )
        
        # Push tokenizer to Hub
        tokenizer.push_to_hub(
            repo_id=repo_name,
            commit_message="Add tokenizer",
            private=config.HF_PRIVATE_REPO,
            use_auth_token=config.USE_AUTH_TOKEN
        )
        
        # Create README.md for the model card
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            token=os.environ.get("HF_TOKEN"),
            repo_id=repo_name,
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            commit_message="Add model card"
        )
        
        logger.info(f"Model successfully pushed to HuggingFace Hub: https://huggingface.co/{repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error pushing model to HuggingFace Hub: {str(e)}")
        return False

def main():
    """Main training function."""
    # Set random seed for reproducibility
    set_seed()
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Print GPU memory usage
    print_gpu_memory_usage()
    
    # Initialize training logger
    training_logger = logging_utils.TrainingLogger()
    
    # Load dataset
    dataset = load_nli_dataset()
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Preprocess dataset
    train_dataset, val_dataset, test_dataset = preprocess_dataset(dataset, tokenizer)
    
    # Get data loaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Initialize model
    model = initialize_model()
    
    # Calculate total number of training steps
    total_steps = len(train_dataloader) * config.NUM_EPOCHS
    
    # Get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, total_steps)
    
    # Train model
    trained_model = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        training_logger=training_logger,
        num_epochs=config.NUM_EPOCHS
    )
    
    # Push model to HuggingFace Hub if configured
    if config.PUSH_TO_HUB:
        save_model_to_hub(trained_model, tokenizer)
    
    # Close logger
    training_logger.close()
    
    return trained_model

if __name__ == "__main__":
    trained_model = main() 