"""
Evaluation script for RoBERTa-based NLI model.
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)
from transformers import RobertaForSequenceClassification

import config
import logging_utils
from utils import set_seed, get_device, load_checkpoint
from data_preprocessing import (
    load_nli_dataset, 
    preprocess_dataset, 
    get_dataloaders,
    get_tokenizer
)

# Initialize logger
logger = logging_utils.get_logger(name="evaluation")

def evaluate_model(model, dataloader, device, output_dir=None):
    """
    Evaluate the model with detailed metrics.
    
    Args:
        model: PyTorch model
        dataloader: Data loader to evaluate on
        device: Device to use for evaluation
        output_dir (str, optional): Directory to save evaluation results
        
    Returns:
        dict: Evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize evaluation variables
    all_logits = []
    all_preds = []
    all_labels = []
    eval_loss = 0
    
    # Disable gradient calculation
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            
            # Track loss
            eval_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Get labels
            labels = batch["labels"].cpu().numpy()
            
            # Track predictions, logits, and labels
            all_logits.append(logits.cpu().numpy())
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calculate average loss
    avg_loss = eval_loss / len(dataloader)
    
    # Convert logits to numpy array
    all_logits = np.vstack(all_logits)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Generate classification report
    class_names = ["entailment", "contradiction", "neutral"]
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Log metrics
    logger.info(f"Evaluation Results:")
    logger.info(f"  Loss: {avg_loss:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1: {f1:.4f}")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "report": report
        }
        
        # Save metrics to JSON
        import json
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        # Save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        
        # Save wrongly classified examples
        # This requires access to the original dataset
        if hasattr(dataloader.dataset, 'encodings') and hasattr(dataloader.dataset, 'labels'):
            tokenizer = get_tokenizer()
            wrong_indices = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p != l]
            
            if wrong_indices:
                # Get input IDs and attention masks
                input_ids = [dataloader.dataset.encodings['input_ids'][i] for i in wrong_indices]
                
                # Decode inputs
                decoded_inputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                
                # Get true and predicted labels
                true_labels = [all_labels[i] for i in wrong_indices]
                pred_labels = [all_preds[i] for i in wrong_indices]
                
                # Map labels to class names
                label_to_name = {0: "entailment", 1: "contradiction", 2: "neutral"}
                true_labels_named = [label_to_name[l] for l in true_labels]
                pred_labels_named = [label_to_name[l] for l in pred_labels]
                
                # Create DataFrame
                wrong_df = pd.DataFrame({
                    'index': wrong_indices,
                    'text': decoded_inputs,
                    'true_label': true_labels,
                    'true_label_name': true_labels_named,
                    'pred_label': pred_labels,
                    'pred_label_name': pred_labels_named
                })
                
                # Save to CSV
                wrong_df.to_csv(os.path.join(output_dir, "wrong_predictions.csv"), index=False)
                logger.info(f"Saved {len(wrong_indices)} wrong predictions to {os.path.join(output_dir, 'wrong_predictions.csv')}")
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report
    }

def analyze_predictions(model, dataloader, device, output_dir=None, n_samples=20):
    """
    Analyze model predictions in detail.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to use
        output_dir (str, optional): Directory to save analysis results
        n_samples (int): Number of random samples to analyze
        
    Returns:
        DataFrame: DataFrame with prediction details
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Initialize variables
    samples = []
    
    # Disable gradient calculation
    with torch.no_grad():
        # Get n_samples random indices
        all_indices = list(range(len(dataloader.dataset)))
        np.random.shuffle(all_indices)
        sample_indices = all_indices[:n_samples]
        
        # Get samples
        for idx in sample_indices:
            # Get sample
            sample = dataloader.dataset[idx]
            
            # Move sample to device
            sample = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
            
            # Forward pass
            outputs = model(**sample)
            logits = outputs.logits
            
            # Get prediction
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            
            # Get label
            label = sample["labels"].cpu().numpy()[0]
            
            # Get input text
            input_ids = sample["input_ids"].cpu().numpy()[0]
            attention_mask = sample["attention_mask"].cpu().numpy()[0]
            
            # Decode input
            input_text = tokenizer.decode(input_ids[attention_mask == 1], skip_special_tokens=True)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Map labels to class names
            label_to_name = {0: "entailment", 1: "contradiction", 2: "neutral"}
            true_label_name = label_to_name[label]
            pred_label_name = label_to_name[pred]
            
            # Add sample to list
            samples.append({
                "index": idx,
                "input_text": input_text,
                "true_label": label,
                "true_label_name": true_label_name,
                "pred_label": pred,
                "pred_label_name": pred_label_name,
                "correct": label == pred,
                "prob_entailment": probs[0],
                "prob_contradiction": probs[1],
                "prob_neutral": probs[2]
            })
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Save to CSV
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "prediction_analysis.csv"), index=False)
        
        # Create visualizations
        # 1. Confusion Matrix (small sample)
        cm = confusion_matrix(df["true_label"], df["pred_label"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                  xticklabels=["entailment", "contradiction", "neutral"], 
                  yticklabels=["entailment", "contradiction", "neutral"])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Sample)')
        plt.savefig(os.path.join(output_dir, "sample_confusion_matrix.png"))
        plt.close()
        
        # 2. Probability Distribution
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        sns.histplot(df["prob_entailment"], kde=True)
        plt.title("Entailment Probability")
        plt.subplot(1, 3, 2)
        sns.histplot(df["prob_contradiction"], kde=True)
        plt.title("Contradiction Probability")
        plt.subplot(1, 3, 3)
        sns.histplot(df["prob_neutral"], kde=True)
        plt.title("Neutral Probability")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "probability_distribution.png"))
        plt.close()
        
    return df

def evaluate_from_checkpoint(checkpoint_path, test_dataloader, device, output_dir=None):
    """
    Evaluate model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint
        test_dataloader: Test data loader
        device: Device to use
        output_dir (str, optional): Directory to save evaluation results
        
    Returns:
        dict: Evaluation metrics
    """
    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        output_attentions=config.OUTPUT_ATTENTIONS,
        output_hidden_states=config.OUTPUT_HIDDEN_STATES
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model)
    
    # Move model to device
    model = model.to(device)
    
    # Log checkpoint info
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint['epoch']}")
    logger.info(f"  Loss: {checkpoint['loss']:.4f}")
    
    # Evaluate model
    metrics = evaluate_model(model, test_dataloader, device, output_dir)
    
    return metrics

def main():
    """Main evaluation function."""
    # Set random seed for reproducibility
    set_seed()
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Determine checkpoint path - use best model if available, otherwise final model
    best_model_path = os.path.join(config.MODEL_DIR, f"{config.EXPERIMENT_NAME}_best.pt")
    final_model_path = os.path.join(config.MODEL_DIR, f"{config.EXPERIMENT_NAME}_final.pt")
    
    if os.path.exists(best_model_path):
        checkpoint_path = best_model_path
        logger.info(f"Using best model checkpoint: {checkpoint_path}")
    elif os.path.exists(final_model_path):
        checkpoint_path = final_model_path
        logger.info(f"Using final model checkpoint: {checkpoint_path}")
    else:
        logger.error("No checkpoint found. Please train a model first.")
        return
    
    # Load dataset
    dataset = load_nli_dataset()
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Preprocess dataset
    _, _, test_dataset = preprocess_dataset(dataset, tokenizer)
    
    # Get test data loader
    _, _, test_dataloader = get_dataloaders(None, None, test_dataset)
    
    if test_dataloader is None:
        logger.error("No test data available.")
        return
    
    # Create output directory
    output_dir = os.path.join(config.LOG_DIR, f"{config.EXPERIMENT_NAME}_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate model
    metrics = evaluate_from_checkpoint(checkpoint_path, test_dataloader, device, output_dir)
    
    # Analyze predictions
    df = analyze_predictions(
        model=RobertaForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=config.NUM_LABELS
        ).to(device),
        dataloader=test_dataloader,
        device=device,
        output_dir=output_dir
    )
    
    # Log completion
    logger.info(f"Evaluation completed. Results saved to {output_dir}")
    
    return metrics

if __name__ == "__main__":
    metrics = main() 