"""
Inference script for RoBERTa-based NLI model.
"""
import os
import torch
import numpy as np
import pandas as pd
from transformers import RobertaForSequenceClassification, RobertaTokenizer

import config
import logging_utils
from utils import set_seed, get_device, load_checkpoint

# Initialize logger
logger = logging_utils.get_logger(name="inference")

class NLIPredictor:
    """Class for making predictions with a trained NLI model."""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize NLI predictor.
        
        Args:
            model_path (str, optional): Path to the model checkpoint
            device (str or torch.device, optional): Device to use for inference
        """
        # Set device
        if device is None:
            self.device = get_device()
        else:
            self.device = device
            
        # Set model path
        if model_path is None:
            best_model_path = os.path.join(config.MODEL_DIR, f"{config.EXPERIMENT_NAME}_best.pt")
            final_model_path = os.path.join(config.MODEL_DIR, f"{config.EXPERIMENT_NAME}_final.pt")
            
            if os.path.exists(best_model_path):
                self.model_path = best_model_path
                logger.info(f"Using best model checkpoint: {self.model_path}")
            elif os.path.exists(final_model_path):
                self.model_path = final_model_path
                logger.info(f"Using final model checkpoint: {self.model_path}")
            else:
                raise ValueError("No checkpoint found. Please train a model first or provide a valid model path.")
        else:
            self.model_path = model_path
            
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME)
        
        # Initialize model
        self.model = RobertaForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=config.NUM_LABELS,
            output_attentions=config.OUTPUT_ATTENTIONS,
            output_hidden_states=config.OUTPUT_HIDDEN_STATES
        )
        
        # Load checkpoint
        load_checkpoint(self.model_path, self.model)
        
        # Move model to device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define label map
        self.id_to_label = {
            0: "entailment",
            1: "contradiction",
            2: "neutral"
        }
        
        logger.info(f"NLI predictor initialized with model from {self.model_path}")
        
    def predict(self, premise, hypothesis, return_probabilities=False):
        """
        Make a prediction for a single premise-hypothesis pair.
        
        Args:
            premise (str): Premise text
            hypothesis (str): Hypothesis text
            return_probabilities (bool): Whether to return probabilities for each class
            
        Returns:
            str or tuple: Predicted label or (predicted label, probabilities)
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            premise,
            hypothesis,
            padding="max_length",
            truncation="longest_first",
            max_length=config.MAX_SEQ_LENGTH,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Get predicted class
        pred_class = torch.argmax(logits, dim=1).cpu().numpy()[0]
        pred_label = self.id_to_label[pred_class]
        
        if return_probabilities:
            # Calculate probabilities
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Create dictionary mapping labels to probabilities
            probabilities = {
                "entailment": probs[0],
                "contradiction": probs[1],
                "neutral": probs[2]
            }
            
            return pred_label, probabilities
        else:
            return pred_label
            
    def predict_batch(self, premises, hypotheses, batch_size=None, return_probabilities=False):
        """
        Make predictions for a batch of premise-hypothesis pairs.
        
        Args:
            premises (list): List of premise texts
            hypotheses (list): List of hypothesis texts
            batch_size (int, optional): Batch size for inference
            return_probabilities (bool): Whether to return probabilities for each class
            
        Returns:
            list or tuple: List of predicted labels or (list of predicted labels, list of probabilities)
        """
        if batch_size is None:
            batch_size = config.EVAL_BATCH_SIZE
            
        # Check input lengths
        assert len(premises) == len(hypotheses), "Premises and hypotheses must have the same length"
        
        # Initialize results
        all_pred_labels = []
        all_probabilities = [] if return_probabilities else None
        
        # Process in batches
        for i in range(0, len(premises), batch_size):
            # Get batch
            batch_premises = premises[i:i+batch_size]
            batch_hypotheses = hypotheses[i:i+batch_size]
            
            # Tokenize inputs
            inputs = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                padding="max_length",
                truncation="longest_first",
                max_length=config.MAX_SEQ_LENGTH,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Get predicted classes
            pred_classes = torch.argmax(logits, dim=1).cpu().numpy()
            pred_labels = [self.id_to_label[pred_class] for pred_class in pred_classes]
            
            # Add to results
            all_pred_labels.extend(pred_labels)
            
            if return_probabilities:
                # Calculate probabilities
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                
                # Create list of dictionaries mapping labels to probabilities
                batch_probabilities = [
                    {
                        "entailment": probs[j][0],
                        "contradiction": probs[j][1],
                        "neutral": probs[j][2]
                    }
                    for j in range(len(probs))
                ]
                
                all_probabilities.extend(batch_probabilities)
        
        if return_probabilities:
            return all_pred_labels, all_probabilities
        else:
            return all_pred_labels
            
    def predict_from_csv(self, input_file, output_file=None, premise_col="premise", hypothesis_col="hypothesis", label_col=None, batch_size=None):
        """
        Make predictions for premise-hypothesis pairs from a CSV file.
        
        Args:
            input_file (str): Path to input CSV file
            output_file (str, optional): Path to output CSV file
            premise_col (str): Name of column containing premises
            hypothesis_col (str): Name of column containing hypotheses
            label_col (str, optional): Name of column containing true labels (for evaluation)
            batch_size (int, optional): Batch size for inference
            
        Returns:
            pandas.DataFrame: DataFrame with predictions
        """
        # Read input file
        df = pd.read_csv(input_file)
        
        # Check columns
        assert premise_col in df.columns, f"Column '{premise_col}' not found in {input_file}"
        assert hypothesis_col in df.columns, f"Column '{hypothesis_col}' not found in {input_file}"
        
        # Get premises and hypotheses
        premises = df[premise_col].tolist()
        hypotheses = df[hypothesis_col].tolist()
        
        # Make predictions
        pred_labels, probabilities = self.predict_batch(
            premises, hypotheses, batch_size=batch_size, return_probabilities=True
        )
        
        # Add predictions to DataFrame
        df["predicted_label"] = pred_labels
        
        # Ensure probabilities is not None and is iterable
        if probabilities and isinstance(probabilities, list):
            df["entailment_prob"] = [p["entailment"] for p in probabilities]
            df["contradiction_prob"] = [p["contradiction"] for p in probabilities]
            df["neutral_prob"] = [p["neutral"] for p in probabilities]
        
        # Calculate accuracy if true labels are available
        if label_col is not None and label_col in df.columns:
            accuracy = (df[label_col] == df["predicted_label"]).mean()
            logger.info(f"Accuracy on {input_file}: {accuracy:.4f}")
        
        # Save output file
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file}")
        
        return df
        
def demo():
    """Demo function with example predictions."""
    # Set random seed for reproducibility
    set_seed()
    
    # Initialize predictor
    predictor = NLIPredictor()
    
    # Example premise-hypothesis pairs in Indonesian
    examples = [
        {
            "premise": "Seorang pria sedang memeriksa seragam di negara Asia Timur.",
            "hypothesis": "Pria itu sedang tidur.",
            "label": "contradiction"
        },
        {
            "premise": "Seorang ibu dan anak perempuan tersenyum.",
            "hypothesis": "Dua wanita sedang tersenyum.",
            "label": "entailment"
        },
        {
            "premise": "Pertandingan sepak bola dengan beberapa pemain laki-laki.",
            "hypothesis": "Beberapa pria sedang berolahraga.",
            "label": "entailment"
        },
        {
            "premise": "Mobil balap hitam mulai melaju di depan kerumunan orang.",
            "hypothesis": "Seorang pria sedang mengendarai mobil mahal.",
            "label": "neutral"
        }
    ]
    
    # Make predictions
    print("\nExample predictions:")
    print("-" * 80)
    
    for i, example in enumerate(examples):
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        true_label = example["label"]
        
        # Predict
        pred_label, probs = predictor.predict(premise, hypothesis, return_probabilities=True)
        
        # Print results
        print(f"\nExample {i+1}:")
        print(f"Premise: {premise}")
        print(f"Hypothesis: {hypothesis}")
        print(f"True label: {true_label}")
        print(f"Predicted label: {pred_label}")
        print(f"Probabilities: {probs}")
        
        # Check correctness
        if pred_label == true_label:
            print("✓ Correct prediction")
        else:
            print("✗ Incorrect prediction")
    
    print("-" * 80)
        
def predict_from_file():
    """Function to make predictions from a file."""
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Make predictions with a trained NLI model")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, help="Path to output CSV file")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--premise-col", type=str, default="premise", help="Name of column containing premises")
    parser.add_argument("--hypothesis-col", type=str, default="hypothesis", help="Name of column containing hypotheses")
    parser.add_argument("--label-col", type=str, help="Name of column containing true labels (for evaluation)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed()
    
    # Initialize predictor
    predictor = NLIPredictor(model_path=args.model)
    
    # Make predictions
    df = predictor.predict_from_csv(
        input_file=args.input,
        output_file=args.output,
        premise_col=args.premise_col,
        hypothesis_col=args.hypothesis_col,
        label_col=args.label_col,
        batch_size=args.batch_size
    )
    
    return df

def push_to_huggingface():
    """
    Push an existing model to HuggingFace Hub.
    
    This function can be used to push a pre-trained model to HuggingFace Hub
    without having to retrain it.
    """
    import argparse
    from model_training import save_model_to_hub
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Push a trained model to HuggingFace Hub")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--username", type=str, help="HuggingFace username")
    parser.add_argument("--model-name", type=str, help="Model name on HuggingFace Hub")
    parser.add_argument("--private", action="store_true", help="Whether to create a private repository")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set model path
    if args.model:
        model_path = args.model
    else:
        best_model_path = os.path.join(config.MODEL_DIR, f"{config.EXPERIMENT_NAME}_best.pt")
        final_model_path = os.path.join(config.MODEL_DIR, f"{config.EXPERIMENT_NAME}_final.pt")
        
        if os.path.exists(best_model_path):
            model_path = best_model_path
            logger.info(f"Using best model checkpoint: {model_path}")
        elif os.path.exists(final_model_path):
            model_path = final_model_path
            logger.info(f"Using final model checkpoint: {model_path}")
        else:
            logger.error("No checkpoint found. Please train a model first or provide a valid model path.")
            return
    
    # Update config with command-line arguments
    if args.username:
        config.HF_USERNAME = args.username
    if args.model_name:
        config.HF_MODEL_NAME = args.model_name
    if args.private:
        config.HF_PRIVATE_REPO = True
        
    # Check if HF_USERNAME is set
    if config.HF_USERNAME is None:
        logger.error("HF_USERNAME not set. Please provide a username with --username or set it in config.py.")
        return
        
    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        output_attentions=config.OUTPUT_ATTENTIONS,
        output_hidden_states=config.OUTPUT_HIDDEN_STATES
    )
    
    # Load checkpoint
    load_checkpoint(model_path, model)
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Push to HuggingFace Hub
    success = save_model_to_hub(model, tokenizer)
    
    if success:
        logger.info(f"Model successfully pushed to HuggingFace Hub: https://huggingface.co/{config.HF_USERNAME}/{config.HF_MODEL_NAME}")
    else:
        logger.error("Failed to push model to HuggingFace Hub.")

if __name__ == "__main__":
    # Check if input file is provided
    import sys
    
    if len(sys.argv) > 1:
        # Check if it's a push to HuggingFace command
        if sys.argv[1] == "push_to_hub":
            # Remove the first argument ("push_to_hub")
            sys.argv.pop(1)
            # Run push to HuggingFace function
            push_to_huggingface()
        else:
            # Make predictions from file
            predict_from_file()
    else:
        # Run demo
        demo() 