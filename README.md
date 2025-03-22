# Indonesian RoBERTa-based NLI Model

This project implements a RoBERTa-based model for Natural Language Inference (NLI) tasks in Indonesian. The model is trained on the IndoNLI dataset to classify the relationship between a premise and hypothesis as either entailment, contradiction, or neutral.

## Features

- Training a RoBERTa model on the IndoNLI dataset
- Uses the Indonesian RoBERTa model (cahya/roberta-base-indonesian-1.5G)
- Comprehensive logging of training metrics
- TensorBoard integration for visualization
- Detailed evaluation with confusion matrices and error analysis
- Inference pipeline for making predictions
- Support for batch inference from CSV files
- Integration with HuggingFace Hub for model sharing

## Project Structure

```
├── config.py              # Configuration parameters
├── data_preprocessing.py  # Data loading and preprocessing
├── model_training.py      # Model definition and training
├── evaluation.py          # Model evaluation
├── inference.py           # Making predictions with trained model
├── logging_utils.py       # Logging functionality
├── utils.py               # General helper functions
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
# OR
venv\Scripts\activate  # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Login to HuggingFace Hub if you plan to push your model:
```bash
huggingface-cli login
```

## Usage

### Training

To train the model with default settings:

```bash
python model_training.py
```

You can modify the configuration parameters in `config.py` to customize the training process. Especially, make sure to set these parameters for HuggingFace Hub integration:

```python
# In config.py
HF_MODEL_NAME = "roberta-indonesian-nli"  # Name for your model on HuggingFace Hub
HF_USERNAME = "your-username"  # Your HuggingFace username
PUSH_TO_HUB = True  # Set to True to push the model to HuggingFace Hub
```

### Evaluation

To evaluate a trained model:

```bash
python evaluation.py
```

This will evaluate the best available model checkpoint on the test set and generate detailed metrics and visualizations.

### Inference

#### Using the Demo

To run a demo with example inputs:

```bash
python inference.py
```

#### Batch Inference from CSV

To make predictions on a CSV file containing premise-hypothesis pairs:

```bash
python inference.py --input data/test.csv --output predictions.csv
```

Additional options:
```
--model MODEL_PATH          # Path to specific model checkpoint
--premise-col PREMISE_COL   # Name of column containing premises (default: "premise")
--hypothesis-col HYPOTHESIS_COL  # Name of column containing hypotheses (default: "hypothesis")
--label-col LABEL_COL       # Name of column containing true labels for evaluation
--batch-size BATCH_SIZE     # Batch size for inference (default: 32)
```

## Logging and Visualization

Training logs and metrics are saved to the `logs` directory. You can visualize the training progress using TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

## Model Checkpoints

The following model checkpoints are saved during training:

- `{EXPERIMENT_NAME}_best.pt`: Best model based on validation loss
- `{EXPERIMENT_NAME}_final.pt`: Final model after training
- `{EXPERIMENT_NAME}_step_{STEP}.pt`: Intermediate checkpoints saved during training

## Configuration

Key configuration parameters in `config.py`:

- **Dataset**: IndoNLI dataset (afaji/indonli)
- **Model**: Indonesian RoBERTa model (cahya/roberta-base-indonesian-1.5G)
- **Training**: Batch size, learning rate, epochs, etc.
- **Logging**: TensorBoard, Weights & Biases integration
- **HuggingFace Hub**: Settings for pushing the model to HuggingFace Hub

## Pushing to HuggingFace Hub

The model is automatically pushed to HuggingFace Hub after training if `PUSH_TO_HUB` is set to `True` in the configuration. You need to:

1. Set `HF_USERNAME` to your HuggingFace username in `config.py`
2. Set `HF_MODEL_NAME` to the desired model name in `config.py`
3. Login to HuggingFace Hub using `huggingface-cli login` before training
4. The model, tokenizer, and model card will be pushed to `{HF_USERNAME}/{HF_MODEL_NAME}`

You can also push an existing trained model to HuggingFace Hub without retraining:

```bash
python inference.py push_to_hub --username your-username --model-name your-model-name
```

Additional options:
```
--model MODEL_PATH    # Path to specific model checkpoint (default: best or final model)
--private             # Whether to create a private repository
```

## License

[Specify license here]

## Acknowledgements

This project uses the following resources:
- [IndoNLI dataset](https://huggingface.co/datasets/afaji/indonli)
- [Indonesian RoBERTa model](https://huggingface.co/cahya/roberta-base-indonesian-1.5G)
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- TensorBoard
- scikit-learn 