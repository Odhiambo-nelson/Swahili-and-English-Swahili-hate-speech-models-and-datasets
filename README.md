# Swahili-and-English-Swahili-hate-speech-models-and-datasets
This repository provides scripts and configurations for fine-tuning SwahBERT, a BERT-based language model optimized for Swahili and code-switched English-Swahili textual data. The primary task demonstrated here is hate speech detection, although the model can be adapted for other downstream NLP tasks such as text classification, named entity recognition (NER), and sentiment analysis, provided appropriate labeled datasets are available. The repository also includes baseline implementations of a Bi-Directional LSTM (BiLSTM) and a non-linear Support Vector Machine (SVM) trained on the same dataset, enabling comprehensive model comparison.

Before running the fine-tuning script, ensure you have the following installed:
•	Python 3.8+
•	PyTorch
•	Transformers (Hugging Face)
•	Datasets (Hugging Face)
•	scikit-learn
•	pandas
•	tqdm
•	CUDA (if training on a GPU)
Install dependencies using:
pip install torch transformers datasets scikit-learn pandas tqdm
For GPU acceleration, ensure you have the correct CUDA version installed and verify it with:
python -c "import torch; print(torch.cuda.is_available())"
Dataset Preparation
Ensure your dataset is formatted as a CSV or JSON file with labeled text samples. The dataset should have the following format:
Text	  label
Sample Swahili sentence1	    0
Sample Swahili sentence 2	    1
Modify the fine-tuning script to load the dataset correctly:
from datasets import load_dataset
 dataset = load_dataset("csv", data_files={"train": "path/to/dataset.csv", "test": "path/to/test_dataset.csv"})
Fine-Tuning
Run the fine-tuning script with the following command:
python fine_tune_swabert.py --train_data path/to/dataset.csv --test_data path/to/test_dataset.csv --epochs 3 --batch_size 16 --learning_rate 5e-5 --output_dir saved_model/
Customizing Hyperparameters
You can modify training parameters such as:
•	--epochs: Number of training epochs (default: 3)
•	--batch_size: Training batch size (default: 16)
•	--learning_rate: Learning rate for the optimizer (default: 5e-5)
•	--max_seq_length: Maximum token length per input text (default: 512)
Evaluation
After training, evaluate the model using:
python evaluate_model.py --model_path saved_model/ --test_data path/to/test_dataset.csv
Metrics Reported
The evaluation script reports key performance metrics:
•	Accuracy
•	Precision
•	Recall
•	F1-score
Model Output
•	Fine-tuned model weights will be saved in the saved_model/ directory.
•	Logs and performance metrics will be stored in logs/.
•	Predictions on test data will be saved in predictions.csv.
Using the Fine-Tuned Model
To use the trained model for inference, run:
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
model_path = "saved_model/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=-1).item()

print(predict("Mfano wa sentensi ya Kiswahili."))
Troubleshooting
Common Issues and Solutions
•	CUDA Out of Memory: Reduce batch size or use gradient accumulation.
•	Dataset Loading Errors: Ensure the dataset file path is correct and formatted properly.
•	Low Performance Metrics: Tune hyperparameters, increase dataset size, or improve preprocessing steps.
Acknowledgments
This project builds on the SwahBERT model and the Hugging Face Transformers library. Ensure proper citation when using in research.
License
This project is released under Creative commons By 4.0 license.
