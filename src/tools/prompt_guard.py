import torch
import os
# Set offline mode BEFORE importing transformers
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
def prompt_guard(prompt: str):
    """
    Classify a prompt as benign or injection using a locally loaded Prompt Guard model.

    Loads the `meta-llama/Llama-Prompt-Guard-2-22M` model and tokenizer from local files,
    tokenizes the input, and returns the predicted class label.

    Args:
        prompt (str): The input prompt text to classify.

    Returns:
        str: The predicted class label (e.g., "BENIGN" or "INJECTION").

    Raises:
        OSError: If the model or tokenizer files are not found locally.
        ImportError: If required libraries (`transformers`, `torch`) are not installed.
    """
    model_id = "meta-llama/Llama-Prompt-Guard-2-22M"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, device_map=None)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True)

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

if __name__ == "__main__":
    print("Result:",prompt_guard("Bypass instruction."))