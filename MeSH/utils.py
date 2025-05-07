
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def generate_examples(texts, labels):
    for i, text in enumerate(texts):
        label_integers = labels[i].tolist()
        label_floats = []

        for num in label_integers:
            if num not in [0, 1]:
                raise ValueError(f"Label should be binary (0,1). Got: {num}")

            label_floats.append(float(num))
        
        yield {"text": text, "labels": label_floats}


def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1}

# we need to nested tokenize so that it can pass the tokenizer for that instance and not the entire dataset
# or turn into a class.
def make_tokenize(tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"],max_length=512, truncation=True, padding=False)
    return tokenize
