
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
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

    exact_match_accuracy = (preds == labels).all(axis=1).mean()
    sample_accs = []

    for true_row, pred_row in zip(labels, preds):
        true_count = true_row.sum()

        if true_count > 0:
            correct = np.logical_and(true_row, pred_row).sum()
            sample_accs.append(correct / true_count)

    if sample_accs:
        sample_accuracy = np.mean(sample_accs)
    else:
        sample_accuracy = 0.0

    # return a dictionary with the metrics
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match_accuracy': exact_match_accuracy,
        'sample_accuracy': sample_accuracy
    }

def label_cut_off(mlb, probs_dev, labels_dev):
    #from 0.1 to 0.9 in steps of 0.05
    cands = np.linspace(0.1, 0.9, 17)

    best_thresholds = {}
    for j, term in enumerate(mlb.classes_):
        best_f1, best_t = -1, 0.5

        for t in cands:
            preds_j = (probs_dev[:, j] >= t).astype(int)
            f1 = f1_score(labels_dev[:, j], preds_j, zero_division=0)

            if f1 > best_f1:
                best_f1, best_t = f1, t

        best_thresholds[term] = best_t

    return best_thresholds

# we need to nested tokenize so that it can pass the tokenizer for that instance and not the entire dataset
# or turn into a class.
def make_tokenize(tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"],max_length=512, truncation=True, padding=False)
    return tokenize
