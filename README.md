# NLP-PubMed

Automate the process of assigning MeSH terms to articles from PubMed using a Biomed-BERT multi-label classifier.

## Project Overview

This repository implements a pipeline to:

1. Load PubMed articles from JSON datasets.
2. Train a multi-label classification model (PubMedBERT) to predict MeSH major terms.
3. Evaluate performance on a held-out test set (20,000 articles).
4. Predict MeSH labels for an un-annotated judge set (10,000 articles) and save results.

## Data

Datasets are publicly availble at: [https://drive.google.com/drive/folders/1MlIDbsrJS5Sf0B48MTfvDOurMjhrQU-_?usp=share_link](datasets)
Please download the following files from the link:

- `training-set-100000.json`
- `test-set-20000-rev2.json`
- `judge-set-10000-unannotated.json`

## Set-up

Follow these steps in your terminal to get started:

1. **Clone the repository**
```bash
git clone git@github.com:ramosv/NLP-PubMed.git
```

2. **Navigate into the project directory**
```bash
cd NLP-PubMed
```

3. **Create a virtual environment**
```bash
python -m venv .venv
```

4. **Activate the virtual environment**
```bash
source .venv/Scripts/activate
```

5. **Install dependencies**
```bash
pip install -r requirements.txt
```
Some requirements may need manual installations. I left comments in the requirements.txt

6. **Test the pipeline**
```bash
python MeSH/model.py
```

model.py will:

- Load training and test sets
- Train a multi‑label classification model (BERT base uncased) to predict MeSH major terms  
- Compute accuracy, micro-precision, micro-recall, and micro-F1 on the test set
- Print evaluation metrics and save the best checkpoint in the `results/` directory

### Judge-Set Predictions

After training, the script automatically runs inference on the judge set and saves it as a json file.

Labels are thresholded at sigmoid ≥ 0.5 and converted using the `MultiLabelBinarizer`.

## Results

Evaluation on the 20,000-article test set (with 3 epochs):

```bash
{'eval_loss': 1.4463,
 'eval_accuracy': 0.0,
 'eval_micro_precision': 0.0008774,
 'eval_micro_recall': 0.5963522,
 'eval_micro_f1': 0.00175225
}
```

Note: Initially, every prediction was "human". The mistake was not applying the sigmoid to the logits.

The model has high recall but very low precision. After discussing with classmates, I realized many were facing similar issues, were accuracy was very low and predicitons were all over the place. I went back and conducted further data exploration see `data.ipynb`.

The data is dense with significant variation, so I explored using the journal as a high-level hierarchy to assess whether certain words appear more frequently in specific journals.

Here are plots showing the most common journals and their top words:
- [plot1](plots/1.png)
- [plot2](plots/2.png)
- [plot3](plots/3.png)
- [plot4](plots/4.png)
- [plot5](plots/5.png)
- [plot6](plots/6.png)
- [plot7](plots/7.png)

I will continue working on this by incorporating the journal hierarchy into the training process, as a way to help the model make more informed predictions.

### References
- BERT model: `bert-base-uncased




