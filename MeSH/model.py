import json
import os
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Features, Value, Sequence, Dataset
from utils import compute_metrics, generate_examples, make_tokenize
import numpy as np

import torch
print(torch.cuda.is_available())

#data format from json file
"""
 {"articles": [
{"abstractText":"To clarify the role of endothelial cells in the pathogenesis of vasculitis affecting peripheral nerve and skeletal muscle, the endothelial expression of adhesion molecules and major histocompatibility antigens (MHC) in different vasculitic syndromes were studied, and related to the presence of anti-endothelial cell antibodies (AECA). Increased expression of the intercellular adhesion molecule ICAM-1 in vasculitic lesions in nerve and muscle was shown, and this was associated with increased expression of MHC class I and II antigens. AECA were detected in low titre in only a minority of patients. The findings suggest that endothelial cells have a critical role in mediating the tissue injury in vasculitis affecting nerve and muscle and that the process is triggered by cellular and not antibody-mediated mechanism in the majority of patients.","journal":"Journal of neurology, neurosurgery, and psychiatry","meshMajor":["Aged","Autoantibodies","Biopsy","Cell Adhesion Molecules","E-Selectin","Endothelium, Vascular","Female","Humans","Immunoenzyme Techniques","Intercellular Adhesion Molecule-1","Lymphocyte Function-Associated Antigen-1","Major Histocompatibility Complex","Male","Middle Aged","Muscles","Neutrophils","Peripheral Nerves","Vasculitis"],"pmid":"1372348","title":"Endothelial cell activation in vasculitis of peripheral nerve and skeletal muscle.","year":"1992"},
"""
def load_data(file_path):
    data_files = {}

    with open(file_path, "r") as f:

        data = json.load(f)

        print(f"Top-level keys in JSON: {list(data.keys())}")

        # articles jas all the the data
        if "articles" in data:
            items = data["articles"]
        elif "documents" in data:
            items = data["documents"]
        else:
            raise ValueError(f"No key found in {file_path}")
        
        for item in items:
            pmid = item.get("pmid")
            pmid = str(pmid)
             
            abstract = item.get("abstractText") or ""
            title = item.get("title") or ""
            labels = item.get("meshMajor")

            text = title + " " + abstract
            save_it = (text, labels)

            data_files[pmid] = save_it

    return data_files

def set_up():
    # set paths
    root = Path(__file__).parent.parent
    training_set_path = root / 'dataset/training-set-100000.json'
    test_set_path = root / 'dataset/test-set-20000-rev2.json'

    # use our load data function which return a dict with pmid and key and a tuple of text and labels
    train_data = load_data(training_set_path)
    print(f"Training set size: {len(train_data)}")
    test_data = load_data(test_set_path)
    print(f"Test set size: {len(test_data)}")

    # get the labels out of the dicts
    train_labels = []
    for pmic, (text, labels) in train_data.items():
        train_labels.append(labels)
    
    test_labels = []
    for pmic, (text, labels) in test_data.items():
        test_labels.append(labels)


    # multi label binarizer
    mlb = MultiLabelBinarizer()
    train_y = mlb.fit_transform(train_labels)
    test_y = mlb.transform(test_labels)

    train_texts = []
    for pmic, (text, labels) in train_data.items():
        train_texts.append(text)
    
    test_texts = []
    for pmic, (text, labels) in test_data.items():
        test_texts.append(text)
    
    features = Features({"text": Value("string"),"labels": Sequence(Value("float32"))})
    
    def train_gen():
        return generate_examples(train_texts, train_y)

    def test_gen():
        return generate_examples(test_texts, test_y)
    
    train_dataset = Dataset.from_generator(train_gen, features=features)
    test_dataset = Dataset.from_generator(test_gen, features=features)

    # bert will tokenize the data biridectionally
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    collector = DataCollatorWithPadding(tokenizer)

    # tokenize the data instance by instance
    tokenized = make_tokenize(tokenizer)

    # map the tokenized function to the dataset
    train_dataset = train_dataset.map(tokenized, batched=True)
    test_dataset = test_dataset.map(tokenized, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    size = len(mlb.classes_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=size, problem_type='multi_label_classification')
    bert_model.to(device)

    return bert_model, train_dataset, test_dataset,tokenizer, collector, mlb

def set_up_predictions(trainer, mlb):
    """
    { "documents": [
    {"pmid": "16854706", "title": "The use of seat belts in cars with smart seat belt reminders--results of an observational study.", "abstractText": "UNLABELLED: Recently, smart seat belt reminders (SBR) have been introduced in cars. By increasingly reminding drivers and passengers if they are not using the seat belt, the intention is to increase the belt use to almost 100%. OBJECTIVE: The objective was to study if there were differences in driver's seat belt use between cars with and without SBR. METHODS: Drivers of cars with and without SBR were observed concerning seat belt use. The case (cars with SBR) and the control group (cars without SBR) were similar in all major aspects except SBR. In all, more than 3,000 drivers were observed in five cities in Sweden. RESULTS: In cars without SBR, 82.3 percent of the drivers used the seat belt, while in cars with SBR, the seat belt use was 98.9 percent. The difference was significant. In cars with mild reminders, the use was 93.0 percent. CONCLUSION: It is concluded, that if the results can be generalised to the whole car population this would have a dramatic impact on the number of fatally and seriously injured car occupants."},
    {"pmid": "12943287", "title": "Alignment in total knee arthroplasty following failed high tibial osteotomy.", "abstractText": "In total knee arthroplasty (TKA) following failed high tibial osteotomy, the mechanical axis does not intersect the center of the tibial component if the tibia has been resected perpendicular to the anatomical axis. Therefore, tibial resection referencing the predicted postoperative mechanical axis instead of the tibial shaft axis is advocated. To obtain the optimal tibial resection, characteristics of the tibial proximal deformity were measured radiographically and predicted postoperative lower limb alignment was calculated using full-length, weight-bearing, lower limb anteroposterior radiographs. Two finite element analysis models also were examined. The proximal tibia was resected perpendicular to the tibial shaft axis in model 1, and perpendicular to the predicted postoperative tibial mechanical axis in model 2. When the proximal tibia was resected perpendicular to the tibial shaft axis, the predicted lower limb mechanical axis was significantly shifted medially to the center of the tibial joint surface. The results of the finite element analysis reflected the medial shift of the lower limb mechanical axis in model 1, where stresses were increased in the medial tibial compartment. Tibial resection referencing the predicted postoperative tibial mechanical axis, instead of the tibial shaft axis, should be performed, especially in cases with a deformed tibia."},
    
    """
    judge_path = Path(__file__).parent.parent / 'dataset/judge-set-10000-unannotated.json'

    data_files = {}

    with open(judge_path, 'r') as f:
        data = json.load(f)

        for items in data["documents"]:
            pmid = items.get("pmid")
            pmid = str(pmid)
            abstract = items.get("abstractText") or ""
            title = items.get("title") or ""
            text = title + " " + abstract
            save_it = (text, pmid)
            data_files[pmid] = save_it

    #set up the dict for the judge set
    judge_texts = []
    for pmid, (text, labels) in data_files.items():
        judge_texts.append(text)

    # using the same features as the training set
    judge_dict = {'text': judge_texts}
    judge_dataset = Dataset.from_dict(judge_dict)
    tokenized = make_tokenize(trainer.tokenizer)

    judge_dataset = judge_dataset.map(tokenized, batched=True)
    judge_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    logits = trainer.predict(judge_dataset).predictions
    probs = 1 / (1 + np.exp(-logits))

    K = 5
    top_k = np.argsort(probs, axis=1)[:, -K:]
    binary = np.zeros_like(probs, dtype=int)
    for i, idxs in enumerate(top_k):
        binary[i, idxs] = 1

    # get the labels for the top k
    output = {"documents": []}

    for pmid, row in zip(data_files.keys(), binary):
        labels = []
        for i in range(len(row)):
            if row[i] == 1:
                labels.append(mlb.classes_[i])
        output["documents"].append({"pmid": pmid, "labels": labels})

    with open('predictions.json', 'w') as f:
        json.dump(output, f)

def train():
    
    model, train_ds, test_ds, tokenizer, collector, mlb = set_up()

    os.makedirs('./results', exist_ok=True)
    training_args = TrainingArguments(
        output_dir='./results',
        do_train=True,
        do_eval=True,
        eval_steps= 1000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        dataloader_num_workers=16,
        num_train_epochs=3,
        save_total_limit=1,
        logging_steps=500,
        fp16=True,
        gradient_checkpointing=True
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=collector,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print(trainer.evaluate())
    set_up_predictions(trainer, mlb)

def evaluate():
    model, train_ds, test_ds, tokenizer, collector, mlb = set_up()

    checkpoint_dir = "./results/checkpoint-9375"
    model = BertForSequenceClassification.from_pretrained(
        checkpoint_dir,
        num_labels=len(mlb.classes_),
        problem_type="multi_label_classification"
    ).to(model.device)

    training_args = TrainingArguments(
        output_dir='./results',
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=32,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=collector,
        compute_metrics=compute_metrics
    )

    print(trainer.evaluate())
    set_up_predictions(trainer, mlb)

if __name__ == "__main__":
    train()