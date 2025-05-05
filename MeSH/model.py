import json
import os
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset# , load_metrics
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments,DataCollatorWithPadding


#data format
"""
 {"articles": [
{"abstractText":"To clarify the role of endothelial cells in the pathogenesis of vasculitis affecting peripheral nerve and skeletal muscle, the endothelial expression of adhesion molecules and major histocompatibility antigens (MHC) in different vasculitic syndromes were studied, and related to the presence of anti-endothelial cell antibodies (AECA). Increased expression of the intercellular adhesion molecule ICAM-1 in vasculitic lesions in nerve and muscle was shown, and this was associated with increased expression of MHC class I and II antigens. AECA were detected in low titre in only a minority of patients. The findings suggest that endothelial cells have a critical role in mediating the tissue injury in vasculitis affecting nerve and muscle and that the process is triggered by cellular and not antibody-mediated mechanism in the majority of patients.","journal":"Journal of neurology, neurosurgery, and psychiatry","meshMajor":["Aged","Autoantibodies","Biopsy","Cell Adhesion Molecules","E-Selectin","Endothelium, Vascular","Female","Humans","Immunoenzyme Techniques","Intercellular Adhesion Molecule-1","Lymphocyte Function-Associated Antigen-1","Major Histocompatibility Complex","Male","Middle Aged","Muscles","Neutrophils","Peripheral Nerves","Vasculitis"],"pmid":"1372348","title":"Endothelial cell activation in vasculitis of peripheral nerve and skeletal muscle.","year":"1992"},
"""
def load_data(file_path):
    data_files = {}

    with open(file_path, "r") as f:

        data = json.load(f)

        # articles jas all the the data
        for item in data['articles']:
            pmid = item.get("pmid")
             
            abstract = item.get("abstractText")
            title = item.get("title")        
            labels = item.get("meshMajor")

            text = [title + " " + abstract]
            save_it = (text, labels)

            data_files[pmid] = save_it

    return data_files

def set_up():
    # set paths
    root = Path(__file__).parent.parent
    training_set_path = root / 'training-set-100000.json'
    test_set_path = root / 'test-set-20000-rev2.json'

    # use our load data function which return a dict with pmid and key and a tuple of text and labels
    train_data = load_data(training_set_path)
    test_data = load_data(test_set_path)

    # get the labels out of the dicts
    train_labels = []
    for pmic, (text, labels) in train_data.items():
        train_labels.append(labels)
    
    test_labels = []
    for pmic, (text, labels) in test_data.items():
        test_labels.append(labels)


    # multi label binarizer
    mlb = MultiLabelBinarizer(train_labels)
    train_y = mlb.fit_transform(train_labels)
    test_y = mlb.transform(test_labels)

    train_texts = []
    for pmic, (text, labels) in train_data.items():
        train_texts.append(text[0])
    
    test_texts = []
    for pmic, (text, labels) in test_data.items():
        test_texts.append(text[0])

    #make dict then we can pass this to Dataset
    train_dict = {'text': train_texts, 'labels': list(train_y)}
    test_dict = {'text': test_texts, 'labels': list(test_y)}

    train_dataset = Dataset.from_dict(train_dict)
    test_dataset = Dataset.from_dict(test_dict)

    # bert will tokenize the data biridectionally
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    collector = DataCollatorWithPadding(tokenizer) 

    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, padding=False)
    
    # tokenize the data
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    size = len(mlb.classes_)
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=size, problem_type='multi_label_classification')

    return bert_model, train_dataset, test_dataset,tokenizer, collector, mlb

def compute_metrics(pred):
    logits, labels = pred
    predictions = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    return {'f1': f1,'precision': precision,'recall': recall}


def set_up_predictions(trainer, mlb):
    """
    { "documents": [
    {"pmid": "16854706", "title": "The use of seat belts in cars with smart seat belt reminders--results of an observational study.", "abstractText": "UNLABELLED: Recently, smart seat belt reminders (SBR) have been introduced in cars. By increasingly reminding drivers and passengers if they are not using the seat belt, the intention is to increase the belt use to almost 100%. OBJECTIVE: The objective was to study if there were differences in driver's seat belt use between cars with and without SBR. METHODS: Drivers of cars with and without SBR were observed concerning seat belt use. The case (cars with SBR) and the control group (cars without SBR) were similar in all major aspects except SBR. In all, more than 3,000 drivers were observed in five cities in Sweden. RESULTS: In cars without SBR, 82.3 percent of the drivers used the seat belt, while in cars with SBR, the seat belt use was 98.9 percent. The difference was significant. In cars with mild reminders, the use was 93.0 percent. CONCLUSION: It is concluded, that if the results can be generalised to the whole car population this would have a dramatic impact on the number of fatally and seriously injured car occupants."},
    {"pmid": "12943287", "title": "Alignment in total knee arthroplasty following failed high tibial osteotomy.", "abstractText": "In total knee arthroplasty (TKA) following failed high tibial osteotomy, the mechanical axis does not intersect the center of the tibial component if the tibia has been resected perpendicular to the anatomical axis. Therefore, tibial resection referencing the predicted postoperative mechanical axis instead of the tibial shaft axis is advocated. To obtain the optimal tibial resection, characteristics of the tibial proximal deformity were measured radiographically and predicted postoperative lower limb alignment was calculated using full-length, weight-bearing, lower limb anteroposterior radiographs. Two finite element analysis models also were examined. The proximal tibia was resected perpendicular to the tibial shaft axis in model 1, and perpendicular to the predicted postoperative tibial mechanical axis in model 2. When the proximal tibia was resected perpendicular to the tibial shaft axis, the predicted lower limb mechanical axis was significantly shifted medially to the center of the tibial joint surface. The results of the finite element analysis reflected the medial shift of the lower limb mechanical axis in model 1, where stresses were increased in the medial tibial compartment. Tibial resection referencing the predicted postoperative tibial mechanical axis, instead of the tibial shaft axis, should be performed, especially in cases with a deformed tibia."},
    
    """
    judge_path = Path(__file__).parent.parent / 'judge-set-10000-unannotated.json'

    data_files = {}

    with open(judge_path, 'r') as f:
        data = json.load(f)

        for items in data['documents']:
            pmid = items.get("pmid")
            abstract = items.get("abstractText")
            title = items.get("title")        
            text = [title + " " + abstract]
            save_it = (text, pmid)
            data_files[pmid] = save_it

    #set up the dict for the judge set
    judge_texts = []
    for pmid, (text, labels) in data_files.items():
        judge_texts.append(text[0])

    judge_dict = {'text': judge_texts}

    def tokenize_fn(batch):
        return trainer.tokenizer(batch['text'], truncation=True, padding=False)

    judge_dataset = Dataset.from_dict(judge_dict)
    judge_dataset = judge_dataset.map(tokenize_fn, batched=True)
    judge_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    preds = trainer.predict(judge_dataset).predictions
    binary_preds = (preds > 0).astype(int)


    # Save predictions
    output = {'documents': []}
    for doc, row in zip(data, binary_preds):
        labels = []
        for i, var in enumerate(row):
            # if there are any labels in the row then we add them
            if var:
                labels.append(mlb.classes_[i])

        output['documents'].append({'pmid': doc['pmid'], 'labels': labels})

    with open('predictions.json', 'w') as f:
        json.dump(output, f)

def main():
    
    model, train_ds, test_ds, collector, tokenizer, mlb = set_up()

    os.makedirs('./results', exist_ok=True)
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_total_limit=1,
        logging_steps=500,
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

