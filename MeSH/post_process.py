import json
import pandas as pd
from pathlib import Path
from collections import Counter

def prune_predictions(predictions_path, mesh_journal_path, quartiles_path, output_path):
    f = open(predictions_path)
    data = json.load(f)
    f.close()
    docs = data["documents"]

    mmj = pd.read_csv(mesh_journal_path)
    term_journal = {}
    term_count = {}
    for i in range(len(mmj)):
        row = mmj.iloc[i]
        term_journal[row["word"]] = row["journal"]
        term_count[row["word"]] = row["in_that_journal"]

    jq = pd.read_csv(quartiles_path)
    journal_med = {}
    for i in range(len(jq)):
        row = jq.iloc[i]
        journal_med[row["journal"]] = int(row["median"])

    all_meds = []
    for val in journal_med.values():
        all_meds.append(val)
    global_med = int(pd.Series(all_meds).median())

    pruned = []
    for doc in docs:
        pmid = doc["pmid"]
        labels = doc["labels"]

        votes = []
        for label in labels:
            if label in term_journal:
                votes.append(term_journal[label])
        if len(votes) > 0:
            cnt = Counter(votes)
            guessed = cnt.most_common(1)[0][0]
            K = journal_med.get(guessed, global_med)
        else:
            K = global_med

        sorted_labels = []
        for label in labels:
            count = term_count.get(label, 0)
            sorted_labels.append((count, label))
        sorted_labels.sort(reverse=True)
        topk = []

        for i in range(K):
            if i < len(sorted_labels):
                topk.append(sorted_labels[i][1])

        pruned.append({"pmid": pmid, "labels": topk})

    # write out
    out = {"documents": pruned}
    f2 = open(output_path, "w")
    json.dump(out, f2, separators=(",",":"))
    f2.close()

if __name__ == "__main__":
    base = Path("/home/vicente/Github/NLP-PubMed/MeSH")
    prune_predictions( base.parent/"predictions05.json", base/"mesh_major_journal.csv", base/"journal_quartiles.csv", base/"predictions_pruned.json")
