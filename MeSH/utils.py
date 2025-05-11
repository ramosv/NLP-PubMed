import json
from pathlib import Path
from collections import Counter

import pandas as pd

def prune_predictions(
    predictions_path: Path,
    mesh_major_journal_path: Path,
    journal_quartiles_path: Path,
    output_path: Path
):
    # 1) load your big-predictions.json
    with open(predictions_path, "r") as f:
        preds = json.load(f)["documents"]

    # 2) load mesh_major_journal.csv
    mmj = pd.read_csv(mesh_major_journal_path)
    term_journal   = dict(zip(mmj.word,            mmj.journal))
    term_in_journal = dict(zip(mmj.word,           mmj.in_that_journal))

    # 3) load journal_quartiles.csv
    jq = pd.read_csv(journal_quartiles_path)
    journal_median = dict(zip(jq.journal, jq.median))
    global_median  = int(jq.median.median())

    pruned = []
    for doc in preds:
        pmid   = doc["pmid"]
        labels = doc["labels"]

        # guess the journal by vote over term_journal:
        journals = [term_journal[l] for l in labels if l in term_journal]
        if journals:
            guessed_journal = Counter(journals).most_common(1)[0][0]
            K = journal_median.get(guessed_journal, global_median)
        else:
            guessed_journal = None
            K = global_median

        # sort by how common each term is in that journal
        labels_sorted = sorted(
            labels,
            key=lambda l: term_in_journal.get(l, 0),
            reverse=True
        )

        pruned_labels = labels_sorted[:K]
        pruned.append({"pmid": pmid, "labels": pruned_labels})

    # 4) write out the pruned set
    with open(output_path, "w") as f:
        json.dump({"documents": pruned}, f, separators=(",",":"))

if __name__ == "__main__":
    root = Path("/home/vicente/Github/NLP-PubMed/MeSH")
    prune_predictions(
        predictions_path         = root.parent / "predictions04.json",
        mesh_major_journal_path  = root / "mesh_major_journal.csv",
        journal_quartiles_path   = root / "journal_quartiles.csv",
        output_path              = root / "predictions_pruned.json"
    )
