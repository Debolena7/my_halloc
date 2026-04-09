# python scripts/prf1.py
import json
import re
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import nltk

from sklearn.metrics import multilabel_confusion_matrix
from collections import Counter, defaultdict
import pandas as pd

#import seaborn as sns
#import matplotlib.pyplot as plt

nltk.download('wordnet')
nltk.download('omw-1.4')

NUMBER_WORDS = {
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'a', 'an', 'the', 'some', 'several', 'many', 'few', 'couple', 'dozen', 'red', 'blue'
}

def clean_and_lemmatize(text):
    """
    Handles: splitting by comma, stripping whitespace, lowercasing, 
    removing empty gaps, numbers/quantifiers and lemmatizing plural to singular.
    """
    if not text or text.lower().strip() == "none":
        return {"__no_hallucination__"}
    
    lemmatizer = WordNetLemmatizer()
    
    raw_words = [w.strip().lower() for w in text.split(',')]
    
    processed_concepts = []
    for phrase in raw_words:
        phrase = re.sub(r'\d+', '', phrase)
        
        words = phrase.split()
        
        filtered = [
            lemmatizer.lemmatize(w, pos='n') 
            for w in words 
            if w not in NUMBER_WORDS and len(w) > 1
        ]
        
        if filtered:
            processed_concepts.append(" ".join(filtered))
        
    return set(processed_concepts) if processed_concepts else {"__no_hallucination__"}

def compute_metrics(file_path):
    y_true = []
    y_pred = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            true_set = clean_and_lemmatize(data.get('labels', ''))
            pred_set = clean_and_lemmatize(data.get('response', ''))
            
            y_true.append(list(true_set))
            y_pred.append(list(pred_set))

    mlb = MultiLabelBinarizer()
    mlb.fit(y_true + y_pred)
    
    bin_true = mlb.transform(y_true)
    bin_pred = mlb.transform(y_pred)

    print("\nClassification Report")
    print("-" * 40)

    print(classification_report(
        bin_true,
        bin_pred,
        target_names=mlb.classes_,
        zero_division=0,
        digits=4
    ))

    metrics = {}
    for average in ['micro', 'macro']:
        metrics[f'{average}_precision'] = precision_score(bin_true, bin_pred, average=average, zero_division=0)
        metrics[f'{average}_recall'] = recall_score(bin_true, bin_pred, average=average, zero_division=0)
        metrics[f'{average}_f1'] = f1_score(bin_true, bin_pred, average=average, zero_division=0)

    return metrics

def error_analysis(file_path, top_k=15):
    
    y_true = []
    y_pred = []
    raw_examples = []

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)

            true_set = clean_and_lemmatize(data.get('labels', ''))
            pred_set = clean_and_lemmatize(data.get('response', ''))

            y_true.append(list(true_set))
            y_pred.append(list(pred_set))

            raw_examples.append((true_set, pred_set, data))

    mlb = MultiLabelBinarizer()
    mlb.fit(y_true + y_pred)

    bin_true = mlb.transform(y_true)
    bin_pred = mlb.transform(y_pred)

    labels = mlb.classes_

    # confusion matrix per class
    mcm = multilabel_confusion_matrix(bin_true, bin_pred)

    concept_stats = []

    for i, label in enumerate(labels):
        tn, fp, fn, tp = mcm[i].ravel()

        precision = tp / (tp + fp) if (tp+fp) else 0
        recall = tp / (tp + fn) if (tp+fn) else 0

        concept_stats.append({
            "concept": label,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall
        })

    df = pd.DataFrame(concept_stats)

    # frequent errors
    fp_counter = Counter()
    fn_counter = Counter()

    for true_set, pred_set, _ in raw_examples:
        fp = pred_set - true_set
        fn = true_set - pred_set

        fp_counter.update(fp)
        fn_counter.update(fn)

    print("\nTop False Negatives (missed):")
    print(fn_counter.most_common(top_k))

    print("\nTop False Positives (hallucinated):")
    print(fp_counter.most_common(top_k))

    print("\nWorst recall:")
    print(df.sort_values("recall").head(top_k)[["concept","recall","fn"]])

    print("\nWorst precision:")
    print(df.sort_values("precision").head(top_k)[["concept","precision","fp"]])

    return df, bin_true, bin_pred

def global_confusion(bin_true, bin_pred):

    tp = ((bin_true == 1) & (bin_pred == 1)).sum()
    tn = ((bin_true == 0) & (bin_pred == 0)).sum()
    fp = ((bin_true == 0) & (bin_pred == 1)).sum()
    fn = ((bin_true == 1) & (bin_pred == 0)).sum()

    print("\nConfusion Matrix")
    print("------------------------")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TN: {tn}")

file_path = '/DATA/ai20resch11003/my_halloc/outputs/halloc_text/results-fin.jsonl'
results = compute_metrics(file_path)

print(f"Metrics:")
print("-" * 30)
for k, v in results.items():
    print(f"{k:18}: {v*100:.2f}")