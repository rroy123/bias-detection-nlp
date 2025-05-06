import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load binary-labeled dataset
def load_binary_dataset():
    file_label_pairs = [
        ("allsides_data/political_articles_left.csv", "Left"),
        ("allsides_data/political_articles_right.csv", "Right"),
    ]

    dfs = []
    for path, label in file_label_pairs:
        df = pd.read_csv(path)
        if "text" in df.columns:
            df = df.rename(columns={"text": "content"})
        df['label'] = label
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all[['content', 'label']]

# Metric computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average='macro')
    }

if __name__ == "__main__":
    df = load_binary_dataset()
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])  # Left: 0, Right: 1

    # Reduce to 3000 articles for speed (1500 each side)
    df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=750, random_state=42)).reset_index(drop=True)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['content'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels
    })
    val_dataset = Dataset.from_dict({
        "input_ids": val_encodings["input_ids"],
        "attention_mask": val_encodings["attention_mask"],
        "labels": val_labels
    })

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)

    training_args = TrainingArguments(
        output_dir="./roberta_binary_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./roberta_binary_logs",
        logging_steps=10,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Final report
    preds = trainer.predict(val_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)
    

    print("Predicted label distribution:", Counter(y_pred))
    print("True label distribution:", Counter(val_labels))

    print("\n=== RoBERTa Binary Classification Report ===")
    print(classification_report(val_labels, y_pred, target_names=label_encoder.classes_))

    # Save model and tokenizer
    model.save_pretrained("./roberta_binary_model")
    tokenizer.save_pretrained("./roberta_binary_model")
    print("\n✅ RoBERTa binary classifier saved to ./roberta_binary_model")
