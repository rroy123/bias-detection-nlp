import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load binary-labeled PARTIAL dataset (small for fast training)
def load_partial_binary_dataset(sample_size_per_class=None):
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
        # df = df.sample(n=min(sample_size_per_class, len(df)), random_state=42)  # limit size
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
    df = load_partial_binary_dataset()  # ~3K articles total
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])  # Left: 0, Right: 1

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['content'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

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

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

    training_args = TrainingArguments(
        output_dir="./binary_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./binary_logs",
        logging_steps=10
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
    print("\n=== Binary Classification Report ===")
    print(classification_report(val_labels, y_pred, target_names=label_encoder.classes_))

    # Save model and tokenizer
    model.save_pretrained("./distilbert_binary_model")
    tokenizer.save_pretrained("./distilbert_binary_model")
    print("\n\u2705 Binary classifier saved to ./distilbert_binary_model")