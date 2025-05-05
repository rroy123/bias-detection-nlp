import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and label your data (small subset for fast training)
def load_dataset(sample_size=None):
    file_label_pairs = [
        ("allsides_data/political_articles_left.csv", "Left"),
        ("allsides_data/political_articles_right.csv", "Right"),
        ("allsides_data/political_articles_center.csv", "Neutral")
    ]

    dfs = []
    for path, label in file_label_pairs:
        df = pd.read_csv(path)
        if "text" in df.columns:
            df = df.rename(columns={"text": "content"})
        df['label'] = label
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all[['content', 'label']].dropna()
    return df_all

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average='macro')
    }

if __name__ == "__main__":
    df = load_dataset()
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['content'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

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

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3).to(device)

    training_args = TrainingArguments(
        output_dir="./results_distilbert",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs_distilbert",
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

    preds = trainer.predict(val_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)
    print("\n=== Classification Report ===")
    print(classification_report(val_labels, y_pred, target_names=label_encoder.classes_))

    # ✅ Save model and tokenizer
    model.save_pretrained("./distilbert_bias_model")
    tokenizer.save_pretrained("./distilbert_bias_model")
    print("✅ Model and tokenizer saved to ./distilbert_bias_model")