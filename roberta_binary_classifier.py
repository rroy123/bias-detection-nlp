import os
import pandas as pd
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    RobertaTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset


os.environ["WANDB_DISABLED"] = "true"


class WeightedRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )


def load_binary_dataset():
    file_label_pairs = [
        ("allsides_data/political_articles_left.csv", "Left"),
        ("allsides_data/political_articles_right.csv", "Right")
    ]
    dfs = []
    for path, label in file_label_pairs:
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='mac_roman')  
        df = df.rename(columns={"text": "content"}) if "text" in df.columns else df
        df["label"] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)[["content", "label"]].dropna()



def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_binary_dataset()
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])
    print(f"Dataset size: {len(df)}")
    print("Label counts:", Counter(df["label"]))

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["content"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
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

    weights = torch.tensor([1.0, 1.0])  
    config = RobertaConfig.from_pretrained("roberta-base", num_labels=2)
    model = WeightedRobertaForSequenceClassification.from_pretrained(
        "roberta-base", config=config, class_weights=weights
    ).to(device)

    training_args = TrainingArguments(
        output_dir="./roberta_binary_results_full",
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=200,
        logging_dir="./roberta_binary_logs_full",
        logging_steps=50,
        save_strategy="no",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    preds = trainer.predict(val_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)

    print("Predicted label distribution:", Counter(y_pred))
    print("True label distribution:", Counter(val_labels))
    print("\n=== Full RoBERTa Binary Classification Report ===")
    print(classification_report(val_labels, y_pred, target_names=label_encoder.classes_))

    model.save_pretrained("./roberta_binary_model_full")
    tokenizer.save_pretrained("./roberta_binary_model_full")
    print("Model saved to ./roberta_binary_model_full")

if __name__ == "__main__":
    main()
