from transformers import BertForSequenceClassification, BertTokenizerFast


trainer.model.saved_pretrained("./bert_bias_model")
tokenizer.save_pretrained("./bert_bias_model")

print("model and tokenizer saved to ./bert_bias_model")