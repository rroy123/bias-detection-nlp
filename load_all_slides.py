import pandas as pd
import os

def fetch_and_split_allsides(input_csv="allsides_balanced_news_headlines-texts.csv", output_dir="allsides_data", min_content_len=300):
    os.makedirs(output_dir, exist_ok=True)

    
    df = pd.read_csv(input_csv)

    
    content_col = "text" if "text" in df.columns else "content"
    bias_col = "bias" if "bias" in df.columns else "bias_rating"

    df = df.dropna(subset=[content_col, bias_col])
    df = df[df[content_col].str.len() > min_content_len]

    
    df[bias_col] = df[bias_col].str.strip().str.capitalize()
    df = df[df[bias_col].isin(["Left", "Center", "Right"])]

    for label in ["Left", "Center", "Right"]:
        group = df[df[bias_col] == label]
        group[[content_col]].to_csv(f"{output_dir}/political_articles_{label.lower()}.csv", index=False, encoding="utf-8")
        print(f"Saved {len(group)} {label}-labeled articles.")

if __name__ == "__main__":
    fetch_and_split_allsides()
