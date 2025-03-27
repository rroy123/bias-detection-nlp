# poli_bias.py
import numpy as np
import pandas as pd
import nltk
import re
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

nlp = spacy.load("en_core_web_sm")

# Initialize global NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

LEFT_LEXICON = {
    "progressive", "diversity", "equity", "climate", "inclusion",
    "immigrant", "healthcare", "social", "regulation", "justice"
}

RIGHT_LEXICON = {
    "patriot", "freedom", "taxes", "border", "constitution",
    "traditional", "illegal", "liberty", "guns", "capitalism"
}

def extract_named_entities(text):
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"}]
    return ents


class BiasScorer:
    def __init__(self, left_lexicon, right_lexicon):
        self.left_lexicon = set(left_lexicon)
        self.right_lexicon = set(right_lexicon)
        self.sia = SentimentIntensityAnalyzer()

    def bias_score(self, tokens):
        """Scores how many left/right lexicon words appear in the token list."""
        tokens_set = set(tokens)
        left_score = len(tokens_set & self.left_lexicon)
        right_score = len(tokens_set & self.right_lexicon)
        return {
            "left_score": left_score,
            "right_score": right_score,
            "bias_score": left_score - right_score
        }

    def sentiment_score(self, raw_text):
        """Returns compound sentiment score from VADER."""
        return self.sia.polarity_scores(raw_text)["compound"]

    def label_bias(self, bias_score, threshold=1):
        """Returns a text label based on score difference."""
        if bias_score >= threshold:
            return "Left"
        elif bias_score <= -threshold:
            return "Right"
        else:
            return "Neutral"

def preprocess(text):
    """
    Tokenize, lowercase, remove punctuation, remove stopwords, lemmatize
    """
    # Lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return cleaned

def analyze_sentiment(text):
    """Return sentiment polarity scores from VADER."""
    return sia.polarity_scores(text)

def load_and_preprocess(csv_path):
    """Load CSV and preprocess content column."""
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')

    df['named_entities'] = df['content'].apply(extract_named_entities)
    df['tokens'] = df['content'].apply(preprocess)
    df['sentiment'] = df['content'].apply(analyze_sentiment)
    return df

def compute_tfidf_vectors(df, column='content'):
    """Compute TF-IDF vectors and return the matrix + feature names."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # tune max_features as needed
    tfidf_matrix = vectorizer.fit_transform(df[column])
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names, vectorizer

def top_tfidf_terms(tfidf_matrix, feature_names, doc_index, top_n=10):
    """Return top N terms with highest TF-IDF scores for one document."""
    row = tfidf_matrix[doc_index].toarray().flatten()
    top_indices = row.argsort()[-top_n:][::-1]
    return [(feature_names[i], row[i]) for i in top_indices]

def load_and_preprocess_multiple(file_label_pairs):
    dfs = []
    for path, label in file_label_pairs:
        df = pd.read_csv(path, encoding='ISO-8859-1')
        df['bias'] = label
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    df_all['named_entities'] = df_all['content'].apply(extract_named_entities)
    df_all['tokens'] = df_all['content'].apply(preprocess)
    df_all['sentiment'] = df_all['content'].apply(analyze_sentiment)
    return df_all


def plot_tfidf_pca(tfidf_matrix, labels):
    if tfidf_matrix.shape[0] < 2:
        print("⚠️ Not enough articles to perform PCA. Add more samples first.")
        return

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(tfidf_matrix.toarray())

    plt.figure(figsize=(8, 6))
    for label in set(labels):
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label, alpha=0.6)
    
    plt.title("PCA of TF-IDF Vectors")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def main():
    # Load multiple labeled datasets
    file_label_pairs = [
        ("political_articles_left.csv", "Left"),
        ("political_articles_right.csv", "Right"),
        ("political_articles_center.csv", "Neutral")
    ]
    df = load_and_preprocess_multiple(file_label_pairs)

    print("\n=== Preprocessed Tokens (First 100 characters) ===")
    print(df['tokens'].apply(lambda x: ' '.join(x[:20])))

    print("\n=== Sentiment Scores ===")
    print(df['sentiment'])

    tfidf_matrix, feature_names, vectorizer = compute_tfidf_vectors(df)

    # Show top TF-IDF terms for first article
    print("\n=== Top TF-IDF Terms (Article 0) ===")
    top_terms = top_tfidf_terms(tfidf_matrix, feature_names, 0)
    for term, score in top_terms:
        print(f"{term}: {score:.4f}")

    # Lexicon-based scoring
    scorer = BiasScorer(LEFT_LEXICON, RIGHT_LEXICON)
    df['bias_scores'] = df['tokens'].apply(scorer.bias_score)
    df['bias_label'] = df['bias_scores'].apply(lambda x: scorer.label_bias(x['bias_score']))

    # Plot using ground-truth bias from CSVs
    plot_tfidf_pca(tfidf_matrix, df['bias'].tolist())

    # Save full output for milestone use
    df.to_csv("processed_articles_with_features.csv", index=False)



if __name__ == '__main__':
    main()
