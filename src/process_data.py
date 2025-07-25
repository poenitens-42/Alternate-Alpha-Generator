import json
import regex as re
import spacy
from typing import List, Dict
from transformers import pipeline

# Load spaCy English NLP model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])

# Load FinBERT sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Ticker pattern: 2–5 uppercase letters, standalone
ticker_regex = re.compile(r"\b[A-Z]{2,5}\b")

def clean_text(text: str) -> str:
    """Remove links, emojis, and non-ASCII characters."""
    text = re.sub(r"http\S+", "", text)                  # remove links
    text = re.sub(r"[^\x00-\x7F]+", " ", text)           # remove non-ascii
    return text.strip()

def extract_tickers(text: str) -> List[str]:
    """Extract potential stock tickers using regex."""
    return list(set(ticker_regex.findall(text)))

def get_sentiment(text: str) -> float:
    """Get FinBERT sentiment score."""
    try:
        result = sentiment_pipeline(text[:512])[0]
        label = result["label"]
        score = result["score"]
        if label == "positive":
            return score
        elif label == "negative":
            return -score
        else:
            return 0.0
    except Exception:
        return 0.0

def enrich_post(post: Dict) -> Dict:
    """Clean, extract tickers, and analyze sentiment for one post."""
    title = post.get("title", "")
    clean = clean_text(title)
    tickers = extract_tickers(clean)
    sentiment = get_sentiment(clean)

    return {
        **post,
        "clean_title": clean,
        "tickers": tickers,
        "sentiment": sentiment,
    }

def enrich_posts(posts: List[Dict]) -> List[Dict]:
    """Process a list of Reddit posts."""
    return [enrich_post(post) for post in posts]

if __name__ == "__main__":
    # Load raw Reddit data
    with open("reddit_data.json", "r", encoding="utf-8") as f:
        raw_posts = json.load(f)

    # Process posts
    enriched = enrich_posts(raw_posts)

    # Save output
    with open("reddit_enriched.json", "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    print(f"✅ Processed {len(enriched)} posts -> reddit_enriched.json")
