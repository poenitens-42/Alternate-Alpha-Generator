# src/sentiment/sentiment_model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class FinBertSentiment:
    def __init__(self):
        """Initialize FinBERT model for financial sentiment analysis."""
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        # Set model to evaluation mode (disables dropout, batchnorm updates)
        self.model.eval()
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Define labels
        self.labels = ["negative", "neutral", "positive"]

    def predict(self, text: str):
        """
        Return sentiment score: positive, neutral, negative with probabilities.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment probabilities
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # Move inputs to same device as model
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
        
        # Convert to numpy/cpu and extract probabilities
        probs_cpu = probs.cpu().numpy()[0]  # Move to CPU and convert to numpy
        
        # Create scores dictionary - FIXED: using .item() or direct numpy conversion
        scores = {
            self.labels[i]: float(probs_cpu[i])  # Convert numpy float to Python float
            for i in range(3)
        }
        
        return scores
    
    def predict_batch(self, texts: list):
        """
        Predict sentiment for a batch of texts (more efficient).
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment score dictionaries
        """
        if not texts:
            return []
        
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move inputs to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Disable gradient calculation
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
        
        # Convert to numpy/cpu
        probs_cpu = probs.cpu().numpy()
        
        # Create results
        results = []
        for i in range(len(texts)):
            scores = {
                self.labels[j]: float(probs_cpu[i][j])
                for j in range(3)
            }
            results.append(scores)
        
        return results
    
    def get_sentiment_score(self, text: str, weighted=False):
        """
        Get a single sentiment score (positive - negative).
        
        Args:
            text: Input text
            weighted: If True, apply weights (negative=-1, neutral=0, positive=1)
                     If False, just return positive - negative
        
        Returns:
            Float sentiment score
        """
        scores = self.predict(text)
        
        if weighted:
            # Weighted score: negative=-1, neutral=0, positive=1
            return scores["positive"] - scores["negative"]
        else:
            # Simple positive-negative difference
            return scores["positive"] - scores["negative"]
    
    def analyze_sentiment_trend(self, texts: list):
        """
        Analyze sentiment trend across multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary with aggregate sentiment statistics
        """
        if not texts:
            return {"average_sentiment": 0, "sentiment_std": 0, "positive_ratio": 0}
        
        scores = self.predict_batch(texts)
        
        # Calculate weighted sentiment scores
        sentiment_scores = [
            s["positive"] - s["negative"] for s in scores
        ]
        
        # Calculate statistics
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        sentiment_std = (sum((s - avg_sentiment) ** 2 for s in sentiment_scores) / len(sentiment_scores)) ** 0.5
        
        # Count positive/negative dominance
        positive_count = sum(1 for s in scores if s["positive"] > s["negative"])
        positive_ratio = positive_count / len(scores)
        
        return {
            "average_sentiment": float(avg_sentiment),
            "sentiment_std": float(sentiment_std),
            "positive_ratio": float(positive_ratio),
            "sample_size": len(texts),
            "sentiment_scores": [float(s) for s in sentiment_scores]
        }


# Utility function for backward compatibility
def get_sentiment_model():
    """Get or create a FinBertSentiment instance (singleton pattern)."""
    if not hasattr(get_sentiment_model, "instance"):
        get_sentiment_model.instance = FinBertSentiment()
    return get_sentiment_model.instance


# Example usage
if __name__ == "__main__":
    # Test the model
    model = FinBertSentiment()
    
    test_texts = [
        "Stocks are rising rapidly today!",
        "Market crash expected due to economic concerns.",
        "The company reported mixed earnings results.",
        "Investors are optimistic about future growth.",
        "Trading volume remains stable with slight volatility."
    ]
    
    print("Testing FinBERT sentiment analysis:")
    print("=" * 50)
    
    for text in test_texts:
        scores = model.predict(text)
        sentiment_score = model.get_sentiment_score(text, weighted=True)
        
        print(f"\nText: {text[:50]}...")
        print(f"Negative: {scores['negative']:.3f}")
        print(f"Neutral:  {scores['neutral']:.3f}")
        print(f"Positive: {scores['positive']:.3f}")
        print(f"Score:    {sentiment_score:.3f}")
    
    # Test batch processing
    print("\n" + "=" * 50)
    print("Batch processing test:")
    batch_results = model.predict_batch(test_texts[:3])
    for i, result in enumerate(batch_results):
        print(f"Text {i+1}: Pos={result['positive']:.3f}, Neg={result['negative']:.3f}")
    
    # Test trend analysis
    print("\n" + "=" * 50)
    print("Trend analysis:")
    trend = model.analyze_sentiment_trend(test_texts)
    print(f"Average sentiment: {trend['average_sentiment']:.3f}")
    print(f"Sentiment std:     {trend['sentiment_std']:.3f}")
    print(f"Positive ratio:    {trend['positive_ratio']:.3f}")