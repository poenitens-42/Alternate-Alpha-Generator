import os
import json

from src.utils.logger import logger
from src.utils.file_utils import ensure_folder, save_json, timestamped_filename
from src.sentiment.sentiment_model import FinBertSentiment

# Initialize the sentiment model
sentiment_model = FinBertSentiment()

def process_reddit_data(input_path: str, output_folder: str):
    logger.info(f"Processing Reddit data from: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []

    # Use batch processing for better performance
    titles = [d.get("title", "") for d in data]
    
    # Get batch sentiment predictions
    sentiment_batch = sentiment_model.predict_batch(titles)
    
    for i, d in enumerate(data):
        title = d.get("title", "")
        sent = sentiment_batch[i] if i < len(sentiment_batch) else sentiment_model.predict(title)
        
        # Use the model's built-in weighted score calculation
        # OR use custom formula if you prefer
        sentiment_score = sentiment_model.get_sentiment_score(title, weighted=True)
        # Alternative: Use your custom formula
        # sentiment_score = sent["positive"] - sent["negative"] - 0.5 * sent["neutral"]

        processed.append({
            "title": title.lower(),
            "score": d.get("score", 0),
            "comments": d.get("num_comments", 0),
            "created_utc": d.get("created_utc"),
            "sentiment": sent,
            "sentiment_score": float(sentiment_score)  # Ensure it's a float
        })

    ensure_folder(output_folder)
    output_file = os.path.join(
        output_folder,
        timestamped_filename("reddit_processed")
    )

    save_json(processed, output_file)
    logger.info(f"Processed file saved: {output_file}")
    
    # Log sentiment statistics
    sentiment_scores = [p["sentiment_score"] for p in processed]
    if sentiment_scores:
        avg_score = sum(sentiment_scores) / len(sentiment_scores)
        logger.info(f"Sentiment analysis: {len(processed)} posts, average score: {avg_score:.3f}")

    return output_file


def process_reddit_data_batch_optimized(input_path: str, output_folder: str, batch_size=32):
    """
    Optimized version with better batch processing for large datasets.
    """
    logger.info(f"Processing Reddit data (optimized) from: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []
    
    # Process in batches for better memory efficiency
    for batch_start in range(0, len(data), batch_size):
        batch_end = min(batch_start + batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        batch_titles = [d.get("title", "") for d in batch_data]
        sentiment_batch = sentiment_model.predict_batch(batch_titles)
        
        for j, d in enumerate(batch_data):
            sent = sentiment_batch[j]
            
            # Calculate weighted sentiment score
            sentiment_score = (
                sent["positive"] - sent["negative"] - 0.5 * sent["neutral"]
            )
            
            processed.append({
                "title": d.get("title", "").lower(),
                "score": d.get("score", 0),
                "comments": d.get("num_comments", 0),
                "created_utc": d.get("created_utc"),
                "sentiment": sent,
                "sentiment_score": float(sentiment_score)
            })
        
        logger.info(f"Processed batch {batch_start//batch_size + 1}/{(len(data)-1)//batch_size + 1}")

    ensure_folder(output_folder)
    output_file = os.path.join(
        output_folder,
        timestamped_filename("reddit_processed_optimized")
    )

    save_json(processed, output_file)
    
    # Analyze sentiment distribution
    sentiment_scores = [p["sentiment_score"] for p in processed]
    if sentiment_scores:
        avg_score = sum(sentiment_scores) / len(sentiment_scores)
        positive_count = sum(1 for s in sentiment_scores if s > 0)
        negative_count = sum(1 for s in sentiment_scores if s < 0)
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        logger.info(f"Sentiment distribution:")
        logger.info(f"  Total posts: {len(processed)}")
        logger.info(f"  Average score: {avg_score:.3f}")
        logger.info(f"  Positive: {positive_count} ({positive_count/len(processed)*100:.1f}%)")
        logger.info(f"  Negative: {negative_count} ({negative_count/len(processed)*100:.1f}%)")
        logger.info(f"  Neutral: {neutral_count} ({neutral_count/len(processed)*100:.1f}%)")

    return output_file


if __name__ == "__main__":
    # Test the processor
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        if os.path.exists(test_file):
            print(f"Testing processor with file: {test_file}")
            result = process_reddit_data(test_file, "test_output")
            print(f"Processed file saved to: {result}")
            
            # Load and show sample
            with open(result, 'r') as f:
                sample_data = json.load(f)
                print(f"\nSample processed entry:")
                print(json.dumps(sample_data[0] if sample_data else {}, indent=2))
        else:
            print(f"File not found: {test_file}")
    else:
        print("Usage: python processor.py <input_json_file>")