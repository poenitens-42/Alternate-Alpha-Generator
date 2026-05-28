# src/twitter/client.py
import tweepy
from datetime import datetime, timedelta
from src.X.config import TWITTER_CONFIG, SEARCH_PARAMS
from src.utils.logger import logger

class TwitterClient:
    def __init__(self):
        self.client = tweepy.Client(
            bearer_token=TWITTER_CONFIG["bearer_token"],
            consumer_key=TWITTER_CONFIG["api_key"],
            consumer_secret=TWITTER_CONFIG["api_secret"],
            access_token=TWITTER_CONFIG["access_token"],
            access_token_secret=TWITTER_CONFIG["access_secret"],
            wait_on_rate_limit=True
        )
    
    def search_tweets(self, query=None, max_results=100, hours_back=24):
        """Search for recent tweets."""
        if query is None:
            query = SEARCH_PARAMS["query"]
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        logger.info(f"Searching Twitter: '{query}'")
        logger.info(f"Time range: {start_time} to {end_time}")
        
        try:
            response = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                start_time=start_time,
                end_time=end_time,
                tweet_fields=["created_at", "public_metrics", "text"]
            )
            
            if response.data:
                tweets = []
                for tweet in response.data:
                    tweets.append({
                        "id": tweet.id,
                        "text": tweet.text,
                        "created_at": tweet.created_at.isoformat(),
                        "likes": tweet.public_metrics["like_count"],
                        "retweets": tweet.public_metrics["retweet_count"],
                        "replies": tweet.public_metrics["reply_count"],
                    })
                logger.info(f"Fetched {len(tweets)} tweets")
                return tweets
            else:
                logger.warning("No tweets found")
                return []
                
        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            return []