import os
from dotenv import load_dotenv
import praw

from src.utils.logger import logger
from src.utils.file_utils import ensure_folder, save_json, timestamped_filename


load_dotenv()


class RedditClient:
    """Wrapper around PRAW for controlled Reddit API access."""

    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("USER_AGENT", "AlternateAlphaBot/0.1")
        )

    def fetch_posts(self, subreddit: str, limit: int = 50):
        """Fetch posts from a subreddit."""
        logger.info(f"Fetching {limit} posts from r/{subreddit} ...")

        posts = []
        try:
            sr = self.reddit.subreddit(subreddit)
            for post in sr.new(limit=limit):
                posts.append({
                    "id": post.id,
                    "title": post.title,
                    "score": post.score,
                    "created_utc": post.created_utc,
                    "url": post.url,
                    "num_comments": post.num_comments,
                })

            logger.info(f"Fetched {len(posts)} posts.")

        except Exception as e:
            logger.error(f"Error fetching from Reddit: {e}")
            return []

        return posts

    def save_posts(self, posts, folder: str, subreddit: str):
        """Save raw subreddit data to timestamped JSON."""
        ensure_folder(folder)
        filename = timestamped_filename(subreddit)
        path = os.path.join(folder, filename)
        save_json(posts, path)
        return path
