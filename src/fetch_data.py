import praw
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="reddit-alpha-scraper"
)

def fetch_posts(subreddit_name, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=limit):
        posts.append({
            "title": post.title,
            "score": post.score,
            "comments": post.num_comments,
            "created_utc": post.created_utc,
            "text": post.selftext
        })
    return posts

if __name__ == "__main__":
    posts = fetch_posts("wallstreetbets", limit=10)
    for p in posts:
        print(p["title"])
