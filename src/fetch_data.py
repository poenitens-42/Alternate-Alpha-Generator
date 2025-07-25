import praw
from dotenv import load_dotenv
import os

# Load .env 
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Create Reddit instance 
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT")
)

def fetch_posts(subreddit_name, limit=100):
    posts = []
    subreddit = reddit.subreddit(subreddit_name)
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
    for p in fetch_posts("wallstreetbets", 10):
       print(p["title"].encode("ascii", errors="ignore").decode())

