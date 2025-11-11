# utils/data_gen.py
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

TOPICS = ["sports","movies","politics","food","travel","tech","health","music"]

def random_sentence_for_topic(topic):
    # Very simple synthetic sentence templates per topic
    templates = {
        "sports": ["What a match!", "Incredible goal today", "training session was tough"],
        "movies": ["Loved the cinematography", "That plot twist was insane", "actor performance was superb"],
        "politics": ["policy debate heats up", "election rallies are crowded", "discussing reform"],
        "food": ["best pizza in town", "trying a new recipe", "restaurant had long queues"],
        "travel": ["airport delays", "beach sunsets", "booking my next trip"],
        "tech": ["new gadget launch", "AI models are advancing", "bug fixed in release"],
        "health": ["morning run", "doctor appointment", "mental health matters"],
        "music": ["concert vibes", "album release", "playlist on repeat"]
    }
    return random.choice(templates.get(topic, ["random thought"]))

def gen_synthetic_users(num_users=200, start_date="2010-01-01", months=12, tweets_per_user_mean=50):
    users = []
    tweets = []
    checkins = []
    start = datetime.fromisoformat(start_date)
    for uid in range(num_users):
        user_id = f"user_{uid}"
        # assign baseline topical preference distribution
        # Dirichlet to provide varied users
        topic_pref = np.random.dirichlet([0.8]*len(TOPICS))
        # baseline negativity/positivity
        sentiment_bias = np.random.normal(loc=0.0, scale=0.3)  # -1..+1 rough
        num_tweets = max(5, int(np.random.poisson(tweets_per_user_mean)))
        for i in range(num_tweets):
            # pick a month uniformly
            month_idx = np.random.randint(0, months)
            timestamp = start + timedelta(days=30*month_idx + np.random.randint(0,28))
            # pick topic according to user's distribution
            topic = np.random.choice(TOPICS, p=topic_pref)
            text = random_sentence_for_topic(topic)
            # add a token with topic label for simpler LDA later
            tweets.append({
                "user_id": user_id,
                "timestamp": timestamp,
                "text": text,
                "true_topic": topic,
                "sentiment_bias": sentiment_bias
            })
        # generate check-ins: simulate category counts by month
        # checkin categories (subset): food, gym, travel, shop
        categories = ["food","gym","travel","shop"]
        for m in range(months):
            # in some months user checks in more or less
            for cat in categories:
                # Poisson count with mean proportional to user's topic_pref for mapping:
                base = 1 + 5*topic_pref[TOPICS.index("food")] if cat == "food" else 1 + np.random.poisson(1)
                count = np.random.poisson(1.5)  # sparse
                # introduce occasional bursts (embark/abandon)
                if random.random() < 0.01:
                    count += np.random.randint(10, 20)
                if count > 0:
                    checkins.append({
                        "user_id": user_id,
                        "timestamp": start + timedelta(days=30*m),
                        "category": cat,
                        "count": int(count)
                    })

    tweets_df = pd.DataFrame(tweets)
    checkins_df = pd.DataFrame(checkins)
    return tweets_df, checkins_df

if __name__ == "__main__":
    # Generate and save synthetic datasets to outputs/
    tweets_df, checkins_df = gen_synthetic_users()

    # Ensure timestamps are proper ISO string for CSV consistency
    tweets_df["timestamp"] = pd.to_datetime(tweets_df["timestamp"])
    checkins_df["timestamp"] = pd.to_datetime(checkins_df["timestamp"])

    outdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(outdir, exist_ok=True)

    tweets_path = os.path.join(outdir, "tweets.csv")
    checkins_path = os.path.join(outdir, "checkins.csv")

    tweets_df.to_csv(tweets_path, index=False)
    checkins_df.to_csv(checkins_path, index=False)

    print(f"Saved tweets to {tweets_path}")
    print(f"Saved checkins to {checkins_path}")
