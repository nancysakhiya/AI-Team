# utils/sentiment.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .text_preprocessor import clean_text
import pandas as pd
analyzer = SentimentIntensityAnalyzer()

def score_sentiment(text):
    text = clean_text(text)
    s = analyzer.polarity_scores(text)
    # compound is in [-1,1]
    return s['compound']

def compute_utsp(tweets_df, time_col='timestamp', user_col='user_id'):
    df = tweets_df.copy()
    df['sentiment'] = df['text'].map(score_sentiment)
    df['month'] = df[time_col].dt.to_period('M')
    k = None
    # if we had topic labels per tweet (e.g., from LDA), we should average sentiment per topic.
    # For the synthetic pipeline, compute per-user monthly average sentiment to compare to community baseline.
    rows = []
    for (u, m), g in df.groupby([user_col, 'month']):
        rows.append({"user_id": u, "month": m.to_timestamp(), "avg_sentiment": float(g['sentiment'].mean())})
    return pd.DataFrame(rows)
