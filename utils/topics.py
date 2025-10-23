# utils/topics.py
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import numpy as np
import pandas as pd
from .text_preprocessor import clean_text, tokenize
from tqdm import tqdm
tqdm.pandas()

def build_corpus(tweets_series, num_topics=8, passes=10):
    # tweets_series: pd.Series of text
    docs = tweets_series.fillna("").map(clean_text).map(tokenize).tolist()
    dictionary = Dictionary(docs)
    # filter extremes
    dictionary.filter_extremes(no_below=3, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=42)
    return lda, dictionary, corpus, docs

def infer_topic_distribution(lda, dictionary, docs):
    # docs: list of token lists
    dists = []
    for doc in docs:
        bow = dictionary.doc2bow(doc)
        dist = dict(lda.get_document_topics(bow, minimum_probability=0.0))
        # convert to np array
        arr = np.array([dist[i] for i in range(lda.num_topics)])
        dists.append(arr)
    return np.vstack(dists)

def compute_utip(tweets_df, lda, dictionary, docs, time_col='timestamp', user_col='user_id'):
    # tweets_df must have same order as docs
    tweets_df = tweets_df.copy().reset_index(drop=True)
    topic_dists = infer_topic_distribution(lda, dictionary, docs)
    k = topic_dists.shape[1]
    tweets_df[[f"t{k}_p" for k in range(k)]] = topic_dists
    # group by user-time window (month)
    tweets_df['month'] = tweets_df[time_col].dt.to_period('M')
    # compute fraction per topic per user per month
    rows = []
    for (user, month), group in tweets_df.groupby([user_col, 'month']):
        probs = group[[f"t{i}_p" for i in range(k)]].mean().values  # average topic probabilities across tweets
        norm = probs.sum()
        if norm == 0:
            probs = np.zeros_like(probs)
        else:
            probs = probs / norm
        row = {"user_id": user, "month": month.to_timestamp(), **{f"topic_{i}": float(probs[i]) for i in range(k)}}
        rows.append(row)
    utip_df = pd.DataFrame(rows)
    return utip_df
