# utils/metrics.py
import pandas as pd
import numpy as np

def compute_ctip(utip_df):
    # CTIP per month: community topic distribution
    k = len([c for c in utip_df.columns if c.startswith("topic_")])
    rows = []
    for month, g in utip_df.groupby("month"):
        cols = [f"topic_{i}" for i in range(k)]
        # take mean across users (or weighted by number of tweets if you track that)
        meanv = g[cols].mean().values
        if meanv.sum() != 0:
            meanv = meanv / meanv.sum()
        rows.append({"month": month, **{cols[i]: float(meanv[i]) for i in range(k)}})
    return pd.DataFrame(rows)

def compute_stia(utip_df, ctip_df):
    # merge by month
    df = utip_df.merge(ctip_df, on="month", suffixes=("","_ct"))
    k = len([c for c in df.columns if c.startswith("topic_") and not c.endswith("_ct")])
    scores = []
    for idx, row in df.iterrows():
        diff = 0.0
        for i in range(k):
            user_p = row[f"topic_{i}"]
            comm_p = row[f"topic_{i}_ct"]
            diff += abs(comm_p - user_p)
        stia = 1.0 - diff  # simple version; paper normalizes to [0,1]
        scores.append(stia)
    df['STIA'] = scores
    return df[['user_id','month','STIA']]

def compute_ctsp_and_stsa(utsp_df):
    # For simplicity we compute:
    # CTSP (monthly mean sentiment) and STSA = 1 - |user_sent - community_sent|
    rows = []
    for month, g in utsp_df.groupby("month"):
        comm_sent = g['avg_sentiment'].mean()
        for idx, row in g.iterrows():
            stsa = 1.0 - abs(row['avg_sentiment'] - comm_sent)
            rows.append({"user_id": row['user_id'], "month": month, "avg_sentiment": row['avg_sentiment'], "CTSP": comm_sent, "STSA": stsa})
    return pd.DataFrame(rows)
