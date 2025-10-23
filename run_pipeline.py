# run_pipeline.py
import pandas as pd
import numpy as np
import os

from utils.data_gen import gen_synthetic_users
from utils.text_preprocessor import clean_text, tokenize      # ✅ renamed to match earlier utils/preprocessing.py
from utils.topics import build_corpus, compute_utip
from utils.sentiment_model import compute_utsp
from utils.metrics import compute_ctip, compute_stia, compute_ctsp_and_stsa
from utils.iv_analysis import prepare_panel, run_2sls, first_stage_f_test

def main():
    print("Generating synthetic data...")
    tweets_df, checkins_df = gen_synthetic_users(num_users=300, tweets_per_user_mean=60, months=12)
    tweets_df['timestamp'] = pd.to_datetime(tweets_df['timestamp'])
    checkins_df['timestamp'] = pd.to_datetime(checkins_df['timestamp'])

    print("Building LDA topics...")
    lda, dictionary, corpus, docs = build_corpus(tweets_df['text'], num_topics=8, passes=15)

    print("Computing UTIP...")
    utip_df = compute_utip(tweets_df, lda, dictionary, docs)

    print("Computing CTIP...")
    ctip_df = compute_ctip(utip_df)

    print("Computing STIA...")
    stia_df = compute_stia(utip_df, ctip_df)

    print("Computing UTSP (sentiment)...")
    utsp_df = compute_utsp(tweets_df)

    print("Computing STSA...")
    stsa_df = compute_ctsp_and_stsa(utsp_df)

    print("Preparing panel and instrument data...")
    panel = prepare_panel(stia_df, stsa_df, checkins_df)

    # === Synthetic influencer mapping (demo only) ===
    print("Creating synthetic influencer mapping for demo...")
    user_list = panel['user_id'].unique()
    rng = np.random.RandomState(42)
    influencer_map = {u: rng.choice(user_list) for u in user_list}
    panel['influencer'] = panel['user_id'].map(influencer_map)

    infl = panel[['user_id', 'month', 'STIA']].rename(
        columns={'user_id': 'influencer', 'STIA': 'influencer_STIA'}
    )
    panel = panel.merge(infl, on=['influencer', 'month'], how='left')
    panel['influencer_STIA'] = panel['influencer_STIA'].fillna(panel['STIA'].mean())

    # === IV analysis ===
    import utils.iv_analysis as ivmod
    controls = ['avg_sentiment']

    print("\n=== First-stage OLS regression ===")
    res, tstat, pval, fstat = ivmod.first_stage_f_test(
        panel, treatment='influencer_STIA', instrument='food', controls=controls
    )
    print(res.summary())
    print(f"Instrument (food) t-stat: {tstat:.3f}, p-value: {pval:.3g}, approx F-stat: {fstat}")

    print("\n=== Running IV (2SLS) regression ===")
    iv_res = ivmod.run_2sls(
        panel,
        dependent='STIA',
        treatment='influencer_STIA',
        instrument='food',
        controls=controls
    )
    print(iv_res.summary)

    # === Save outputs ===
    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)
    panel.to_csv(os.path.join(outdir, "panel.csv"), index=False)
    print(f"\nPanel saved to {outdir}/panel.csv")
    print("✅ Done successfully.")

if __name__ == "__main__":
    main()
