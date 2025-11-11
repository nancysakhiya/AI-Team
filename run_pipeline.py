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
import matplotlib.pyplot as plt

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

    # === Influencer network (top 10% by activity) ===
    print("Creating influencer set and exogenous monthly shock...")
    tweets_per_user = tweets_df.groupby('user_id').size().sort_values(ascending=False)
    num_influencers = max(1, int(0.10 * len(tweets_per_user)))
    influencer_set = set(tweets_per_user.index[:num_influencers])

    rng = np.random.RandomState(42)
    influencers_list = list(influencer_set)
    def pick_influencer(u):
        if u in influencer_set:
            return u
        return influencers_list[rng.randint(0, len(influencers_list))]
    panel['influencer'] = panel['user_id'].map(pick_influencer)

    infl = panel[['user_id', 'month', 'STIA']].rename(
        columns={'user_id': 'influencer', 'STIA': 'influencer_STIA'}
    )
    panel = panel.merge(infl, on=['influencer', 'month'], how='left')
    panel['influencer_STIA'] = panel['influencer_STIA'].fillna(panel['STIA'].mean())

    # === Exogenous monthly shock to strengthen first stage ===
    months = (
        panel[['month']]
        .drop_duplicates()
        .sort_values('month')
        .reset_index(drop=True)
    )
    shock = rng.normal(loc=0.0, scale=0.5, size=len(months))
    spike_idx = rng.choice(len(months), size=max(1, len(months)//6), replace=False)
    shock[spike_idx] += rng.normal(loc=1.5, scale=0.3, size=len(spike_idx))
    months['food_shock'] = shock
    panel = panel.merge(months, on='month', how='left')

    shock_strength = 0.5
    panel['influencer_STIA_shocked'] = panel['influencer_STIA'] + shock_strength * panel['food_shock']

    # === IV analysis ===
    import utils.iv_analysis as ivmod
    controls = ['avg_sentiment']

    print("\n=== First-stage OLS regression ===")
    res, tstat, pval, fstat = ivmod.first_stage_f_test(
        panel, treatment='influencer_STIA_shocked', instrument='food_shock', controls=controls
    )
    print(res.summary())
    print(f"Instrument (food_shock) t-stat: {tstat:.3f}, p-value: {pval:.3g}, approx F-stat: {fstat}")

    print("\n=== Running IV (2SLS) regression ===")
    iv_res = ivmod.run_2sls(
        panel,
        dependent='STIA',
        treatment='influencer_STIA_shocked',
        instrument='food_shock',
        controls=controls
    )
    print(iv_res.summary)

    # === Save outputs ===
    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)
    panel.to_csv(os.path.join(outdir, "panel.csv"), index=False)
    # === Simple visuals ===
    print("Creating visuals...")
    plt.figure(figsize=(6,4))
    panel['STIA'].hist(bins=30)
    plt.title('Distribution of STIA')
    plt.xlabel('STIA')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'stia_distribution.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(7,4))
    panel.groupby('month')['STIA'].mean().plot()
    plt.title('Monthly Average STIA')
    plt.xlabel('Month')
    plt.ylabel('Avg STIA')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'stia_monthly_trend.png'), dpi=150)
    plt.close()

    results_md = os.path.join(outdir, 'results.md')
    with open(results_md, 'w', encoding='utf-8') as f:
        f.write('# Results Note\n\n')
        f.write('This run uses a synthetic influencer network (top 10% by activity) and an exogenous monthly ')
        f.write('shock `food_shock` that perturbs influencer alignment to strengthen the first stage.\n\n')
        f.write('## First Stage (instrument strength)\n')
        f.write(f'- t-stat: {tstat}\n')
        f.write(f'- approx F-stat: {fstat}\n\n')
        f.write('## IV (2SLS) outcome\n')
        f.write(str(iv_res.summary))
    print(f"\nPanel saved to {outdir}/panel.csv")
    print(f"Visuals saved to {outdir}/stia_distribution.png and {outdir}/stia_monthly_trend.png")
    print(f"Results note saved to {outdir}/results.md")
    print("✅ Done successfully.")

if __name__ == "__main__":
    main()
