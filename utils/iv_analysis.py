# utils/iv_analysis.py  (corrected)
import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

def prepare_panel(stia_df, stsa_df, checkins_df):
    c = checkins_df.copy()
    c['month'] = c['timestamp'].dt.to_period('M').dt.to_timestamp()
    agg = c.groupby(['user_id','month','category'])['count'].sum().unstack(fill_value=0).reset_index()
    df = stia_df.merge(stsa_df[['user_id','month','avg_sentiment','STSA']], on=['user_id','month'])
    df = df.merge(agg, on=['user_id','month'], how='left').fillna(0)
    return df

def run_2sls(df, dependent='STIA', treatment='influencer_STIA', instrument='food', controls=None):
    """
    Run IV 2SLS with:
      dependent ~ exog + [treatment (endogenous) instrumented by instrument]
    - df: DataFrame
    - controls: list of column names to include as exogenous controls (besides constant)
    - instrument: column name (or list) used as instrument(s) for 'treatment'
    """
    if controls is None:
        controls = ['avg_sentiment']

    # Exogenous regressors (constant + controls)
    exog = sm.add_constant(df[controls], has_constant='add')

    # Dependent variable (outcome)
    endog = df[dependent]

    # Endogenous regressor(s) (treatment)
    # linearmodels expects these as a DataFrame or array
    endog_reg = df[[treatment]]

    # Instruments: ONLY the external instruments (do NOT include controls or an added constant here)
    instr = df[[instrument]]

    # Fit IV2SLS
    iv = IV2SLS(endog, exog, endog_reg, instr).fit(cov_type='robust')
    return iv

def first_stage_f_test(df, treatment='influencer_STIA', instrument='food', controls=['avg_sentiment']):
    # regress treatment on instrument + controls and compute F-stat for instrument coef
    X = df[[instrument] + controls]
    X = sm.add_constant(X, has_constant='add')
    y = df[treatment]
    res = sm.OLS(y, X).fit()
    # F-test for null that instrument coeff == 0
    try:
        tstat = res.tvalues[instrument]
        pval = res.pvalues[instrument]
    except Exception:
        tstat, pval = None, None
    # compute overall F-stat for instruments (if multiple instruments)
    # For single instrument we can approximate F = (t^2)
    f_stat = None
    if tstat is not None:
        f_stat = float(tstat**2)
    return res, tstat, pval, f_stat
