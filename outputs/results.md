# Results Note

This run uses a synthetic influencer network (top 10% by activity) and an exogenous monthly shock `food_shock` that perturbs influencer alignment to strengthen the first stage.

## First Stage (instrument strength)
- t-stat: 66.11582645691948
- approx F-stat: 4371.302508081494

## IV (2SLS) outcome
                          IV-2SLS Estimation Summary                          
==============================================================================
Dep. Variable:                   STIA   R-squared:                     -0.0011
Estimator:                    IV-2SLS   Adj. R-squared:                -0.0017
No. Observations:                3578   F-statistic:                    12.259
Date:                Tue, Nov 11 2025   P-value (F-stat)                0.0022
Time:                        14:25:01   Distribution:                  chi2(2)
Cov. Estimator:                robust                                         
                                                                              
                                    Parameter Estimates                                    
===========================================================================================
                         Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
-------------------------------------------------------------------------------------------
const                       0.2212     0.0080     27.508     0.0000      0.2054      0.2370
avg_sentiment               0.1133     0.0374     3.0308     0.0024      0.0400      0.1866
influencer_STIA_shocked    -0.0273     0.0152    -1.7993     0.0720     -0.0571      0.0024
===========================================================================================

Endogenous: influencer_STIA_shocked
Instruments: food_shock
Robust Covariance (Heteroskedastic)
Debiased: False