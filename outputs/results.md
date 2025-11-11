# Results Note

This run uses a synthetic influencer network (top 10% by activity) and an exogenous monthly shock `food_shock` that perturbs influencer alignment to strengthen the first stage.

## First Stage (instrument strength)
- t-stat: 139.94508071006808
- approx F-stat: 19584.62561494747

## IV (2SLS) outcome
                          IV-2SLS Estimation Summary                          
==============================================================================
Dep. Variable:                   STIA   R-squared:                     -0.0002
Estimator:                    IV-2SLS   Adj. R-squared:                -0.0007
No. Observations:                3576   F-statistic:                    0.1215
Date:                Tue, Nov 11 2025   P-value (F-stat)                0.9411
Time:                        13:05:36   Distribution:                  chi2(2)
Cov. Estimator:                robust                                         
                                                                              
                                    Parameter Estimates                                    
===========================================================================================
                         Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
-------------------------------------------------------------------------------------------
const                       0.2115     0.0049     43.052     0.0000      0.2019      0.2211
avg_sentiment               0.0123     0.0378     0.3248     0.7453     -0.0618      0.0864
influencer_STIA_shocked    -0.0009     0.0075    -0.1198     0.9046     -0.0156      0.0138
===========================================================================================

Endogenous: influencer_STIA_shocked
Instruments: food_shock
Robust Covariance (Heteroskedastic)
Debiased: False