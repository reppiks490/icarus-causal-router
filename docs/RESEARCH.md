# Algorithm Evidence and Limits

Primary sources consulted, 2026-09-23:

- Joulani, Gyorgy, Szepesvari (2013), [Online Learning under Delayed Feedback](https://proceedings.mlr.press/v28/joulani13.html).
  Receipt is a separate event from decision; frozen predictions are scored only
  after maturation. This implementation does not claim the paper's regret bound.
- Herbster, Warmuth (1998), [Tracking the Best Expert](https://mwarmuth.bitbucket.io/pubs/J39.pdf).
  Exponential loss weights with share prevent an old winner from permanently
  suppressing other horizons. Our share-to-uniform variant includes self-mass;
  it is not the paper's exact off-diagonal parameterization.
- Russac, Vernade, Cappe (2019), [Weighted Linear Bandits for Non-Stationary Environments](https://proceedings.neurips.cc/paper_files/paper/2019/file/263fc48aae39f219b4c71d9d4bb4aed2-Paper.pdf).
  Discounted ridge is a transparent baseline for drift. Our candidate prediction
  ensemble is not D-LinUCB and has no asserted bandit confidence/regret guarantee.
- Dawid (1984), [The Prequential Approach](https://academic.oup.com/jrsssa/article/147/2/278/7106293).
  Forecast first, score later. Post-fit forecasts cannot validate themselves.
- Swaminathan, Joachims (2015), [Counterfactual Risk Minimization](https://proceedings.mlr.press/v37/swaminathan15.html).
  Selected-only logs do not reveal all candidate outcomes. This build rejects
  partial reward vectors and does not implement propensity-based evaluation.

## Implemented equations

For each candidate/version and horizon h, decay gamma = 2^(-1/h):
G_t = sum gamma^(t-s) z_s z_s', b_t = sum gamma^(t-s) z_s r_s,
including only received labels; z=[1,features]. Solve (G_t+lambda I) beta=b_t
by Cholesky. Predictions are clipped to [-1,1]; labels outside that range are
rejected, not silently clipped. The producer must supply common bounded utility
units, not compare dollar outcomes with incompatible sizing.

Frozen member errors l=(prediction-reward)^2/4 update exponential weights,
followed by a fixed share to the uniform horizon mixture. Outcome arrival order
can affect those adaptive weights; it is part of the recorded event stream.
Forgetting is indexed by slot decision count, not number of label arrivals.
Late old labels receive age-discounted sufficient-statistic weight.

Research score = ensemble mean - uncertainty penalty - switch penalty.
No candidate is selected until minimum feedback and minimum score pass. A
research drawdown cap stops further selections in that slot for the run.
Abstention does not stop shadow labels. Failures produce review records, never
automatic code rewrites or relaxed admission criteria.

## Nonclaims

Shadow outcomes are simulator estimates, not observed counterfactual trading.
Input manifests are operator review records, not proof of authentic data.
Uncertainty is heuristic, not calibrated coverage. Synthetic benchmarks prove
software properties, not market edge. Aggregate reward is not portfolio P&L:
overlapping exposure, capital allocation and market impact are unmodeled.
No global final-holdout data may enter online adaptation.
