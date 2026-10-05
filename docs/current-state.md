# Current State

## 2026-10-05 CI / workflow evolution

Branch `chatgpt/workflow-evolution-20261005` adds a first-class GitHub Actions assurance layer while
preserving the router's research-only boundary. The workflow runs on relevant
pushes and pull requests, once daily for environment-drift detection, and on
manual dispatch.

The gate covers Python 3.10-3.13 compilation/unit tests, two independent seeded
synthetic replays with digest equality, replay verification, explicit assertions
that execution remains disabled and synthetic results cannot become market or
profitability evidence, plus wheel/sdist build and installed CLI smoke tests.

No market feed, broker, credential, live-order path, final holdout, or sibling
write authority is introduced.


New isolated build; active sibling source untouched. Implementing PLAN packages
1-5, then independent review. No market-validation or completion claim.
Previous turn clarified evidence-lab scope but did not implement this goal.

Agents: Descartes inspects sibling contracts read-only; Galileo researches
algorithm assumptions. Main owns causal runtime, persistence and integration.

TickerLayer/FactorWeave: no callable tools discovered in this environment.
HeyTraders skill is present; authenticated market access is not yet verified.
