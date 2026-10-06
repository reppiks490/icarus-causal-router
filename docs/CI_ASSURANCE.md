# CI assurance contract

This repository uses `.github/workflows/assurance.yml` to enforce the
research-only causal-router contract.

## Change-time gates

Pushes to `main`, pull requests, merge-queue checks, and manual dispatches run:

- Python 3.10 through 3.13 compilation and the complete deterministic unit suite.
- Dependency consistency with `pip check`.
- Two same-seed synthetic runs whose result digests must match.
- Replay verification.
- Explicit assertions that authority remains research-only, execution remains
  disabled, and synthetic output is not market validation or a profitability
  claim.
- Wheel/sdist build and installed CLI smoke test.
- A final aggregate gate.

## Scheduled drift gate

The daily schedule uses one Python 3.12 job to re-run compilation, tests,
dependency checks, a seeded synthetic replay, replay verification, and the
research-authority firewall.

## Machine-readable evidence

Every aggregate gate writes `assurance/assurance-summary.json` and uploads it
as a 30-day workflow artifact for durable owner automation or UI ingestion.

The evidence hard-codes:

- `authority=research_only`
- `execution_allowed=false`
- `market_validation=false`
- `profitability_claim=false`

## Current infrastructure caveat

As of 2026-10-05, GitHub accepted and scheduled this private-repository workflow
but GitHub-hosted jobs failed before usable step logs were produced. A temporary
bare runner probe failed the same way, while the public AION workflow executed
normally. The workflow remains installed and ready; private-repository runner
eligibility/quota/settings must allow a hosted runner before these gates can
execute.

## Dependency maintenance automation

Dependabot checks GitHub Actions and Python packaging metadata every Monday in
`America/Chicago`. Minor and patch updates are grouped to reduce pull-request
noise; major updates remain isolated for explicit review. Dependency changes that
touch `pyproject.toml` or workflow files are still subject to the repository's
normal assurance gates before merge.

## Runner portability

All assurance jobs default to GitHub-hosted `ubuntu-latest`. The repository
variable `ASSURANCE_RUNNER` can override that label without editing the
workflow. This provides a controlled path to a Linux self-hosted runner if
private hosted-runner quota or billing prevents execution.

Leave `ASSURANCE_RUNNER` unset unless a maintained Linux runner is actually
registered and isolated for this repository. A self-hosted runner executes
repository code on the machine that hosts it, so it should not be exposed to
untrusted pull requests or secrets beyond the minimum required.

