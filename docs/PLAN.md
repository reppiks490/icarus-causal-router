# Implementation Contract

## Purpose and boundaries

Add a genuinely working delayed-feedback adaptive candidate router, independently
of the active loop builds. Decisions are research allocations, never orders.
Keep the larger user goal active until sibling integration, representative corpus
validation and independent review have evidence. Passing synthetic tests is not
market validation. No throughput/return/accuracy guarantees.

## Work packages

1. Multi-timescale discounted contextual ridge predictors and fixed-share
   prediction-loss aggregation. Mainstream equations with primary references;
   frozen pre-outcome predictions, finite bounded input, deterministic behavior.
2. Immutable event contracts: reviewed candidate/version/source identity; slot
   (asset/timeframe); feature schema; availability before decision; delayed full
   shadow feedback; no final-holdout ingestion or invented counterfactuals.
3. Risk-aware research routing with abstention, cold-start diagnostics, candidate
   changes, switch friction, per-slot isolation, and outcome failure review.
4. SQLite append-only hash chain with exact-once events, crash recovery by replay,
   corruption detection and concurrent-writer conflict detection.
5. CLI JSONL replay, synthetic regime benchmark and reports, CI, independent audit,
   private GitHub publication, concise baton checkpoints.
6. Follow-on: real sibling artifact adapters with honest missing-field diagnostics;
   representative owner-data causal validation; algorithm ablations and scaling.

## Exchange contract v1

Times are nonnegative integer UTC epoch milliseconds, strictly increasing for
decision events within a slot. Event observed times cannot go backwards. All
features are pre-scaled by the producer using training-only statistics, finite,
bounded [-1,1], and have a declared immutable schema. Unknown fields rejected.
An evidence manifest supplies reviewed candidate SHA-256 identities and source
SHA-256 identities. Trust is operator-configured, not cryptographically attested
by this module. Candidates may be revoked by event and must not outlive expiry.

Every decision records all admissible candidate predictions before any label.
Every outcome batch must include exactly those candidates, in net risk units
after fees/slippage. A candidate reward is a shadow simulation estimate, not a
real-world counterfactual. Partial logged-policy feedback is unsupported and is
rejected rather than filled with zeros. Final evaluation data is never learned.
Outcome event_time >= decision time, and observed_at >= event_time. Revisions
require a new run; silently rewriting past learning is forbidden.

## Review focus

Future injection invariance, same-timestamp ordering, label maturity, immutable
version lineage, unauthorized/protected inputs, exact-once feedback, stale source
and feature rejection, survivor bias, missing rewards, adaptive variance claims,
checkpoint drift, corruption, concurrency, numerical overflow, schema mismatch.

## Rulings

- Separate new repository: user forbids modifying active loops.
- Shadow full-information only: selected-only historical logs cannot establish
  the reward of unselected strategies without additional assumptions.
- Uncertainty is a research heuristic, not a coverage/profitability guarantee.
- Existing authorizations cover reversible build and private publication; no
  additional design approval round is required.
