# Validation Strategy (Target: PR-AUC / average_precision_score)

## Primary metric
- `sklearn.metrics.average_precision_score`

## Time-aware split
- Strictly time-based folds.
- Train window must be earlier than validation window.
- No future information leakage in features.

## Leaderboard alignment
- Keep an internal pseudo-public / pseudo-private weekly analysis.
- Prefer models stable across time slices, not best on a single slice.

## Submission policy
- Max 5 submissions/day -> send only top candidates by local CV.
- Track each submission with commit hash and run id.
