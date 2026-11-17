# Novara — ball possession for More than a Hack 2027, Phase 1

Team **Novara**'s solution to the More than a Hack 2027 ball-possession challenge (FC Barcelona vs
Athletic Club). For every frame of the tactical video it writes who has the ball: `H` (home), `A`
(away), `D` (dead ball) or `X` (not graded).

The method is rule-based and has no trained parameters. Evidence for each state is computed per
frame from automatic tracking and Smart Tagging, then a three-state Viterbi decoder picks the
sequence of possessions. Every change of state has a fixed cost, so a touch or a deflection does
not hand possession over. The broadcast tracking is aligned to the tactical timeline from the ball
tracks alone and used to fill tactical ball gaps.

The full description, including what was tried and dropped and how the public ground truth was
used, is in **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**.

## Run it

Requirements: Python 3.11.9, numpy 2.4.6 and pandas 3.1.1 (pinned in `requirements-solution.txt`).

```bash
pip install -r requirements-solution.txt

# Unpack the match bundle (gated: see DATA.md) into ./data, keeping its names:
#   data/tactical/BAR-ATH_tactical_atd.{xml,txt}, BAR-ATH_tactical_smart_tagging.xml, ...
#   data/broadcast/BAR-ATH_broadcast_atd.{xml,txt}, BAR-ATH_broadcast_homography.csv
# or point MTAH_DATA_DIR at wherever it is.

python -m solution.run --team Novara                         # writes evaluation/results/Novara.csv
python evaluation/validate_submission.py evaluation/results/Novara.csv
python evaluation/run_evaluation.py                          # optional: the public 5-minute check
```

It takes about 2.5 minutes. The output is deterministic. In the pinned environment it is
byte-identical to the submitted file, SHA-256
`e49aec949eb599f9f17a6072f2e10cd98d86a681e8d64452f31e37881cd9eaf2`.

## Layout

```
solution/      the method: atd.py (tracking reader), features.py, tagging.py, broadcast.py,
               model.py (evidence + Viterbi), config.py (every parameter, with its reason), run.py
training/      experiment on other games (Metrica public sample data), not used for the submission
evaluation/    the organisers' official scorer and validator
starter/       the organisers' starter code
data/          the match data goes here; nothing in it is committed
```

The challenge itself is described in [README_ORGANISERS.md](README_ORGANISERS.md),
[CHALLENGE.md](CHALLENGE.md) and [RULES.md](RULES.md). Which files are the organisers' and which
are ours: [NOTICE.md](NOTICE.md).

No match data is in this repository; the data's terms of use forbid redistributing it.
