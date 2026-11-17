# Ball possession from automatic tracking — technical document

*More than a Hack 2027, Phase 1. Team: `Novara`*

## 1. Approach

We treat possession as what it is in the ground truth definition: a sequence of intervals, not a
set of independent frames. The pipeline has two stages.

1. **Evidence.** For every frame we compute a score for each of the three states `H`, `A`, `D`
   from the tracking and the Smart Tagging. Scores are additive and expressed in one unit — "one
   frame of clear, uncontested ball control" — so that every source of evidence can be weighed
   against the others by reasoning rather than fitting.
2. **Decoding.** A three-state Viterbi decoder picks, per half, the state sequence with the
   highest total score minus a fixed cost for every change of state (`P_SWITCH = 12` for a change
   of team, `P_DEAD = 10` for entering or leaving a dead ball).

The switching cost is the core of the method. The challenge defines a change of possession as the
opponent gaining *controlled* possession, and excludes touches without control, deflections and
lost aerial duels. With the cost above, the opponent must show roughly half a second more control
than the current team before the decoder will hand possession over. A touch of a few frames is
absorbed. While the ball travels between players (a pass) nobody is close to it, there is no
evidence either way, and possession stays with the last team, as CHALLENGE.md requires for loose
and contested balls.

We chose this over the reference baseline (nearest player plus a majority-vote window) because a
fixed window has to trade fragmentation against lateness. A wide window removes flicker but shifts
every transition, and spellF1 punishes a one-second shift heavily. The decoder places each
transition at the frame where the evidence changes, and still rejects short-lived flicker.

We chose it over a learned classifier because labels from the provided game may not be used for
training or validation, and we had no labelled data from other games.

## 2. Evidence used

| source | feature | effect |
|---|---|---|
| ATD ball + players | distance from the ball to the nearest player of each team, soft-thresholded at 1.8 m | `+` for the team in control, `−` for the other; zero when both are equally close (contested) |
| ATD ball | ball speed; above 10–20 m/s proximity evidence fades out | a fast ball passing a player is not control |
| ATD ball | ball outside the pitch by > 1 m | strong `D` |
| ATD ball | ball nearly still, nobody on it | weak `D` (placed ball before a restart) |
| ATD ball | ball not seen at all | very weak `D` |
| ATD ball + players | ball moving under a player's control | weak `not D` |
| ATD players | collective movement: mean player speed, smoothed over ±2 s | `D` when below the match's own live/dead threshold |
| ATD players | team shape: mean distance of each team's players to its median position (farthest player, usually the goalkeeper, dropped); home minus away, smoothed over ±1 s | weak team evidence (0.1 per frame at full strength), **only where the ball track gives none** — the team on the ball stretches, the other compacts |
| Smart Tagging, phases | team tag of attacking phases (possessor) and defending/defensive-transition phases (non-possessor) | weak team prior, 0.03 per frame |
| Smart Tagging, set pieces | restart detected inside the tag as the ball going from slow to fast | `D` before the restart, `not D` and a small bonus for the tagged team after |
| Smart Tagging, goals | from the goal until the following "Kick Off (After Goal)" | `D` |
| Smart Tagging, kick-offs | from the start of a half's graded window to the restart detected in that half's "Kick Off (Start)" tag | strong `D` — the second half's graded window opens before its kick-off (FAQ.md); found from the tag, no frame hard-coded |

**Tracking clean-up.** Ball gaps of up to 8 frames (0.32 s) are filled by linear interpolation.
Gaps up to 38 frames (1.5 s) are also interpolated, but only for the control evidence and with a
confidence that falls linearly from 1 to 0 with the gap length. Longer gaps are left as missing:
the decoder carries the current state through them, with only the team-shape evidence to go on.

**Team shape, checked without labels.** On graded frames where one team clearly controls a
visible, slow ball (a player within 1.5 m, nobody from the other team within 4 m), the sign of the
shape signal agrees with the team in control on **80.8%** of ~24,000 frames (50% would be no
information). Its zero point is set midway between its mean under clear home control and under
clear away control — again from the tracking's own control evidence, not from a label. A first
version used Otsu's split as the zero point and applied the signal on every frame; it was biased
toward the home team (the two conditional means are +0.64 and −0.42 standard deviations) and,
because shape lags a turnover by seconds, delayed switches the ball track already placed. It was
replaced by the version above for those two reasons.

**Tie-breaking in ball gaps.** Where the ball is unseen, a change of team can often sit anywhere
in the gap at almost the same total score, and the decoder's argmax breaks such ties by state
index — late for H→A, early for A→H, a systematic asymmetry. After decoding, every change of team
touching an unseen stretch is moved to the position closest to the middle of that stretch among
those scoring within 0.5 units of the best. Real evidence in the gap still decides; only where it
is flat does the middle win, which minimises the expected error when the moment is unknown.

**Self-calibration.** "Low movement" depends on the recording, so the threshold is not a
constant. It is found by Otsu's method on the distribution of the smoothed collective speed over
the graded frames of this match. This is unsupervised and uses no labels.

**Smart Tagging team names** ("Barcelona", "Atletic Club") are matched to the ATD team names by
string similarity, not hard-coded. Tags with team `N/A`, including every throw-in, contribute
only their dead/alive timing.

## 3. Assets used and dropped

- **Used:** tactical ATD (players and ball), tactical Smart Tagging.
- **Used:** broadcast ATD ball track and the broadcast homography's `valid` column, to fill
  tactical ball gaps (`solution/broadcast.py`). The broadcast is not synchronised to the tactical
  timeline, but both ATD files share pitch coordinates, so we align them from the tracking alone:
  - The broadcast is cut into 10 s windows. Each window with enough visible ball is matched to the
    tactical offset at which the two ball tracks agree (coarse search every 5 frames over the
    whole match, refined to one frame). A window matches when the median ball distance is
    ≤ 1 m: 207 of 657 windows, median distance **0.53 m**, at only 7 distinct offsets that
    increase monotonically — the broadcast's cuts.
  - The offset is piecewise constant, so an unmatched window between two matched windows with
    the same offset inherits it — **unless its own data contradicts it**: if enough ball is seen
    in both recordings and the median distance at that offset is above 1 m, it is rejected. A
    first version without that check was tested label-free on the inherited windows that could
    be checked, gave a median distance of 2.07 m, and was replaced. With it, 222 windows are
    inherited and 143 rejected.
  - Where the tactical ball is missing and the mapped broadcast frame has a ball and a valid
    homography (no replay, close-up or crowd shot), the broadcast position is used: 5,700 frames.
    The ball is visible on **62.9%** of graded frames instead of 59.3%. All downstream evidence
    treats these frames as seen.
- **Dropped:** broadcast video and broadcast player tracks (the tactical player tracks already
  cover the whole pitch), tactical homography and tactical video. Our method works entirely on
  pitch coordinates, which the ATD already provides. Re-detecting the ball from video is the
  obvious next step (see §6) but was out of scope.

## 4. Training data and use of ground truth

**No labelled data of any kind was used.** The method has no trained parameters. All weights in
`solution/config.py` are set from physical and football reasoning (control radius, pass speed,
seconds of evidence), and the one data-dependent threshold is estimated without labels (§2).

The public five-minute ground truth was used only to check that the pipeline is correct: frame
alignment, file format and label meaning. No parameter was chosen by comparing scores on it, and
no variant was kept or dropped for its score. Each change in §2 was decided on football or
technical grounds; the public score was read once per version as a regression check, and the one
time it dropped sharply (first team-shape version) the cause was diagnosed with label-free
statistics, not by trying values. For
testing, `solution/synthetic.py` generates a fake match in the real file formats, with its own
ground truth. That checks the mechanics end to end. Its score says nothing about the real match.

No match data was sent to any third-party or cloud-hosted LLM.

## 5. Possession rules implemented

- **Change of possession** needs sustained, uncontested proximity of the new team to a ball that
  is slow enough to be controlled. Roughly 0.5 s of net control is enough. Touches, deflections
  and contested balls are not.
- **Loose and in-flight balls** produce no evidence, so possession stays with the last team.
- **Contested balls** (players of both teams equally close) produce no evidence either.
- **Restarts:** dead until the ball is struck. The detected restart moment inside a Smart Tagging
  set-piece window marks that point. After it, the tagged team (when known) gets a small prior
  for 2 s, and the proximity evidence decides after that.
- **Dead ball** comes from a combination of: ball out of the pitch, collective slowdown,
  set-piece and goal windows, and a still ball with nobody on it.
- Frames outside the two graded halves are written as `X`. Inside the halves we always commit to
  `H`, `A` or `D`. An `X` there would cost macroF1 and gain nothing on spellF1.

## 6. Limitations and what we would do next

- **Ball invisible on ~41% of graded frames** (~37% after filling from the broadcast — the
  broadcast tends to lose the ball in the same situations, and cuts to replays at exactly the
  moments that matter). Through long gaps the decoder can only carry the last state forward and
  the team-shape evidence, which is weak and lags. A turnover that happens entirely while the ball
  is unseen is still often detected late. Next step: re-detect the ball in the tactical video with
  a small detector and project it with the homography.
- **Broadcast alignment is per 10 s window.** A cut inside a window is not located precisely, and
  the two ball tracks differ by ~0.5 m, which slightly perturbs ball speed where the two sources
  meet.
- **Aerial balls.** The ATD ball position is a ground projection, so a high ball appears near
  players it is far above. Ball speed gating removes part of this, not all of it.
- **Dead-ball timing** relies on collective slowdown, which lags the whistle by a second or more,
  and can misfire during very slow build-up. Ball-out detection is only as good as the ball track
  near the lines.
- **Weights are reasoned, not fitted.** With labelled tracking from other matches, the same
  decoder could have its emission weights learned (a hidden semi-Markov model with learned
  duration distributions would be the natural upgrade).
- **Track identity switches** do not affect the method, which only uses team membership and the
  nearest player per team. A slot assigned to the wrong team would.

## 7. Reproduction

```bash
pip install -r requirements-solution.txt
# unpack the match bundle into ./data (or set MTAH_DATA_DIR)
python -m solution.run --team Novara
python evaluation/validate_submission.py evaluation/results/Novara.csv
```

Runtime is about 2.5 minutes on a laptop, most of it the broadcast alignment (set
`USE_BROADCAST = False` in `solution/config.py` to skip it; without the broadcast files in
`data/broadcast/` it is skipped automatically, and the result then differs). The method is fully deterministic: no randomness, no
training. The same data produces the same file byte for byte.

Optional mechanics check without the data:

```bash
python -m solution.synthetic /tmp/synth
python -m solution.run --team synth --data-dir /tmp/synth --out-dir /tmp/synth/results
python evaluation/run_evaluation.py --gt /tmp/synth/synthetic_gt.csv --results /tmp/synth/results
```

Code layout: `solution/atd.py` (tracking reader), `solution/features.py` (per-frame features and
self-calibration), `solution/tagging.py` (Smart Tagging), `solution/broadcast.py` (broadcast
alignment and ball fill), `solution/model.py` (evidence and
Viterbi decoder), `solution/config.py` (every parameter, with its rationale), `solution/run.py`
(entry point).
