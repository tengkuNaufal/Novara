"""Every tunable number of the pipeline, in one place.

None of these values was fitted to any ground truth -- not the private one and not the public five
minutes. They come from football and tracking common sense (how close a player has to be to
control a ball, how fast a pass travels) or are expressed in units that make them easy to reason
about (frames of evidence). Thresholds that depend on this particular recording, such as what
"low collective movement" means, are not set here at all: they are estimated from the match's own
tracking without labels, in features.py.
"""

FPS = 25

# -- tracking clean-up -------------------------------------------------------------------------
# Ball gaps up to this many frames are filled by linear interpolation. 0.32 s: short enough that
# the ball cannot have changed hands unseen in any meaningful way.
BALL_MAX_GAP = 8
# Gaps longer than that, up to BALL_MAX_GAP_LONG frames (1.5 s), are also interpolated, but only for
# the control evidence and with a confidence that falls linearly from 1 to 0 with the gap length:
# the longer the ball was unseen, the less a straight line between its two sightings says about
# where it actually was.
BALL_MAX_GAP_LONG = 38
# A ball counts as "out of the pitch" when it is this many metres beyond a line.
OUT_MARGIN_M = 1.0

# -- control evidence --------------------------------------------------------------------------
# A player "controls" the ball when within roughly CONTROL_RADIUS_M metres of it. The sigmoid width
# softens the edge because tracking positions of player and ball both carry ~0.5 m of noise.
CONTROL_RADIUS_M = 1.8
CONTROL_SOFT_M = 0.4
# A ball moving faster than this is in flight (pass, shot, clearance): a player near its path is
# not controlling it. Between the two values the evidence fades linearly.
BALL_FAST_LO = 10.0   # m/s
BALL_FAST_HI = 20.0   # m/s
# Weight of one frame of clear, uncontested control.
W_CONTROL = 1.0

# -- Smart Tagging prior -------------------------------------------------------------------------
# Per-frame weight of a phase tag. Deliberately weak: a 10 s phase adds ~7.5, less than one
# possession switch costs, so tags break ties while the ball is invisible but cannot overrule it.
W_TAG = 0.03
# Bonus for the restarting team in the first seconds after a tagged set piece restart.
W_RESTART_TEAM = 0.15
RESTART_TEAM_SECONDS = 2.0

# -- team shape evidence -----------------------------------------------------------------------
# The team in possession stretches out to create space; the team without the ball compacts. Per
# team we measure the mean distance of its players to the team's median position, dropping the
# single farthest player (usually the goalkeeper). The home-minus-away difference, smoothed over
# +/- SHAPE_WINDOW_S and divided by its own spread over the graded frames (no labels), is weak
# team evidence. It works with the ball unseen, which is where it matters. At W_SHAPE = 0.1 it
# takes ~5 s of clearly opposite shape to pay for one change of possession on its own, so it
# cannot overrule visible control (1.0 per frame), only decide what the ball track cannot.
W_SHAPE = 0.1
SHAPE_WINDOW_S = 1.0
SHAPE_MIN_PLAYERS = 8

# -- dead-ball evidence ------------------------------------------------------------------------
# Collective movement: per-frame weight when the smoothed mean player speed is clearly below the
# match's own live/dead threshold (estimated without labels).
W_SLOW = 0.25
ACTIVITY_WINDOW_S = 2.0
# Ball visible and outside the pitch.
W_OUT = 2.0
# Ball visible and (almost) motionless with nobody on it.
W_STILL = 0.2
BALL_STILL_MS = 0.5
# Ball not seen at all. Small: the ball is missing on 41% of graded frames, mostly during play.
W_MISSING = 0.08
# Ball being carried / moving under a player's control: evidence that play is live.
W_LIVE = 0.3
# Inside a tagged set piece, before the detected restart: dead. Just after it: alive.
W_SET_PIECE = 0.8
# After a tagged goal, until the following kick-off.
W_AFTER_GOAL = 1.5
# Restart detection inside a set-piece tag: the ball must go from slow to fast.
RESTART_FAST_MS = 4.0
RESTART_SLOW_MS = 1.5

# -- decoder -----------------------------------------------------------------------------------
# Cost, in units of "frames of clear control", of changing state. A possession changes hands only
# when the new team shows about half a second more control than the old one -- which is what
# "controlled possession" means in CHALLENGE.md: a touch, a deflection or a lost aerial duel is
# not enough.
P_SWITCH = 12.0
# Cost of entering or leaving a dead-ball stretch.
P_DEAD = 10.0
# Where the ball is unseen, a change of team can often sit anywhere in the gap at (almost) the
# same total score, and the decoder's argmax would then break the tie by state index -- late for
# H->A, early for A->H. Any position within this many units of the best is treated as equally
# good, and the one closest to the middle of the gap is taken: the expected-error minimiser when
# the moment of the turnover is unknown.
SWITCH_TIE_TOL = 0.5
