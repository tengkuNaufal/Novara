"""Per-frame evidence for the three states, and the decoder that turns it into possessions.

States: 0 = H, 1 = A, 2 = D. Every frame gets a score per state (higher = more plausible); the
decoder picks the state sequence with the highest total score minus a fixed cost per state change.
That cost is what turns flickering frame-by-frame evidence into possessions: a change of hands has
to be paid for by sustained control, exactly the "controlled possession" rule of CHALLENGE.md.
"""

import numpy as np

from . import config as C
from . import tagging

H, A, D = 0, 1, 2


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def find_restart(start, end, v_ball, fps):
    """Frame inside (or just before) a set-piece tag where the ball goes from slow to fast."""
    lo = max(1, start - 2 * fps)
    hi = min(len(v_ball) - 1, end)
    still = int(0.5 * fps)
    for t in range(lo + still, hi):
        if v_ball[t] > C.RESTART_FAST_MS:
            before = v_ball[t - still:t]
            before = before[~np.isnan(before)]
            if before.size and np.min(before) < C.RESTART_SLOW_MS:
                return t
    return (start + end) // 2


def emissions(feat, events, team_names, n_frames):
    """(n_frames, 3) score matrix, and a short summary of what the tags contributed."""
    fps = C.FPS
    E = np.zeros((n_frames, 3))
    seen = feat["seen"]
    v = feat["v_ball"]

    # 1. Control: which team has a player on the ball, and is the ball slow enough to be controlled.
    def near(d):
        out = _sigmoid((C.CONTROL_RADIUS_M - d) / C.CONTROL_SOFT_M)
        return np.where(np.isnan(d), 0.0, out)

    near_h, near_a = near(feat["d_home"]), near(feat["d_away"])
    slow = np.clip((C.BALL_FAST_HI - np.nan_to_num(v, nan=0.0))
                   / (C.BALL_FAST_HI - C.BALL_FAST_LO), 0.0, 1.0)
    in_play_area = seen & ~feat["out"]
    # Control uses the long-gap interpolation too, weighted by its confidence (1 where seen).
    ctrl_area = feat["seen_long"] & ~feat["out"]
    ctrl = np.where(ctrl_area, (near_h - near_a) * slow * feat["conf"], 0.0)
    E[:, H] += C.W_CONTROL * ctrl / 2
    E[:, A] -= C.W_CONTROL * ctrl / 2

    # 1b. Team shape: the team on the ball is the stretched one. Only where the ball track says
    # nothing (weight 1 - conf): shape lags a turnover by seconds, so where the ball is seen it
    # would only delay the switch the control evidence already places.
    w_shape = C.W_SHAPE * feat["shape"] * (1.0 - feat["conf"])
    E[:, H] += w_shape / 2
    E[:, A] -= w_shape / 2

    # 2. Dead-ball evidence.
    cal = feat["calibration"]
    act = np.nan_to_num(feat["activity"], nan=cal["activity_threshold"] - cal["activity_spread"])
    low = np.clip((cal["activity_threshold"] - act) / cal["activity_spread"], -1.5, 1.5)
    E[:, D] += C.W_SLOW * low
    E[:, D] += C.W_OUT * feat["out"]
    on_ball = np.maximum(near_h, near_a)
    still = seen & (np.nan_to_num(v, nan=99.0) < C.BALL_STILL_MS)
    E[:, D] += C.W_STILL * (still * (1.0 - on_ball))
    E[:, D] += C.W_MISSING * ~feat["seen_raw"]
    moving = in_play_area & (np.nan_to_num(v, nan=0.0) > 2.0)
    E[:, D] -= C.W_LIVE * (moving * on_ball)

    # 3. Smart Tagging.
    tag_team = np.zeros(n_frames)
    n_phase = n_setpiece = n_goal = 0
    restarts = []
    for ev in events:
        s, e = max(0, ev["start"]), min(n_frames, ev["end"])
        if e <= s:
            continue
        side = tagging.team_side(ev["team_text"], team_names)
        if ev["code"] in tagging.PHASE_SIGN and side is not None:
            sign = tagging.PHASE_SIGN[ev["code"]] * (1 if side == H else -1)
            tag_team[s:e] += sign
            n_phase += 1
        elif ev["code"] == tagging.SET_PIECE:
            r = find_restart(s, e, v, fps)
            restarts.append((r, ev["type"]))
            pre = max(0, min(s, r - 2 * fps))
            E[pre:r, D] += C.W_SET_PIECE
            post = min(n_frames, r + int(C.RESTART_TEAM_SECONDS * fps))
            E[r:post, D] -= C.W_SET_PIECE
            if side is not None:
                bonus = C.W_RESTART_TEAM if side == H else -C.W_RESTART_TEAM
                E[r:post, H] += bonus / 2
                E[r:post, A] -= bonus / 2
            n_setpiece += 1
    for ev in events:
        if ev["code"] != tagging.GOAL:
            continue
        kickoffs = [r for r, typ in restarts
                    if typ == tagging.KICKOFF_AFTER_GOAL and ev["end"] < r < ev["end"] + 180 * fps]
        if kickoffs:
            E[ev["end"]:min(kickoffs), D] += C.W_AFTER_GOAL
            n_goal += 1
    tag_team = np.clip(tag_team, -1, 1)
    E[:, H] += C.W_TAG * tag_team
    E[:, A] -= C.W_TAG * tag_team

    summary = {"phase_tags_used": n_phase, "set_pieces_used": n_setpiece,
               "goals_used": n_goal,
               "kickoff_start_frames": [r for r, typ in restarts if typ == tagging.KICKOFF_START]}
    return E, summary


def dead_before_kickoff(E, first, last, kickoff_frames):
    """A half's graded window can open before its kick-off (FAQ.md). Everything from the start of
    the window to the detected kick-off restart is a dead ball. The kick-off comes from the Smart
    Tagging "Kick Off (Start)" set piece, found in the half's own window -- no frame is hard-coded.
    Returns the number of frames covered."""
    covered = 0
    for r in kickoff_frames:
        if first < r <= last:
            E[first:r, D] += C.W_PRE_KICKOFF
            covered += r - first
    return covered


def recentre_switches(path, E, missing, tol=None):
    """Move each change of team that touches a ball-unseen stretch to the middle of the positions
    that score within `tol` of the best one.

    The decoder breaks exact or near ties by state index, which places H->A late in a gap and
    A->H early. Here the candidates are every switch position inside the unseen stretch (bounded
    by the two possessions involved); the score of a position is the evidence the old team keeps
    up to it. Real evidence in the gap (interpolated ball, team shape) still decides; only where it
    is flat does the middle win.
    """
    tol = C.SWITCH_TIE_TOL if tol is None else tol
    path = path.copy()
    n = len(path)
    switches = [t for t in range(1, n)
                if path[t - 1] != path[t] and path[t - 1] != D and path[t] != D]
    for t in switches:
        s1, s2 = path[t - 1], path[t]
        if not (missing[t - 1] or missing[t]):
            continue
        i = t - 1 if missing[t - 1] else t
        lo = i
        while lo > 0 and missing[lo - 1]:
            lo -= 1
        hi = i
        while hi < n - 1 and missing[hi + 1]:
            hi += 1
        a = t - 1
        while a > 0 and path[a - 1] == s1:
            a -= 1
        b = t
        while b < n - 1 and path[b + 1] == s2:
            b += 1
        klo, khi = max(lo, a + 1), min(hi + 1, b)
        if khi <= klo:
            continue
        gain = np.concatenate(([0.0], np.cumsum(E[klo:khi, s1] - E[klo:khi, s2])))
        ks = np.arange(klo, khi + 1)
        cand = ks[gain >= gain.max() - tol]
        mid = (lo + hi + 1) / 2.0
        k = int(cand[np.argmin(np.abs(cand - mid))])
        path[klo:k] = s1
        path[k:khi] = s2
    return path


def viterbi(E, start_state=D):
    """Best state path under fixed switching costs. E is (n, 3)."""
    P = np.array([[0.0, C.P_SWITCH, C.P_DEAD],
                  [C.P_SWITCH, 0.0, C.P_DEAD],
                  [C.P_DEAD, C.P_DEAD, 0.0]])
    n = len(E)
    back = np.empty((n, 3), dtype=np.int8)
    score = E[0] - P[start_state]
    for t in range(1, n):
        cand = score[:, None] - P
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], (0, 1, 2)] + E[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
