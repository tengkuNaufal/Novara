"""Labelled data from OTHER games: Metrica Sports' public sample data (games 1 and 2).

RULES.md allows labelled data about other games. This module turns the sample data into exactly
what the solution consumes, so the very same code (solution.run.label_track) can be scored with
the official metric library on matches whose answer we know:

- tracking   players and ball, normalised 0-1 -> metres on a 105 x 68 pitch
- labels     H / A / D per frame, derived from the event data (see build_labels)
- tags       Smart-Tagging-like set-piece windows, simulated from the set-piece events
- degraded   the sample tracking is clean; ours is not. The ball is removed in long blocks with the
             same mean lengths as our match's own ball gaps, positions get noise, players drop out.

Download (public, MIT): https://github.com/metrica-sports/sample-data -> external/metrica/
"""

import csv
import os

import numpy as np

PITCH = (105.0, 68.0)
FPS = 25
HOME, AWAY = 0, 1

# Ball gap structure measured on our match (aggregates only): mean run lengths in frames.
MEAN_MISSING_RUN = 431
MEAN_VISIBLE_RUN = 632
POS_NOISE_M = 0.3
PLAYER_DROP = 0.07          # share of player-frames lost, in 1 s blocks (ours: 20.5 of 22 tracked)
TAG_LEN = (725, 875)        # Smart Tagging set-piece window length, frames (our p10..median)

POSSESSION_TYPES = {"PASS", "RECOVERY", "SET PIECE", "SHOT"}
RESTART_TYPES = {"PASS", "RECOVERY", "SET PIECE", "SHOT"}


def _read_tracking(path):
    with open(path) as handle:
        rows = list(csv.reader(handle))
    header = rows[2]
    data = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in r]
                     for r in rows[3:]], dtype=np.float64)
    names = [h for h in header[3:] if h]
    return header, data, names


def load_tracking(folder, game):
    """track dict in metres (index = Frame - 1), slot_team, period array."""
    base = os.path.join(folder, "Sample_Game_{0}".format(game), "Sample_Game_{0}_RawTrackingData_".format(game))
    _, home, h_names = _read_tracking(base + "Home_Team.csv")
    _, away, a_names = _read_tracking(base + "Away_Team.csv")
    n = int(max(home[:, 1].max(), away[:, 1].max()))
    hp = (len(h_names) - 1)              # last name is Ball
    ap = (len(a_names) - 1)
    px = np.full((n, hp + ap), np.nan)
    py = np.full_like(px, np.nan)
    hi = home[:, 1].astype(int) - 1
    ai = away[:, 1].astype(int) - 1
    for k in range(hp):
        px[hi, k] = home[:, 3 + 2 * k] * PITCH[0]
        py[hi, k] = home[:, 4 + 2 * k] * PITCH[1]
    for k in range(ap):
        px[ai, hp + k] = away[:, 3 + 2 * k] * PITCH[0]
        py[ai, hp + k] = away[:, 4 + 2 * k] * PITCH[1]
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    bx[hi] = home[:, 3 + 2 * hp] * PITCH[0]
    by[hi] = home[:, 4 + 2 * hp] * PITCH[1]
    period = np.zeros(n, dtype=int)
    period[hi] = home[:, 0].astype(int)
    slot_team = np.array([HOME] * hp + [AWAY] * ap, dtype=np.int8)
    track = {"px": px, "py": py, "ps": np.full_like(px, np.nan), "bx": bx, "by": by}
    return track, slot_team, period


def load_events(folder, game):
    path = os.path.join(folder, "Sample_Game_{0}".format(game), "Sample_Game_{0}_RawEventsData.csv".format(game))
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for r in rows:
        out.append({"team": HOME if r["Team"] == "Home" else AWAY, "type": r["Type"],
                    "sub": r["Subtype"] or "", "period": int(r["Period"]),
                    "start": int(r["Start Frame"]) - 1,
                    "end": max(int(r["Start Frame"]), int(r["End Frame"])) - 1})
    out.sort(key=lambda e: (e["start"], e["end"]))
    return out


def _stops(e):
    """Frame at which this event stops play, or None."""
    sub = e["sub"].upper()
    if e["type"] == "BALL OUT":
        return e["end"]
    if e["type"] == "SHOT" and ("OUT" in sub or "GOAL" in sub):
        return e["end"]
    if e["type"] == "FAULT RECEIVED":
        return e["start"]
    return None


def build_labels(events, n):
    """H/A/D/X per frame and [(half, first, last)] graded blocks.

    Alive: the team of the latest PASS / RECOVERY / SET PIECE / SHOT has the ball (a BALL LOST is
    followed by the opponent's RECOVERY, which is where the change is dated). Dead: from a ball
    out, a shot out or in, or a foul, until the next restart-type event starts. Graded: each period
    from its first event (the kick-off) to its last event's end.
    """
    labels = np.full(n, "X", dtype="<U1")
    blocks = []
    for half in (1, 2):
        ev = [e for e in events if e["period"] == half]
        if not ev:
            continue
        first, last = ev[0]["start"], min(n - 1, max(e["end"] for e in ev))
        blocks.append((half, first, last))
        team = ev[0]["team"]
        cur = first
        state = "H" if team == HOME else "A"
        dead_until = None
        for i, e in enumerate(ev):
            s = min(e["start"], last)
            if s > cur:
                labels[cur:s] = state
                cur = s
            if dead_until is not None:
                if e["type"] in RESTART_TYPES:
                    dead_until = None
                else:
                    continue
            if e["type"] in POSSESSION_TYPES:
                state = "H" if e["team"] == HOME else "A"
            stop = _stops(e)
            if stop is not None:
                stop = min(stop, last)
                if stop > cur:
                    labels[cur:stop] = state
                    cur = stop
                state = "D"
                dead_until = True
        labels[cur:last + 1] = state
    return labels, blocks


def simulate_tags(events, rng):
    """Smart-Tagging-like set-piece windows (the kind solution.tagging.load returns).

    One window per SET PIECE, of a length drawn from our match's tag lengths, placed so the
    restart falls somewhere inside it. Team as the tagging gives it: N/A for throw-ins.
    """
    tags = []
    goal_pending = False
    first_kickoff = set()
    for e in events:
        if e["type"] == "SHOT" and "GOAL" in e["sub"].upper():
            goal_pending = True
            g0 = e["end"]
            tags.append({"code": "goals", "start": max(0, g0 - 250), "end": g0 + 25,
                         "team_text": "Home" if e["team"] == HOME else "Away", "type": ""})
        if e["type"] != "SET PIECE":
            continue
        sub = e["sub"].upper()
        if "KICK OFF" in sub:
            if e["period"] not in first_kickoff:
                first_kickoff.add(e["period"])
                typ = "kick off (start)"
            elif goal_pending:
                typ = "kick off (after goal)"
            else:
                typ = "kick off (start)"
            goal_pending = False
        elif "THROW" in sub:
            typ = "throw-in"
        elif "CORNER" in sub:
            typ = "corner kick"
        elif "FREE" in sub:
            typ = "free kick"
        else:
            typ = sub.lower()
        length = int(rng.integers(TAG_LEN[0], TAG_LEN[1] + 1))
        before = int(length * rng.uniform(0.2, 0.8))
        tags.append({"code": "set pieces", "start": max(0, e["start"] - before),
                     "end": e["start"] - before + length,
                     "team_text": "N/A" if typ == "throw-in" else ("Home" if e["team"] == HOME else "Away"),
                     "type": typ})
    return tags


def degrade(track, rng):
    """A copy of the tracking with our match's kind of imperfections."""
    t = {k: v.copy() for k, v in track.items()}
    n = len(t["bx"])
    # ball: alternating visible / missing blocks with exponential lengths of our means
    mask = np.zeros(n, dtype=bool)
    i, visible = 0, rng.random() < MEAN_VISIBLE_RUN / (MEAN_VISIBLE_RUN + MEAN_MISSING_RUN)
    while i < n:
        mean = MEAN_VISIBLE_RUN if visible else MEAN_MISSING_RUN
        length = max(1, int(rng.exponential(mean)))
        if not visible:
            mask[i:i + length] = True
        i += length
        visible = not visible
    t["bx"][mask] = np.nan
    t["by"][mask] = np.nan
    t["bx"] += rng.normal(0, POS_NOISE_M, n)
    t["by"] += rng.normal(0, POS_NOISE_M, n)
    # players: noise, and dropouts in 1 s blocks
    t["px"] += rng.normal(0, POS_NOISE_M, t["px"].shape)
    t["py"] += rng.normal(0, POS_NOISE_M, t["py"].shape)
    blocks = rng.random(((n + FPS - 1) // FPS, t["px"].shape[1])) < PLAYER_DROP
    drop = np.repeat(blocks, FPS, axis=0)[:n]
    t["px"][drop] = np.nan
    t["py"][drop] = np.nan
    return t


def load_game(folder, game, seed=0):
    """Everything label_track needs for one game, degraded, plus its ground truth."""
    rng = np.random.default_rng(1000 * game + seed)
    track, slot_team, _ = load_tracking(folder, game)
    events = load_events(folder, game)
    labels, blocks = build_labels(events, len(track["bx"]))
    tags = simulate_tags(events, rng)
    return {"track": degrade(track, rng), "clean": track, "slot_team": slot_team,
            "team_names": ["Home", "Away"], "tags": tags, "blocks": blocks, "gt": labels,
            "n": len(track["bx"])}
