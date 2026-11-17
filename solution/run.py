"""Produce the possession file from the provided match data.

    python -m solution.run --team MyTeam
    python -m solution.run --team MyTeam --data-dir D:/mtah2027

Writes evaluation/results/<team>.csv (199,319 rows). Deterministic: the same data gives the same
file, byte for byte.

Everything printed is an aggregate (counts, durations, calibrated thresholds). No frame-level data
is printed, so the console output can be shared without sharing the match data.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from solution import atd, broadcast, config as C, features, model, tagging  # noqa: E402

LETTERS = np.array(["H", "A", "D"])


def label_track(track, slot_team, team_names, events, blocks, pitch, n_frames):
    """The method itself, independent of where the tracking came from: (labels, features, tag
    summary). `blocks` is [(half, first, last)]; frames outside them are written X."""
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    feat = features.compute(track, slot_team, pitch, graded)
    E, tag_summary = model.emissions(feat, events, team_names, n_frames)
    kickoffs = tag_summary.pop("kickoff_start_frames")
    tag_summary["pre_kickoff_dead_frames"] = sum(
        model.dead_before_kickoff(E, first, last, kickoffs) for _, first, last in blocks)

    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in blocks:
        path = model.viterbi(E[first:last + 1])
        path = model.recentre_switches(path, E[first:last + 1], ~feat["seen_raw"][first:last + 1])
        labels[first:last + 1] = LETTERS[path]
    return labels, feat, tag_summary


def predict(data_dir=None, verbose=True):
    """Label array (strings, one per video frame) and a diagnostics dict."""
    t0 = time.time()
    meta, raw, tags_path = require(["atd_meta", "atd_raw", "smart_tagging"], GAME, data_dir)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    blocks = half_video_blocks(GAME)

    team_names, slot_team = atd.read_meta(meta)
    if verbose:
        print("reading tracking ...")
    track = atd.read_raw(raw, n_frames, pitch)
    bc_stats = {"broadcast_used": False}
    if C.USE_BROADCAST:
        try:
            b_meta, b_raw, b_hom = require(["atd_meta", "atd_raw", "homography"], GAME, data_dir,
                                          source="broadcast")
        except (IOError, OSError, KeyError):
            b_raw = None
        if b_raw is not None:
            if verbose:
                print("aligning broadcast ...")
            nb = source_frames("broadcast", GAME)
            b_track = atd.read_raw(b_raw, nb, pitch)
            _, bc_stats = broadcast.fill_ball(track, b_track, broadcast.read_valid(b_hom, nb))
            bc_stats["broadcast_used"] = True

    events = tagging.load(tags_path, C.FPS)
    labels, feat, tag_summary = label_track(track, slot_team, team_names, events, blocks,
                                            pitch, n_frames)
    graded = labels != "X"

    diag = {
        "seconds": round(time.time() - t0, 1),
        "ball_seen_graded_pct": round(100.0 * feat["seen_raw"][graded].mean(), 1),
        "ball_seen_after_gap_fill_pct": round(100.0 * feat["seen"][graded].mean(), 1),
        "activity_threshold_ms": round(feat["calibration"]["activity_threshold"], 2),
        "ball_usable_long_gap_pct": round(100.0 * feat["seen_long"][graded].mean(), 1),
        "shape_centre_m": round(feat["calibration"]["shape_centre_m"], 2),
        "shape_spread_m": round(feat["calibration"]["shape_spread_m"], 2),
        "smart_tagging_instances": len(events),
    }
    diag.update(tag_summary)
    diag.update(bc_stats)
    return labels, diag


def write(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def summarise(labels):
    """Possession count and mean duration, computed the way the scorer defines possessions."""
    out = {}
    for letter in "HAD":
        out[letter] = int(np.sum(labels == letter))
    graded = labels != "X"
    runs = 0
    prev = None
    for lab, g in zip(labels, graded):
        if g and lab in "HA" and lab != prev:
            runs += 1
        prev = lab if g else None
    out["possessions"] = runs
    out["mean_possession_s"] = round((out["H"] + out["A"]) / C.FPS / max(runs, 1), 1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="team name; the file is <team>.csv")
    parser.add_argument("--data-dir", default=None, help="match bundle (default: ./data or "
                                                         "MTAH_DATA_DIR)")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "evaluation", "results"))
    args = parser.parse_args()

    labels, diag = predict(args.data_dir)
    path = os.path.join(args.out_dir, "{}.csv".format(args.team))
    write(path, labels)

    print("\ndiagnostics (aggregates only)")
    for key, value in list(diag.items()) + list(summarise(labels).items()):
        print("  {:<30} {}".format(key, value))
    print("\nwrote {}".format(path))
    print("next: python evaluation/validate_submission.py {}".format(path))


if __name__ == "__main__":
    main()
