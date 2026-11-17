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
from solution import atd, config as C, features, model, tagging  # noqa: E402

LETTERS = np.array(["H", "A", "D"])


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

    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    feat = features.compute(track, slot_team, pitch, graded)
    events = tagging.load(tags_path, C.FPS)
    E, tag_summary = model.emissions(feat, events, team_names, n_frames)

    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in blocks:
        path = model.viterbi(E[first:last + 1])
        path = model.recentre_switches(path, E[first:last + 1], ~feat["seen_raw"][first:last + 1])
        labels[first:last + 1] = LETTERS[path]

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
