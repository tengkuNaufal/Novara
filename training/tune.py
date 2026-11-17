"""Fit the team-side parameters on other games, and check them on a held-out game.

    python training/tune.py ../external/metrica

Design
------
The Metrica sample tracking blanks the ball almost whenever play is dead (92-100% of dead frames),
so on it "ball missing" would be a free and false proxy for a dead ball. The dead-ball side of the
model is therefore NOT fitted here. Instead the true dead-ball frames are given to the decoder
(oracle D), and only the parameters that decide WHICH team has the ball are fitted, scored by the
official spellF1. That isolates exactly the part this data can teach.

Validation is leave-one-game-out: fit on one game, score on the other, both ways. A parameter set
is only worth adopting if it beats the current one on the game it was not fitted on.
"""

import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from solution import config as C, features, model  # noqa: E402
from training import metrica  # noqa: E402

LETTERS = np.array(["H", "A", "D"])
SEEDS = (0, 1)               # two independent degradations per game

GRID = {
    "P_SWITCH": [4.0, 8.0, 12.0, 18.0, 25.0, 35.0],
    "CONTROL_RADIUS_M": [1.2, 1.8, 2.5, 3.5],
    "BALL_FAST": [(6.0, 12.0), (10.0, 20.0), (15.0, 30.0)],
    "W_SHAPE": [0.0, 0.1, 0.3],
    "W_TAG": [0.0, 0.03, 0.1],
}


def current():
    return {"P_SWITCH": C.P_SWITCH, "CONTROL_RADIUS_M": C.CONTROL_RADIUS_M,
            "BALL_FAST": (C.BALL_FAST_LO, C.BALL_FAST_HI), "W_SHAPE": C.W_SHAPE, "W_TAG": C.W_TAG}


def apply(params):
    C.P_SWITCH = params["P_SWITCH"]
    C.CONTROL_RADIUS_M = params["CONTROL_RADIUS_M"]
    C.BALL_FAST_LO, C.BALL_FAST_HI = params["BALL_FAST"]
    C.W_SHAPE = params["W_SHAPE"]
    C.W_TAG = params["W_TAG"]


def prepare(folder, game, seed):
    d = metrica.load_game(folder, game, seed)
    graded = d["gt"] != "X"
    d["feat"] = features.compute(d["track"], d["slot_team"], metrica.PITCH, graded)
    d["gt_codes"] = pm.label_codes(d["gt"])
    return d


def predict(d, oracle_dead=True):
    E, _ = model.emissions(d["feat"], d["tags"], d["team_names"], d["n"])
    if oracle_dead:
        dead = d["gt"] == "D"
        E[:, 2] = np.where(dead, 50.0, -50.0)
    labels = np.full(d["n"], "X", dtype="<U1")
    for _, first, last in d["blocks"]:
        path = model.viterbi(E[first:last + 1])
        path = model.recentre_switches(path, E[first:last + 1], ~d["feat"]["seen_raw"][first:last + 1])
        labels[first:last + 1] = LETTERS[path]
    return labels


def score(d, params, oracle_dead=True):
    apply(params)
    labels = predict(d, oracle_dead)
    r = pm.evaluate(d["gt_codes"], pm.label_codes(labels), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"]


def mean_score(games, params, oracle_dead=True):
    s = [score(d, params, oracle_dead) for d in games]
    return float(np.mean([x[0] for x in s])), float(np.mean([x[1] for x in s]))


def fit(games, start):
    """Coordinate ascent on mean spellF1, two passes over the grid."""
    best = dict(start)
    best_s = mean_score(games, best)[0]
    for _ in range(2):
        for key, values in GRID.items():
            for v in values:
                cand = dict(best)
                cand[key] = v
                s = mean_score(games, cand)[0]
                if s > best_s + 1e-9:
                    best, best_s = cand, s
    return best, best_s


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "external", "metrica")
    data = {g: [prepare(folder, g, s) for s in SEEDS] for g in (1, 2)}
    base = current()
    print("current parameters:", base)

    fitted = {}
    for train, test in ((1, 2), (2, 1)):
        p, s_train = fit(data[train], base)
        fitted[train] = p
        b_test = mean_score(data[test], base)
        p_test = mean_score(data[test], p)
        print("\nfit on game {} -> {}".format(train, p))
        print("  train spellF1 {:.4f}".format(s_train))
        print("  held-out game {}: current spellF1 {:.4f} macroF1 {:.4f} | fitted spellF1 {:.4f} macroF1 {:.4f}"
              .format(test, b_test[0], b_test[1], p_test[0], p_test[1]))

    allg = data[1] + data[2]
    p_all, s_all = fit(allg, base)
    print("\nfit on both games -> {}  (spellF1 {:.4f}, current {:.4f})".format(
        p_all, s_all, mean_score(allg, base)[0]))
    print("\nfull model (no oracle; Metrica's ball-missing leak flatters D -- for reference only)")
    print("  current {:.4f} / {:.4f}   fitted-on-both {:.4f} / {:.4f}  (spellF1 / macroF1)".format(
        *mean_score(allg, base, False), *mean_score(allg, p_all, False)))


if __name__ == "__main__":
    main()
