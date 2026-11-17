"""A fake match bundle in the real file formats, with its own ground truth. For testing only.

It exists so the pipeline can be run end to end without the gated data: it checks parsing, frame
alignment, file format and that the decoder recovers possessions from noisy proximity. It says
nothing about the score on the real match -- the simulation is far simpler than football.

    python -m solution.synthetic OUT_DIR
    python -m solution.run --team synth --data-dir OUT_DIR --out-dir OUT_DIR
    python evaluation/run_evaluation.py --gt OUT_DIR/synthetic_gt.csv --results OUT_DIR
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402

FPS = 25
N_SLOTS = 46


def simulate(seed=0):
    rng = np.random.default_rng(seed)
    n = source_frames("tactical", GAME)
    L, W = GAMES[GAME]["pitch"]
    gt = np.full(n, "X", dtype="<U1")
    px = np.full((n, N_SLOTS), np.nan)
    py = np.full_like(px, np.nan)
    ps = np.full_like(px, np.nan)
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    tags = []  # (code, start_s, end_s, team, type)
    home_slots, away_slots = np.arange(0, 11), np.arange(23, 34)

    for half, first, last in half_video_blocks(GAME):
        m = last - first + 1
        # Players: smooth random walks inside the pitch.
        pos = np.column_stack([rng.uniform(5, L - 5, 22), rng.uniform(5, W - 5, 22)])
        vel = np.zeros((22, 2))
        slots = np.concatenate([home_slots, away_slots])
        labels = np.empty(m, dtype="<U1")

        # Build the label timeline.
        t = 2934 if half == 2 else 0  # second half graded before kick-off
        labels[:t] = "D"
        team = "H" if half == 1 else "A"
        while t < m:
            dur = int(max(1.0, rng.exponential(12.0)) * FPS)
            labels[t:t + dur] = team
            t += dur
            if rng.random() < 0.3:
                dd = int(rng.uniform(5, 35) * FPS)
                labels[t:t + dd] = "D"
                team = team if rng.random() < 0.5 else ("A" if team == "H" else "H")
                restart = t + dd
                if restart < m:
                    tags.append(("SET PIECES", (first + restart) / FPS - 2.5,
                                 (first + restart) / FPS + 2.5,
                                 "N/A" if rng.random() < 0.4 else
                                 ("Barcelona" if team == "H" else "Atletic Club"), "Throw-in"))
                t = restart
            else:
                team = "A" if team == "H" else "H"
        gt[first:last + 1] = labels

        # Move players and the ball.
        carrier = None
        target = None
        pass_left = 0
        for k in range(m):
            lab = labels[k]
            live = lab in "HA"
            scale = 1.0 if live else 0.3
            vel = 0.9 * vel + rng.normal(0, 0.25 * scale, vel.shape)
            vel = np.clip(vel, -7, 7)
            pos += vel / FPS
            pos[:, 0] = np.clip(pos[:, 0], 0.5, L - 0.5)
            pos[:, 1] = np.clip(pos[:, 1], 0.5, W - 0.5)
            f = first + k
            px[f, slots] = pos[:, 0]
            py[f, slots] = pos[:, 1]
            ps[f, slots] = np.hypot(vel[:, 0], vel[:, 1])
            if live:
                own = range(0, 11) if lab == "H" else range(11, 22)
                if carrier is None or carrier not in own:
                    carrier = int(rng.choice(list(own)))
                    pass_left = 0
                if pass_left > 0:
                    frac = 1.0 / pass_left
                    ball = ball + (pos[target] - ball) * frac
                    pass_left -= 1
                    if pass_left == 0:
                        carrier = target
                else:
                    ball = pos[carrier] + rng.normal(0, 0.4, 2)
                    if rng.random() < 1.0 / (2.0 * FPS):  # pass about every 2 s
                        target = int(rng.choice([p for p in own if p != carrier]))
                        pass_left = int(rng.uniform(0.5, 1.5) * FPS)
                bx[f], by[f] = ball
            else:
                carrier = None
                if k == 0 or labels[k - 1] in "HA":
                    # Stoppage starts: ball out over a touchline, or at the centre spot pre-kick-off.
                    dead_pos = (np.array([L / 2, W / 2]) if k == 0
                                else np.array([rng.uniform(5, L - 5), -2.0]))
                ball = dead_pos
                bx[f], by[f] = ball

    # Tracking loss: ball missing in bursts (~40%), players in occasional gaps.
    missing = np.zeros(n, dtype=bool)
    t = 0
    while t < n:
        t += int(rng.exponential(3.0) * FPS)
        dur = int(rng.exponential(2.0) * FPS)
        missing[t:t + dur] = True
        t += dur
    bx[missing] = np.nan
    by[missing] = np.nan
    gaps = rng.random(px.shape) < 0.03
    px[gaps] = py[gaps] = ps[gaps] = np.nan
    return gt, (px / L, py / W, ps, bx / L, by / W), tags


def write_bundle(out_dir, seed=0):
    gt, (px, py, ps, bx, by), tags = simulate(seed)
    folder = os.path.join(out_dir, "tactical")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, "BAR-ATH_tactical")

    with open(stem + "_atd.xml", "w") as fh:
        fh.write('<?xml version="1.0"?>\n<Root><Teams>\n'
                 '<Team id="T1"><Name>FC Barcelona</Name></Team>\n'
                 '<Team id="T2"><Name>Athletic Club</Name></Team>\n</Teams><Players>\n')
        for k in range(N_SLOTS):
            fh.write('<Player id="P{0}" teamId="{1}"><Name>Track {2}</Name>'
                     '<ShirtNumber>{2}</ShirtNumber></Player>\n'
                     .format(k, "T1" if k < 23 else "T2", k + 1))
        fh.write("</Players></Root>\n")

    def fmt(x):
        return "NaN" if np.isnan(x) else "{:.5f}".format(x)

    with open(stem + "_atd.txt", "w") as fh:
        for f in range(len(gt)):
            players = ";".join("{},{},{}".format(fmt(px[f, k]), fmt(py[f, k]), fmt(ps[f, k]))
                               for k in range(N_SLOTS))
            fh.write("{}:{}:{},{},{}\n".format(f + 1, players, fmt(bx[f]), fmt(by[f]), "NaN"))

    with open(stem + "_smart_tagging.xml", "w") as fh:
        fh.write('<?xml version="1.0"?>\n<file><ALL_INSTANCES>\n')
        for i, (code, s, e, team, typ) in enumerate(tags):
            fh.write("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>"
                     "<label><group>Team</group><text>{}</text></label>"
                     "<label><group>Type</group><text>{}</text></label></instance>\n"
                     .format(i, s, e, code, team, typ))
        fh.write("</ALL_INSTANCES></file>\n")

    with open(os.path.join(out_dir, "synthetic_gt.csv"), "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(gt))


if __name__ == "__main__":
    write_bundle(sys.argv[1])
    print("wrote synthetic bundle to", sys.argv[1])
