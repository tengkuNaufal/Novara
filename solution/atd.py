"""Read the Automatic Tracking Data (FIFA/EPTS export) into dense numpy arrays.

Raw line format: "<frame_id>:<x,y,s>;<x,y,s>;...:<ball x,y[,...]>", normalised 0-1 coordinates,
NaN for a missing observation, frame_id counting from 1 (video_frame = frame_id - 1).
"""

import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1
CHUNK = 4000


def read_meta(meta_path):
    """(team names in file order, team of each track slot). The first team is home."""
    root = ET.parse(meta_path).getroot()
    teams = [(team.get("id"), team.findtext("Name") or "") for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0][0]
    slot_team = np.array([HOME if p.get("teamId") == home_id else AWAY
                          for p in root.iter("Player")], dtype=np.int8)
    return [name for _, name in teams], slot_team


def _layout(raw_path):
    """(groups per line, values per player group) from the first well-formed line."""
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) >= 3:
                groups = parts[1].split(";")
                return len(groups), len(groups[0].split(","))
    raise ValueError("{}: no tracking lines found".format(raw_path))


def read_raw(raw_path, n_frames, pitch):
    """Players and ball in metres, indexed by video frame.

    Returns dict with px, py, ps (n_frames, n_slots) and bx, by (n_frames,). Player speed `ps` is
    taken from the file when present, otherwise left NaN and derived later.
    """
    length, width = pitch
    n_slots, per_group = _layout(raw_path)
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full_like(px, np.nan)
    ps = np.full_like(px, np.nan)
    bx = np.full(n_frames, np.nan)
    by = np.full(n_frames, np.nan)

    rows, players, balls = [], [], []

    def flush():
        if not rows:
            return
        idx = np.array(rows)
        flat = ",".join(players).split(",")
        if len(flat) == len(rows) * n_slots * per_group:
            values = np.array(flat, dtype=np.float64).reshape(len(rows), n_slots, per_group)
        else:  # a malformed line somewhere in the chunk: fall back to line by line
            values = np.full((len(rows), n_slots, per_group), np.nan)
            for k, text in enumerate(players):
                line_vals = np.array(text.split(","), dtype=np.float64)
                if line_vals.size == n_slots * per_group:
                    values[k] = line_vals.reshape(n_slots, per_group)
        px[idx] = values[:, :, 0] * length
        py[idx] = values[:, :, 1] * width
        if per_group >= 3:
            ps[idx] = values[:, :, 2]
        ball = np.array(balls, dtype=np.float64)
        bx[idx] = ball[:, 0] * length
        by[idx] = ball[:, 1] * width
        del rows[:], players[:], balls[:]

    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) < 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            ball = parts[2].split(",")
            if len(ball) < 2:
                ball = ["NaN", "NaN"]
            rows.append(frame)
            players.append(parts[1].replace(";", ","))
            balls.append((ball[0], ball[1]))
            if len(rows) >= CHUNK:
                flush()
    flush()
    return {"px": px, "py": py, "ps": ps, "bx": bx, "by": by}
