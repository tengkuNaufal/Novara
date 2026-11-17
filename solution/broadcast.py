"""Fill tactical ball gaps with the broadcast ball track.

The broadcast recording is not frame-aligned to the tactical one and no mapping is provided. Both
ATD files are in the same pitch coordinates, though, so the timelines can be aligned from the
tracking alone, without labels:

1. Cut the broadcast into 10 s windows. For each window with enough visible ball, find the tactical
   frame offset at which the two ball tracks agree (median distance <= 1 m). Coarse search every
   5 frames over the whole match, refined to one frame.
2. The offset is piecewise constant -- it only changes at broadcast cuts (replays, halftime). An
   unmatched window lying between two matched windows with the same offset (within 1 s) inherits
   it, unless its own ball data contradicts that offset (enough ball seen in both recordings, and
   the median distance between them above 1 m) -- a replay inside the stretch, for instance.
3. Where the tactical ball is missing and the mapped broadcast frame has a ball *and* a valid
   broadcast homography (so no replay, close-up or crowd shot), the broadcast position is used.

Nothing here reads a label or a tactical frame index from anywhere but the two tracking files.
"""

import csv
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import config as C


def _window_offset(tbx, tby, bbx, bby):
    """(best tactical start frame, median ball distance) for one broadcast window, or None."""
    step, win = C.BC_STEP, len(bbx)
    wx, wy = bbx[::step], bby[::step]
    seen = ~np.isnan(wx)
    if seen.sum() < C.BC_MIN_COMMON:
        return None
    k = len(wx)
    tx = sliding_window_view(tbx[::step], k)
    ty = sliding_window_view(tby[::step], k)
    d = np.hypot(tx[:, seen] - wx[seen], ty[:, seen] - wy[seen])
    common = np.sum(~np.isnan(d), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(d, axis=1)
    med[common < C.BC_MIN_COMMON] = np.inf
    j = int(np.argmin(med))
    if not np.isfinite(med[j]):
        return None
    best = (np.inf, None)
    for s in range(max(0, j * step - step), min(len(tbx) - win, j * step + step) + 1):
        dd = np.hypot(tbx[s:s + win] - bbx, tby[s:s + win] - bby)
        if np.sum(~np.isnan(dd)) >= C.BC_MIN_COMMON * step // 2:
            m = float(np.nanmedian(dd))
            if m < best[0]:
                best = (m, s)
    if best[1] is None:
        return None
    return best[1], best[0]


def align(tbx, tby, bbx, bby):
    """Broadcast frame -> tactical frame (-1 where unknown), and aggregate stats."""
    nb, win = len(bbx), C.BC_WINDOW
    mapping = np.full(nb, -1, dtype=np.int64)
    starts = list(range(0, nb - win + 1, win))
    offset = {}
    errs = []
    for st in starts:
        r = _window_offset(tbx, tby, bbx[st:st + win], bby[st:st + win])
        if r is not None and r[1] <= C.BC_MATCH_M:
            offset[st] = r[0] - st
            errs.append(r[1])
    matched = len(offset)
    keys = sorted(offset)
    rejected = 0
    for a, b in zip(keys[:-1], keys[1:]):
        if b - a > win and abs(offset[a] - offset[b]) <= C.FPS:
            for st in range(a + win, b, win):
                # Inherit only if the window's own ball data does not contradict the offset:
                # either too little ball seen in both to tell, or the two tracks agree.
                tg = np.arange(st, st + win) + offset[a]
                tg = tg[(tg >= 0) & (tg < len(tbx))]
                src = tg - offset[a]
                d = np.hypot(tbx[tg] - bbx[src], tby[tg] - bby[src])
                d = d[~np.isnan(d)]
                if d.size >= C.BC_MIN_COMMON and np.median(d) > C.BC_MATCH_M:
                    rejected += 1
                    continue
                offset[st] = offset[a]
    for st, off in offset.items():
        mapping[st:st + win] = np.arange(st, st + win) + off
    mapping[(mapping < 0) | (mapping >= len(tbx))] = -1
    stats = {"broadcast_windows": len(starts), "broadcast_windows_matched": matched,
             "broadcast_windows_inherited": len(offset) - matched,
             "broadcast_windows_rejected": rejected,
             "broadcast_match_median_m": round(float(np.median(errs)), 2) if errs else None}
    return mapping, stats


def read_valid(homography_path, n):
    """Broadcast homography `valid` column as a boolean array of length n."""
    valid = np.zeros(n, dtype=bool)
    with open(homography_path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            f = int(row["frame"])
            if 0 <= f < n:
                valid[f] = row["valid"].strip() == "1"
    return valid


def fill_ball(track, b_track, b_valid):
    """Fill NaN tactical ball positions in `track` (in place). Returns (mask filled, stats)."""
    tbx, tby = track["bx"], track["by"]
    bbx, bby = b_track["bx"], b_track["by"]
    mapping, stats = align(tbx, tby, bbx, bby)
    ok = (mapping >= 0) & ~np.isnan(bbx) & b_valid[:len(bbx)]
    tgt = mapping[ok]
    src = np.where(ok)[0]
    use = np.isnan(tbx[tgt])
    tgt, src = tgt[use], src[use]
    filled = np.zeros(len(tbx), dtype=bool)
    tbx[tgt] = bbx[src]
    tby[tgt] = bby[src]
    filled[tgt] = True
    stats["broadcast_ball_frames_filled"] = int(filled.sum())
    return filled, stats
