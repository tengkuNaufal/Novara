"""Per-frame features computed from the tracking, plus thresholds learned from the match itself.

Nothing here reads a label. The only "learned" quantity is the live/dead threshold on collective
player movement, which is found by Otsu's method on this match's own movement distribution --
unsupervised, as RULES.md allows.
"""

import warnings

import numpy as np

from . import config as C
from .atd import HOME, AWAY


def fill_short_gaps(x, max_gap):
    """Linear interpolation over NaN runs of at most max_gap frames, bounded on both sides."""
    x = x.copy()
    missing = np.isnan(x)
    if not missing.any():
        return x
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    for start, end in zip(starts, ends):
        if start == 0 or end == len(x) or end - start > max_gap:
            continue
        x[start:end] = np.linspace(x[start - 1], x[end], end - start + 2)[1:-1]
    return x


def gap_length(missing):
    """For every frame, the length of the run of missing frames it belongs to (0 where seen)."""
    out = np.zeros(len(missing), dtype=np.int64)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        out[start:end] = end - start
    return out


def team_spread(px, py, cols, min_players):
    """Mean distance of a team's players to its median position, farthest player dropped.

    NaN on frames with fewer than min_players tracked. Dropping the farthest one removes the
    goalkeeper on almost every frame without having to know who the goalkeeper is.
    """
    x, y = px[:, cols].astype(np.float64), py[:, cols].astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cx = np.nanmedian(x, axis=1)
        cy = np.nanmedian(y, axis=1)
        d = np.hypot(x - cx[:, None], y - cy[:, None])
        count = np.sum(~np.isnan(d), axis=1)
        total = np.nansum(d, axis=1) - np.nanmax(d, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = total / (count - 1)
    spread[count < min_players] = np.nan
    return spread


def moving_nanmean(x, window):
    """Centred moving mean over +/- window frames, ignoring NaN. NaN where nothing is seen."""
    valid = ~np.isnan(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - window)
    hi = np.minimum(len(x), idx + window + 1)
    n = count[hi] - count[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[hi] - total[lo]) / n, np.nan)


def speed(x, y, fps, step=2):
    """Central-difference speed in m/s over +/- step frames. NaN where either end is unseen."""
    v = np.full(len(x), np.nan)
    dx = x[2 * step:] - x[:-2 * step]
    dy = y[2 * step:] - y[:-2 * step]
    v[step:-step] = np.hypot(dx, dy) * fps / (2.0 * step)
    return v


def otsu(values, bins=256):
    """Threshold that best separates a bimodal sample into two classes."""
    values = values[~np.isnan(values)]
    hist, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(hist)
    mean = np.cumsum(hist * centres)
    total, total_mean = weight[-1], mean[-1]
    w1 = total - weight
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (total_mean * weight / total - mean) ** 2 / (weight * w1)
    between[~np.isfinite(between)] = -1
    return float(centres[np.argmax(between)])


def compute(track, slot_team, pitch, graded):
    """All frame features as a dict of 1-D arrays, plus the self-calibrated thresholds."""
    fps = C.FPS
    length, width = pitch
    n_slots = track["px"].shape[1]
    slot_team = slot_team[:n_slots]
    if len(slot_team) < n_slots:  # metadata lists fewer slots than the file: treat extras as unknown
        slot_team = np.concatenate((slot_team, np.full(n_slots - len(slot_team), -1, np.int8)))

    seen_raw = ~np.isnan(track["bx"])
    bx = fill_short_gaps(track["bx"], C.BALL_MAX_GAP)
    by = fill_short_gaps(track["by"], C.BALL_MAX_GAP)
    seen = ~np.isnan(bx)

    # Longer gaps: interpolated for the control evidence only, with a confidence that decays with
    # the length of the gap. Short gaps (and seen frames) keep confidence 1.
    gap = gap_length(~seen_raw)
    bx_l = fill_short_gaps(track["bx"], C.BALL_MAX_GAP_LONG)
    by_l = fill_short_gaps(track["by"], C.BALL_MAX_GAP_LONG)
    seen_long = ~np.isnan(bx_l)
    span = max(1.0, float(C.BALL_MAX_GAP_LONG - C.BALL_MAX_GAP))
    conf = np.where(gap <= C.BALL_MAX_GAP, 1.0,
                    np.clip((C.BALL_MAX_GAP_LONG - gap) / span, 0.0, 1.0))
    conf = np.where(seen_long, conf, 0.0)
    v_ball = speed(bx_l, by_l, fps)

    # Distance from the ball to the nearest player of each team.
    dx = track["px"] - bx_l[:, None]
    dy = track["py"] - by_l[:, None]
    dist = np.hypot(dx, dy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d_home = np.nanmin(np.where(slot_team[None, :] == HOME, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(slot_team[None, :] == AWAY, dist, np.nan), axis=1)

    # Collective movement. Use the file's speeds; derive them where the file has none.
    ps = track["ps"]
    if np.all(np.isnan(ps)):
        ps = np.stack([speed(track["px"][:, k], track["py"][:, k], fps)
                       for k in range(n_slots)], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_speed = np.nanmean(ps, axis=1)
    activity = moving_nanmean(mean_speed, int(C.ACTIVITY_WINDOW_S * fps))

    # Live/dead threshold on movement, from this match's own graded-window distribution.
    act_graded = activity[graded]
    threshold = otsu(act_graded)
    spread = float(np.nanstd(act_graded)) or 1.0

    # Team shape: home spread minus away spread, smoothed, in units of its own spread (no labels).
    s_home = team_spread(track["px"], track["py"], slot_team == HOME, C.SHAPE_MIN_PLAYERS)
    s_away = team_spread(track["px"], track["py"], slot_team == AWAY, C.SHAPE_MIN_PLAYERS)
    shape_raw = moving_nanmean(s_home - s_away, int(C.SHAPE_WINDOW_S * fps))
    # Centre: midway between the shape seen while home clearly controls a visible ball and while
    # away does -- not 0, because the two teams' structures and tracking coverage differ. This uses
    # the tracking's own control evidence, never a label.
    v_tmp = np.nan_to_num(v_ball, nan=99.0)
    clear = seen_raw & graded & (v_tmp < C.BALL_FAST_LO)
    far = 2.0 * C.CONTROL_RADIUS_M
    home_clear = clear & (d_home < C.CONTROL_RADIUS_M) & (d_away > far)
    away_clear = clear & (d_away < C.CONTROL_RADIUS_M) & (d_home > far)
    if home_clear.sum() > 100 and away_clear.sum() > 100:
        shape_centre = 0.5 * (np.nanmean(shape_raw[home_clear]) + np.nanmean(shape_raw[away_clear]))
    else:
        shape_centre = otsu(shape_raw[graded])
    shape_spread = float(np.nanstd(shape_raw[graded])) or 1.0
    shape = np.clip(np.nan_to_num((shape_raw - shape_centre) / shape_spread, nan=0.0), -1.0, 1.0)

    out = seen & ((bx < -C.OUT_MARGIN_M) | (bx > length + C.OUT_MARGIN_M)
                  | (by < -C.OUT_MARGIN_M) | (by > width + C.OUT_MARGIN_M))

    return {
        "seen_raw": seen_raw, "seen": seen, "bx": bx, "by": by, "v_ball": v_ball,
        "d_home": d_home, "d_away": d_away, "activity": activity, "out": out,
        "seen_long": seen_long, "conf": conf, "shape": shape,
        "calibration": {"activity_threshold": threshold, "activity_spread": spread,
                        "shape_centre_m": shape_centre, "shape_spread_m": shape_spread},
    }
