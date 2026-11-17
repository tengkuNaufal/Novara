"""Smart Tagging as weak per-frame evidence.

Phases carry a Team tag. For attacking phases the tagged team is the one in possession; for
defending phases and defensive transitions it is the team *without* the ball. Set pieces mark
restarts: dead before, alive after. Goals are followed by a dead ball until the next kick-off.

Tags are automatic and noisy, so they only ever add small weights -- they are never copied across
as labels.
"""

import difflib
import xml.etree.ElementTree as ET

# Code -> +1 if the tagged team has the ball, -1 if the tagged team is the one defending.
PHASE_SIGN = {
    "build up": 1, "progression": 1, "creating chances": 1, "attacking transition": 1,
    "long ball": 1, "ball in final third": 1, "ball in the box": 1, "players in the box": 1,
    "defending in attacking third": -1, "defending in middle third": -1,
    "defending in defensive third": -1, "defensive transition": -1,
}
SET_PIECE = "set pieces"
GOAL = "goals"
KICKOFF_AFTER_GOAL = "kick off (after goal)"
KICKOFF_START = "kick off (start)"


def load(path, fps):
    """[{code, start, end, team_text, type}] with frames, code and type lower-cased."""
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {(lab.findtext("group") or "").strip().lower(): (lab.findtext("text") or "").strip()
                for lab in inst.findall("label")}
        out.append({
            "code": (inst.findtext("code") or "").strip().lower(),
            "start": int(round(float(inst.findtext("start")) * fps)),
            "end": int(round(float(inst.findtext("end")) * fps)),
            "team_text": tags.get("team", ""),
            "type": tags.get("type", "").lower(),
        })
    return out


def team_side(team_text, team_names):
    """0 home, 1 away, None if the tag names no team. Matched by name similarity, because the
    tagging and the tracking spell clubs differently (e.g. 'Atletic Club')."""
    text = team_text.strip().lower()
    if not text or text in ("n/a", "na", "none"):
        return None
    scores = [difflib.SequenceMatcher(None, text, name.lower()).ratio() for name in team_names]
    for side, name in enumerate(team_names):  # containment beats similarity
        if text in name.lower() or name.lower() in text:
            scores[side] += 1.0
    return 0 if scores[0] >= scores[1] else 1
