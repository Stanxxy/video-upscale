"""Founder ground truth and the scores every pre-label model is compared on.

``EXPECTED``/``NEGATIVE`` are the founder's labels for Cage Warriors R2 3:50-4:47 (XFBseoY4ER0,
2026-11-15): a = Dark brown shorts (fighter-a), b = Light grey shorts (fighter-b).
Sequences are scored at exchange level (untimed steps): a labelled technique is credited to
a sequence of the same fighter whose span covers its time (±600 ms).
"""
STRIKES = {"strike_punch", "strike_kick", "strike_knee", "strike_elbow", "ground_strike"}
# (t_ms, fighter, name, outcome or None). name is a technique name or an action class.
EXPECTED = [
    (230900, "a", "jab", "unsuccessful"), (231300, "a", "jab", None),          # "covers two jabs"
    (232450, "a", "side_kick", "successful"),                                   # "side kick", landed
    (232900, "a", "strike_kick", None), (233300, "a", "takedown_attempt", None),  # "kick and takedown"
    (250100, "b", "jab", None), (250400, "b", "hook", None),                    # wrong fighter, "jab and hook"
    (251600, "b", "jab", None), (251850, "b", "jab", None), (252200, "b", "strike_kick", None),  # "jab jab kick"
    (258600, "b", "roundhouse", None),                                          # "defense and low kick"
    (262200, "b", "strike_kick", "unsuccessful"),
    (264300, "a", "hook", None),                                                # "a right hook"
    (264400, "b", "jab", "successful"),
    (264600, "a", "overhand", "successful"),
    (265400, "a", "hook", "unsuccessful"),                                      # "left hook"
    (265900, "a", "overhand", None),                                            # "...and the right overhand"
    (266250, "a", "overhand", None),                                            # "right overhand again, not a cross"
    (267700, "a", "strike_kick", "successful"),
    (270100, "b", "jab", "successful"),
    (270200, "a", "hook", "unsuccessful"),
    (271200, "a", "jab", None),                                                 # "a jab after a hook"
    (272900, "b", "jab", "successful"),
    (273200, "b", "double_leg", "successful"),
    (275200, "b", "single_leg", "successful"),
    (278100, "b", "clinch_knee", None),                                         # wrong fighter + "should be clinch knee"
    (283900, "b", "clinch_entry", None), (284100, "b", "strike_punch", None),   # "covers both clinch and a punch"
    (284600, "a", "jab", None),                                                 # wrong fighter
    (286000, "a", "cross", None),                                               # wrong fighter
]
# (t_ms, fighter, what must not be there): blocks and defence read as techniques.
NEGATIVE = [
    (258500, "a", "strike"), (264000, "a", "strike"), (266050, "b", "strike"),
    (266500, "b", "strike"), (283500, "b", "get_up"), (285500, "b", "takedown_attempt"),
]
SPAN = (230000, 287000)
PAD = 600
GAP = 1500


def kind(n):
    if n in ("jab", "cross", "hook", "overhand", "uppercut", "strike_punch"): return "punch"
    if n in ("side_kick", "roundhouse", "teep", "calf_kick", "strike_kick"): return "kick"
    if n in ("clinch_knee", "strike_knee"): return "knee"
    if n in ("double_leg", "single_leg", "takedown_attempt"): return "takedown"
    if n == "clinch_entry": return "clinch"
    return n


def _groups(items):
    items, out = sorted(items), []
    for it in items:
        if out and it[0] - out[-1][-1][0] <= GAP:
            out[-1].append(it)
        else:
            out.append([it])
    return out


TRUTH = _groups([(t, a, kind(n)) for t, a, n, o in EXPECTED])


def score_minute(sequences):
    """sequences: flat dicts (actor a|b, start, end, steps[action, technique]) for the whole round."""
    qs = []
    for q in sequences:
        if q["end"] < SPAN[0] or q["start"] >= SPAN[1]:
            continue
        names = [st["technique"] or st["action"] for st in q["steps"]]
        acts = [st["action"] for st in q["steps"]]
        qs.append(dict(actor=q["actor"], start=q["start"], end=q["end"], names=names, acts=acts))
    covers = lambda q, t, pad=PAD: q["start"] - pad <= t <= q["end"] + pad
    right = other_only = named = 0
    for t, a, n, o in EXPECTED:
        mine = [q for q in qs if q["actor"] == a and covers(q, t)]
        if mine:
            right += 1
            named += any(n in q["names"] or n in q["acts"] for q in mine)
        elif any(covers(q, t) for q in qs):
            other_only += 1
    defence = sum(any(q["actor"] == a and covers(q, t, 300) and (any(c in STRIKES for c in q["acts"]) if k == "strike" else k in q["acts"])
                      for q in qs) for t, a, k in NEGATIVE)
    dup = sum(1 for i, p in enumerate(qs) for q in qs[i + 1:] if p["actor"] == q["actor"] and p["start"] < q["end"] and q["start"] < p["end"])
    found = starter = types = count = 0
    for g in TRUTH:
        lo, hi = g[0][0] - 700, g[-1][0] + 700
        mine = [q for q in qs if q["start"] <= hi and q["end"] >= lo]
        if not mine:
            continue
        found += 1
        starter += min(mine, key=lambda q: q["start"])["actor"] == g[0][1]
        types += {kind(n) for q in mine for n in q["names"]} == {k for _, _, k in g}
        count += abs(sum(len(q["names"]) for q in mine) - len(g)) <= 1
    return dict(sequences=len(qs), right_fighter=right, other_fighter_only=other_only, name_present=named,
                defence_counted=defence, duplicates=dup, ex_found=found, ex_starter=starter, ex_types=types,
                ex_count=count, n_techniques=len(EXPECTED), n_exchanges=len(TRUTH), n_defence=len(NEGATIVE))
