"""Pull a Sleeper league, compute playoff odds as of every finished week, write site/index.html.

Usage: python build.py
"""
import json
import urllib.request
from pathlib import Path

import numpy as np

LEAGUE_ID = "1389689970221907968"
API = "https://api.sleeper.app/v1"
SIMS = 20000
ROOT = Path(__file__).parent
SITE = ROOT / "site"
FLEX = {"RB", "WR", "TE"}


def get(path):
    with urllib.request.urlopen(f"{API}/{path}", timeout=60) as r:
        return json.load(r)


# ---- download ---------------------------------------------------------------
league = get(f"league/{LEAGUE_ID}")
users = {u["user_id"]: u for u in get(f"league/{LEAGUE_ID}/users")}
rosters = get(f"league/{LEAGUE_ID}/rosters")
players_db = get("players/nfl")

S = league["settings"]
last_week = S["playoff_week_start"] - 1          # final regular-season week
done_week = min(S.get("last_scored_leg", 0), last_week)   # last week Sleeper has scored into the standings
if done_week < 1:
    raise SystemExit("No finished weeks yet.")
playoff_teams = S["playoff_teams"]
byes = 2 if playoff_teams == 6 else 0             # Sleeper's default 6-team bracket gives seeds 1-2 a bye
slots = [p for p in league["roster_positions"] if p not in ("BN", "IR")]
matchups = {w: get(f"league/{LEAGUE_ID}/matchups/{w}") for w in range(1, last_week + 1)}


def pinfo(pid):
    p = players_db.get(pid)
    if p is None:          # team defenses are keyed by team code
        return {"name": f"{pid} D/ST", "pos": "DEF", "nfl": pid}
    return {"name": p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
            "pos": p.get("position"), "nfl": p.get("team") or "FA"}


def best_lineup(points):
    """Highest score the roster could have started: fixed slots first, then FLEX from what's left."""
    ranked = sorted(points.items(), key=lambda x: -x[1])
    used, total = set(), 0.0
    for slot in [s for s in slots if s != "FLEX"] + [s for s in slots if s == "FLEX"]:
        for pid, pts in ranked:
            pos = pinfo(pid)["pos"]
            if pid not in used and (pos == slot or (slot == "FLEX" and pos in FLEX)):
                used.add(pid)
                total += pts
                break
    return total


# ---- teams and weekly data ---------------------------------------------------
teams = {}
for r in rosters:
    u = users[r["owner_id"]]
    teams[r["roster_id"]] = {"id": r["roster_id"],
                             "name": (u.get("metadata") or {}).get("team_name") or u["display_name"],
                             "owner": u["display_name"]}
ids = sorted(teams)
idx = {t: i for i, t in enumerate(ids)}
N = len(ids)

schedule = {}                               # week -> [(a, b)]
scores = np.zeros((done_week, N))
maxpts = np.zeros((done_week, N))
games = []
player_rows = {}
for w in range(1, last_week + 1):
    pairs = {}
    for e in matchups[w]:
        pairs.setdefault(e["matchup_id"], []).append(e)
    schedule[w] = []
    for _, (a, b) in sorted(pairs.items()):
        schedule[w].append((a["roster_id"], b["roster_id"]))
        if w > done_week:
            continue
        for e in (a, b):
            i = idx[e["roster_id"]]
            scores[w - 1, i] = e["points"]
            maxpts[w - 1, i] = best_lineup(e["players_points"])
            starters = set(e["starters"])
            for pid, pts in e["players_points"].items():
                row = player_rows.setdefault((e["roster_id"], pid), {"roster": e["roster_id"], **pinfo(pid), "weeks": {}})
                row["weeks"][w] = [pts, pid in starters]
        games.append({"week": w, "a": a["roster_id"], "b": b["roster_id"], "a_pts": a["points"], "b_pts": b["points"]})


def records(k):
    """Wins (ties count half), losses, ties, points for, points against after week k."""
    rec = {t: {"w": 0, "l": 0, "t": 0, "pf": 0.0, "pa": 0.0} for t in ids}
    for g in games:
        if g["week"] > k:
            continue
        A, B = rec[g["a"]], rec[g["b"]]
        A["pf"] += g["a_pts"]; A["pa"] += g["b_pts"]
        B["pf"] += g["b_pts"]; B["pa"] += g["a_pts"]
        if g["a_pts"] == g["b_pts"]:
            A["t"] += 1; B["t"] += 1
        elif g["a_pts"] > g["b_pts"]:
            A["w"] += 1; B["l"] += 1
        else:
            A["l"] += 1; B["w"] += 1
    return rec


# ---- check our totals against Sleeper's standings ---------------------------
latest = records(done_week)
mismatch = []
for r in rosters:
    s, mine = r["settings"], latest[r["roster_id"]]
    site = {"w": s["wins"], "l": s["losses"], "t": s.get("ties", 0),
            "pf": s["fpts"] + s.get("fpts_decimal", 0) / 100,
            "pa": s.get("fpts_against", 0) + s.get("fpts_against_decimal", 0) / 100,
            "max": s.get("ppts", 0) + s.get("ppts_decimal", 0) / 100}
    mine_max = maxpts[:, idx[r["roster_id"]]].sum()
    for key in ("w", "l", "t"):
        if mine[key] != site[key]:
            mismatch.append(f"{teams[r['roster_id']]['name']} {key}: {mine[key]} vs Sleeper {site[key]}")
    for key, val in (("pf", mine["pf"]), ("pa", mine["pa"]), ("max", mine_max)):
        if abs(val - site[key]) > 0.011:
            mismatch.append(f"{teams[r['roster_id']]['name']} {key}: {val:.2f} vs Sleeper {site[key]:.2f}")
if mismatch:
    raise SystemExit("Totals do not match Sleeper standings:\n" + "\n".join(mismatch))

# ---- noise model, estimated from every finished week --------------------------
# Weekly scores swing far more than teams differ, so a team's average is pulled
# toward the league average. The two spreads come from all finished weeks and
# are reused for every snapshot (week 1 alone can't separate them).
league_mean = scores.mean()
within_var = scores.var(axis=0, ddof=1).mean() if done_week > 1 else scores.var(ddof=1)
between_var = max(scores.mean(axis=0).var(ddof=1) - within_var / done_week, 1.0)
within_sd = np.sqrt(within_var)


def snapshot(k, rng):
    """Every derived metric as it stood after week k."""
    rec = records(k)
    sc = scores[:k]
    team_mean = sc.mean(axis=0)
    shrink = between_var / (between_var + within_var / k)
    post_mean = league_mean + shrink * (team_mean - league_mean)
    post_sd = np.sqrt(1 / (1 / between_var + k / within_var))

    w0 = np.array([rec[t]["w"] + 0.5 * rec[t]["t"] for t in ids])
    pf0 = np.array([rec[t]["pf"] for t in ids])
    true = rng.normal(post_mean, post_sd, size=(SIMS, N))
    wins = np.repeat(w0[None, :], SIMS, axis=0)
    pf = np.repeat(pf0[None, :], SIMS, axis=0)
    for w in range(k + 1, last_week + 1):
        wk = true + rng.normal(0, within_sd, size=(SIMS, N))
        pf += wk
        for a, b in schedule[w]:
            ia, ib = idx[a], idx[b]
            wins[:, ia] += wk[:, ia] > wk[:, ib]
            wins[:, ib] += wk[:, ib] > wk[:, ia]
    order = np.argsort(-(wins * 1e5 + pf), axis=1)          # seed by wins, then points for
    seed = np.empty_like(order)
    seed[np.arange(SIMS)[:, None], order] = np.arange(1, N + 1)[None, :]

    out = {"shrink": float(shrink), "post_sd": float(post_sd), "teams": {}}
    for t in ids:
        i = idx[t]
        ap_w = int((sc[:, [i]] > sc).sum())
        ap_t = int((sc[:, [i]] == sc).sum()) - k
        ap_l = k * (N - 1) - ap_w - ap_t
        ap_pct = (ap_w + 0.5 * ap_t) / (k * (N - 1))
        future = [post_mean[idx[b if a == t else a]] for w in range(k + 1, last_week + 1)
                  for a, b in schedule[w] if t in (a, b)]
        out["teams"][t] = {
            **rec[t],
            "ap_w": ap_w, "ap_l": ap_l, "exp_wins": ap_pct * k,
            "luck": rec[t]["w"] + 0.5 * rec[t]["t"] - ap_pct * k,
            "strength": float(post_mean[i]),
            "sos_future": float(np.mean(future)) if future else None,
            "playoff_pct": float((seed[:, i] <= playoff_teams).mean()),
            "bye_pct": float((seed[:, i] <= byes).mean()),
            "avg_seed": float(seed[:, i].mean()),
            "seed_dist": [float((seed[:, i] == s).mean()) for s in range(1, N + 1)],
            "proj_w": float(wins[:, i].mean()),
            "wins_dist": {str(n): float((np.round(wins[:, i]) == n).mean()) for n in range(last_week + 1)},
        }
    return out


rng = np.random.default_rng(2026)
snapshots = {k: snapshot(k, rng) for k in range(1, done_week + 1)}

out = {
    "league": {"name": league["name"], "season": league["season"], "teams": N,
               "playoff_teams": playoff_teams, "byes": byes, "done_week": done_week,
               "last_week": last_week, "sims": SIMS},
    "model": {"league_mean": float(league_mean), "within_sd": float(within_sd),
              "between_sd": float(np.sqrt(between_var))},
    "teams": [teams[t] for t in ids],
    "scores": scores.round(2).tolist(),
    "maxpts": maxpts.round(2).tolist(),
    "games": games,
    "schedule": schedule,
    "players": list(player_rows.values()),
    "snapshots": snapshots,
}
payload = json.dumps(out, separators=(",", ":"))
SITE.mkdir(exist_ok=True)
template = (ROOT / "template.html").read_text(encoding="utf-8")
(SITE / "index.html").write_text(template.replace("__DATA__", payload.replace("</", "<\\/")), encoding="utf-8")

print(f"verified: records, points for, points against and max points match Sleeper for all {N} teams")
print(f"weeks 1-{done_week} built; page {len(payload) // 1024} KB of data")
cur = snapshots[done_week]["teams"]
for t in sorted(ids, key=lambda t: -cur[t]["playoff_pct"]):
    c = cur[t]
    print(f"  {teams[t]['name'][:28]:28} {c['w']}-{c['l']}  playoffs {c['playoff_pct']*100:5.1f}%  "
          + " ".join(f"{snapshots[k]['teams'][t]['playoff_pct']*100:4.0f}" for k in snapshots))
