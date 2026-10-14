"""Experiments: which extra information improves the NFL points model?

Usage:
    python -m nfl.experiments      # about a minute

A self-contained version of the points model (nfl/model.py) where each idea can be switched on
separately: the older usual-starter flag, quarterback ratings, rest days (byes and short weeks),
travel across time zones, domes, and weather (wind and cold, outdoor games only). Each is scored
by walk-forward log loss on the tuning seasons (2010-2017); the winner is then checked once on
the held-out seasons (2018-2025).

Result (October 2026): only quarterback ratings helped, so they're the only addition in
nfl/model.py. Weather also can't be used for live predictions: nflverse records it only after
a game is played.
"""
import sqlite3
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import norm

from db import DB_PATH
from nfl.elo import load_nfl_games
from nfl.model import add_qb_changes

TZ = {"NE":-5,"NYJ":-5,"NYG":-5,"BUF":-5,"MIA":-5,"PHI":-5,"WAS":-5,"BAL":-5,"PIT":-5,"CLE":-5,"CIN":-5,"DET":-5,
      "IND":-5,"JAX":-5,"TB":-5,"ATL":-5,"CAR":-5,"CHI":-6,"GB":-6,"MIN":-6,"HOU":-6,"DAL":-6,"KC":-6,"NO":-6,"TEN":-6,
      "LA":-8,"LAC":-8,"SF":-8,"SEA":-8,"LV":-8,"DEN":-7,"ARI":-7}

def load(conn):
    g = add_qb_changes(load_nfl_games(conn))
    extra = pd.read_sql("SELECT game_id id, home_rest, away_rest, roof, temp, wind FROM nfl_game_info", conn)
    g = g.merge(extra, on="id").sort_values(["date","id"]).reset_index(drop=True)
    # career starts before each game, per QB (all history since 1999)
    starts = Counter(); hs, as_ = [], []
    for r in g.itertuples():
        for side, out in (("home", hs), ("away", as_)):
            q = getattr(r, f"{side}_qb"); q = q if isinstance(q, str) and q else None
            out.append(starts[q] if q else np.nan)
        if r.status == "final":
            for side in ("home","away"):
                q = getattr(r, f"{side}_qb")
                if isinstance(q, str) and q: starts[q] += 1
    g["home_qb_starts"], g["away_qb_starts"] = hs, as_
    for s in ("home","away"):
        rest = g[f"{s}_rest"].fillna(7)
        g[f"{s}_bye"] = (rest >= 13).astype(float); g[f"{s}_short"] = (rest <= 5).astype(float)
    neutral = g["neutral_site"] == 1
    g["away_tz"] = np.where(neutral, 0, (g["home"].map(TZ) - g["away"].map(TZ)).abs().fillna(0))
    g["dome"] = g["roof"].isin(["dome","closed"]).astype(float)
    outdoors = g["roof"].isin(["outdoors","open"])
    g["wind10"] = np.where(outdoors, (g["wind"].fillna(0) - 10).clip(lower=0), 0) / 10
    g["cold"] = np.where(outdoors & g["temp"].notna(), (40 - g["temp"].fillna(60)).clip(lower=0), 0) / 10
    return g

@dataclass
class Cfg:
    half_life: float = 150; shrink: float = 3.0; years: int = 4; sd: float = 13.0
    qb_flag: bool = True
    qb_ratings: bool = False; qb_shrink: float = 10.0; inexp_k: float = 16
    prev_weight: float = 1.0          # extra weight multiplier for earlier seasons' games
    rest: bool = False; travel: bool = False; dome: bool = False; weather: bool = False

def side_features(df, side, opp, cfg):
    """Feature columns affecting `side` team's points."""
    f = {}
    if cfg.qb_flag: f["qb_change"] = df[f"{side}_qb_change"].to_numpy(float)
    if cfg.qb_ratings:
        st = df[f"{side}_qb_starts"].to_numpy(float)
        f["qb_inexp"] = np.where(np.isnan(st), 0, np.clip(1 - st / cfg.inexp_k, 0, 1))
    if cfg.rest:
        f["own_bye"] = df[f"{side}_bye"].to_numpy(float); f["own_short"] = df[f"{side}_short"].to_numpy(float)
        f["opp_bye"] = df[f"{opp}_bye"].to_numpy(float); f["opp_short"] = df[f"{opp}_short"].to_numpy(float)
    if cfg.travel:
        f["travel_tz"] = df["away_tz"].to_numpy(float) * (1 if side == "away" else 0)
    if cfg.dome: f["dome"] = df["dome"].to_numpy(float)
    if cfg.weather: f["wind10"] = df["wind10"].to_numpy(float); f["cold"] = df["cold"].to_numpy(float)
    return f

def fit(train, as_of, cfg, season):
    teams = sorted(set(train["home"]) | set(train["away"])); ti = {t:i for i,t in enumerate(teams)}; n = len(teams)
    m = len(train); rows = np.arange(m)
    days = (pd.Timestamp(as_of) - pd.to_datetime(train["date"])).dt.days.to_numpy()
    w = 0.5 ** (days / cfg.half_life) * np.where(train["season"].to_numpy() < season, cfg.prev_weight, 1.0)
    w = np.tile(w, 2)
    fh = side_features(train, "home", "away", cfg); fa = side_features(train, "away", "home", cfg)
    names = list(fh)
    qbs = []
    if cfg.qb_ratings:
        qbs = sorted(set(train["home_qb"].dropna()) | set(train["away_qb"].dropna())); qi = {q:i for i,q in enumerate(qbs)}
    k = 2*n + 2 + len(names) + len(qbs)
    x = np.zeros((2*m, k))
    h = train["home"].map(ti).to_numpy(); a = train["away"].map(ti).to_numpy()
    hs = np.where(train["neutral_site"].to_numpy() == 1, 0.0, 0.5)
    x[rows, h] = 1; x[rows, n+a] = -1; x[rows, 2*n] = 1; x[rows, 2*n+1] = hs
    x[m+rows, a] = 1; x[m+rows, n+h] = -1; x[m+rows, 2*n] = 1; x[m+rows, 2*n+1] = -hs
    for j, nm in enumerate(names):
        x[rows, 2*n+2+j] = fh[nm]; x[m+rows, 2*n+2+j] = fa[nm]
    if qbs:
        off = 2*n+2+len(names)
        for side, base_rows in (("home", rows), ("away", m+rows)):
            col = train[f"{side}_qb"].map(qi).to_numpy()
            ok = ~pd.isna(col)
            x[base_rows[ok], off + col[ok].astype(int)] = 1
    y = np.concatenate([train["home_score"].to_numpy(float), train["away_score"].to_numpy(float)])
    pen = np.zeros(k); pen[:2*n] = cfg.shrink; pen[2*n+2:2*n+2+len(names)] = 1e-6
    if qbs: pen[2*n+2+len(names):] = cfg.qb_shrink
    xtw = x.T * w
    beta = np.linalg.solve(xtw @ x + np.diag(pen), xtw @ y)
    return {"teams": ti, "n": n, "beta": beta, "names": names, "qbs": {q:i for i,q in enumerate(qbs)}, "cfg": cfg}

def predict_rows(model, df):
    """Expected points for each game in df (home, away)."""
    cfg, b, n, ti = model["cfg"], model["beta"], model["n"], model["teams"]
    h = df["home"].map(ti).to_numpy(); a = df["away"].map(ti).to_numpy()
    hs = np.where(df["neutral_site"].to_numpy() == 1, 0.0, 0.5)
    off, de = b[:n], b[n:2*n]
    hp = b[2*n] + b[2*n+1]*hs + off[h] - de[a]; ap = b[2*n] - b[2*n+1]*hs + off[a] - de[h]
    fh = side_features(df, "home", "away", cfg); fa = side_features(df, "away", "home", cfg)
    for j, nm in enumerate(model["names"]):
        hp = hp + b[2*n+2+j]*fh[nm]; ap = ap + b[2*n+2+j]*fa[nm]
    if model["qbs"]:
        o = 2*n+2+len(model["names"])
        qv = lambda q: b[o + model["qbs"][q]] if isinstance(q, str) and q in model["qbs"] else 0.0
        hp = hp + df["home_qb"].map(qv).to_numpy(float); ap = ap + df["away_qb"].map(qv).to_numpy(float)
    return hp, ap

def backtest(g, start, end, cfg):
    played = g[g["status"] == "final"]
    test = played[played["season"].between(start, end)]
    out = []
    for (season, week), wk in test.groupby(["season","week"], sort=True):
        cutoff = wk["date"].min()
        st = (pd.Timestamp(cutoff) - pd.DateOffset(years=cfg.years)).strftime("%Y-%m-%d")
        train = played[(played["date"] >= st) & (played["date"] < cutoff)]
        model = fit(train, cutoff, cfg, season)
        wk = wk[wk["home"].isin(model["teams"]) & wk["away"].isin(model["teams"])]
        hp, ap = predict_rows(model, wk)
        out.append(pd.DataFrame({"id": wk["id"].to_numpy(), "hp": hp, "ap": ap}))
    return pd.concat(out, ignore_index=True)

def score(g, res, sd=None, sds=(11,12,13,14)):
    d = g.merge(res, on="id"); d = d[d["home_score"] != d["away_score"]]
    y = (d["home_score"] > d["away_score"]).to_numpy(float); mg = (d["hp"] - d["ap"]).to_numpy()
    def ll(s):
        p = norm.cdf(mg / s).clip(1e-9, 1-1e-9); return float(-(y*np.log(p) + (1-y)*np.log(1-p)).mean())
    if sd is not None: return ll(sd), sd
    best = min(sds, key=ll); return ll(best), best

def total_error(g, res):
    d = g.merge(res, on="id")
    return float(((d["home_score"] + d["away_score"]) - (d["hp"] + d["ap"])).abs().mean())


def main() -> None:
    games = load(sqlite3.connect(DB_PATH))
    ratings = Cfg(qb_ratings=True, qb_flag=False, qb_shrink=10, inexp_k=32)
    candidates = [
        ("Usual-starter flag (model version 1)", Cfg()),
        ("Quarterback ratings instead of the flag", ratings),
        ("Quarterback ratings + rest days", replace(ratings, rest=True)),
        ("Quarterback ratings + travel", replace(ratings, travel=True)),
        ("Quarterback ratings + domes", replace(ratings, dome=True)),
        ("Quarterback ratings + weather", replace(ratings, weather=True)),
        ("Quarterback ratings, last season's games counted less", replace(ratings, prev_weight=0.75)),
    ]
    print("Tuning seasons 2010-2017: log loss (lower is better) and average miss on total points")
    for label, cfg in candidates:
        result = backtest(games, 2010, 2017, cfg)
        loss, _ = score(games, result, sd=cfg.sd)
        print(f"  {label:<56}{loss:.4f}   {total_error(games, result):.2f}")
    print("\nHeld-out seasons 2018-2025 (checked once):")
    for label, cfg in candidates[:2]:
        loss, _ = score(games, backtest(games, 2018, 2025, cfg), sd=cfg.sd)
        print(f"  {label:<56}{loss:.4f}")


if __name__ == "__main__":
    main()
