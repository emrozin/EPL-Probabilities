"""NFL player projections: passing, rushing and receiving yards, touchdowns and receptions.

Usage:
    python -m nfl.import_nfl_stats     # once, then weekly (update.py does it)
    python -m nfl.players              # test the projections on past seasons

For each team in a game, built only from games played before it:

  1. Volume. The team's expected pass attempts and carries: its recent average, with recent games
     counting more. (Adjusting for the expected score, since trailing teams throw more, was tested
     and made the projections worse, so it's switched off.)
  2. Shares. Each player's share of the team's targets and carries in his recent games with the
     team. The expected starting quarterback throws the passes.
  3. Efficiency. Catch rate, yards per target, yards per carry, yards per pass attempt and
     touchdown and interception rates, each pulled toward the average for the player's position
     when he has few opportunities, and adjusted for how much the opponent's defense allows.
  4. Touchdowns are scaled by how many points the team model expects the team to score. The
     chance of scoring at least one touchdown comes from the expected number (a Poisson count).

Players who didn't record a stat in their team's latest game are left out (usually injured or
inactive): the data has no injury reports, a known limitation.
"""

import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from db import DB_PATH

POSITIONS = ("QB", "RB", "WR", "TE")
LEAGUE_DEFAULTS = {  # starting points for the running league averages (roughly 2010s NFL levels)
    "catch": 0.64, "yds_per_target": 7.4, "rec_td_per_target": 0.045,
    "yds_per_carry": 4.2, "rush_td_per_carry": 0.03,
    "yds_per_att": 6.9, "pass_td_per_att": 0.043, "int_per_att": 0.025,
    "pass_att": 34.0, "carries": 26.0, "points": 22.0,
}


@dataclass
class Settings:
    """Tuned on 2012-2017 (nfl/players.py main)."""
    decay: float = 0.9           # each older game counts this much as the next one, for shares and volume
    efficiency_decay: float = 0.95   # efficiency changes more slowly, so it remembers more games
    shrink_targets: float = 40   # pseudo-opportunities pulling a receiver's rates toward his position's
    shrink_carries: float = 60
    shrink_attempts: float = 150
    shrink_defense: float = 200  # pseudo-plays pulling a defense's allowed rates toward the league's
    script: float = 0.0          # extra pass attempts per point the team is expected to trail by:
                                 # tested at 0.1-0.5 and made projections worse, so it's off


SETTINGS = Settings()


def load(conn) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(games, team stats, player stats), with dates and teams, in date order."""
    games = pd.read_sql("""
        SELECT g.id, g.season, g.date, g.status, h.name AS home, a.name AS away, g.home_score, g.away_score,
               i.week, i.game_type, i.home_qb, i.away_qb
        FROM games g JOIN leagues l ON l.id = g.league_id AND l.code = 'NFL'
        JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
        JOIN nfl_game_info i ON i.game_id = g.id
        ORDER BY g.date, g.id""", conn)
    teams = pd.read_sql("SELECT * FROM nfl_team_stats", conn)
    players = pd.read_sql("SELECT * FROM nfl_player_stats", conn)
    return games, teams, players


class Ewma:
    """Exponentially weighted sums: each update first multiplies everything held so far by `decay`."""

    def __init__(self, decay: float):
        self.decay = decay
        self.sums: dict = defaultdict(float)

    def add(self, **values):
        for k in list(self.sums):
            self.sums[k] *= self.decay
        for k, v in values.items():
            self.sums[k] += v

    def get(self, key, default=0.0):
        return self.sums.get(key, default)


def rate(num: float, den: float, prior: float, strength: float) -> float:
    """A rate pulled toward `prior`, as if `strength` extra opportunities at the prior rate had been seen."""
    return (num + prior * strength) / (den + strength)


class Projector:
    """Plays through games in date order, keeping running team, defense and player numbers, and
    projects any game asked for using only games before its date."""

    def __init__(self, games, teams, players, settings: Settings = SETTINGS):
        self.s = settings
        self.games = games.set_index("id")
        self.team_stats = teams.set_index(["game_id", "team"])
        self.player_rows = {gid: grp for gid, grp in players.groupby("game_id")}
        self.volume = defaultdict(lambda: Ewma(settings.decay))        # team -> pass_att, carries, points, n
        self.allowed = defaultdict(lambda: Ewma(settings.efficiency_decay))   # team's defense
        self.usage = defaultdict(lambda: Ewma(settings.decay))         # (team, player) -> targets/carries shares
        self.skill = defaultdict(lambda: Ewma(settings.efficiency_decay))   # player -> efficiency numbers
        self.league = Ewma(0.995)
        self.last_game_players: dict[str, set] = {}
        self.roster: dict[str, set] = defaultdict(set)                # team -> every player seen with it
        self.info: dict[str, dict] = {}                                # player -> name, position

    # ---- learning from a finished game
    def update(self, game_id: int) -> None:
        g = self.games.loc[game_id]
        rows = self.player_rows.get(game_id)
        for team, opp, pts in ((g.home, g.away, g.home_score), (g.away, g.home, g.away_score)):
            if (game_id, team) not in self.team_stats.index:
                continue
            t = self.team_stats.loc[(game_id, team)]
            mine = rows[rows["team"] == team] if rows is not None else pd.DataFrame()
            targets = float(mine["targets"].fillna(0).sum()) if len(mine) else 0.0
            carries = float(t["carries"] or 0)
            pass_att = float(t["pass_attempts"] or 0)
            self.volume[team].add(pass_att=pass_att, carries=carries, points=float(pts), n=1.0)
            self.allowed[opp].add(pass_yds=float(t["passing_yards"] or 0), pass_att=pass_att,
                                  rush_yds=float(t["rushing_yards"] or 0), carries=carries)
            self.league.add(pass_yds=float(t["passing_yards"] or 0), pass_att=pass_att,
                            rush_yds=float(t["rushing_yards"] or 0), carries=carries)
            present = set()
            for p in mine.itertuples():
                present.add(p.player_id)
                self.info[p.player_id] = {"name": p.name, "position": p.position}
                v = lambda x: 0.0 if pd.isna(x) else float(x)
                self.usage[(team, p.player_id)].add(
                    target_share=v(p.targets) / targets if targets else 0.0,
                    carry_share=v(p.carries) / carries if carries else 0.0, n=1.0)
                self.skill[p.player_id].add(
                    targets=v(p.targets), receptions=v(p.receptions), rec_yds=v(p.receiving_yards),
                    rec_tds=v(p.receiving_tds), carries=v(p.carries), rush_yds=v(p.rushing_yards),
                    rush_tds=v(p.rushing_tds), att=v(p.pass_attempts), pass_yds=v(p.passing_yards),
                    pass_tds=v(p.passing_tds), ints=v(p.interceptions))
            # players who didn't appear this game: their usage decays as if they had 0 share
            for pid in self.roster[team] - present:
                self.usage[(team, pid)].add(n=0.0)
            self.roster[team] |= present
            self.last_game_players[team] = present

    # ---- projecting a game
    def defense_factor(self, opp: str, kind: str) -> float:
        a, lg, k = self.allowed[opp], self.league, self.s.shrink_defense
        if kind == "pass":
            league = lg.get("pass_yds") / max(lg.get("pass_att"), 1) or LEAGUE_DEFAULTS["yds_per_att"]
            allowed = rate(a.get("pass_yds"), a.get("pass_att"), league, k)
        else:
            league = lg.get("rush_yds") / max(lg.get("carries"), 1) or LEAGUE_DEFAULTS["yds_per_carry"]
            allowed = rate(a.get("rush_yds"), a.get("carries"), league, k)
        return allowed / league

    def position_rates(self) -> dict:
        """League-wide efficiency by position, from every player's running numbers (cached per date)."""
        totals = defaultdict(lambda: defaultdict(float))
        for pid, sk in self.skill.items():
            pos = self.info.get(pid, {}).get("position")
            if pos in POSITIONS:
                for k, v in sk.sums.items():
                    totals[pos][k] += v
        d = LEAGUE_DEFAULTS
        out = {}
        for pos in POSITIONS:
            t = totals[pos]
            out[pos] = {
                "catch": t["receptions"] / t["targets"] if t["targets"] > 50 else d["catch"],
                "yds_per_target": t["rec_yds"] / t["targets"] if t["targets"] > 50 else d["yds_per_target"],
                "rec_td_per_target": t["rec_tds"] / t["targets"] if t["targets"] > 50 else d["rec_td_per_target"],
                "yds_per_carry": t["rush_yds"] / t["carries"] if t["carries"] > 50 else d["yds_per_carry"],
                "rush_td_per_carry": t["rush_tds"] / t["carries"] if t["carries"] > 50 else d["rush_td_per_carry"],
                "yds_per_att": t["pass_yds"] / t["att"] if t["att"] > 50 else d["yds_per_att"],
                "pass_td_per_att": t["pass_tds"] / t["att"] if t["att"] > 50 else d["pass_td_per_att"],
                "int_per_att": t["ints"] / t["att"] if t["att"] > 50 else d["int_per_att"],
            }
        return out

    def project_team(self, team: str, opp: str, expected_points: float | None, expected_margin: float | None,
                     qb_name: str | None, rates: dict) -> list[dict]:
        s, vol = self.s, self.volume[team]
        n = vol.get("n")
        if n <= 0:
            return []
        pass_att = vol.get("pass_att") / n
        carries = vol.get("carries") / n
        avg_points = vol.get("points") / n or LEAGUE_DEFAULTS["points"]
        if expected_margin is not None:                      # game script: trailing teams throw more
            pass_att -= s.script * expected_margin
            carries += s.script * expected_margin * 0.6
        td_scale = float(np.clip((expected_points or avg_points) / avg_points, 0.6, 1.6))
        pass_factor, rush_factor = self.defense_factor(opp, "pass"), self.defense_factor(opp, "rush")

        candidates = set(self.last_game_players.get(team, set()))
        # The listed starting quarterback plays even if he sat out the team's latest game.
        listed = None
        if qb_name:
            named = [pid for pid, i in self.info.items() if i["name"] == qb_name and i["position"] == "QB"]
            listed = next((pid for pid in named if pid in self.roster[team]), named[0] if named else None)
            if listed is None:
                # A quarterback with no stats yet (a debut): projected at his position's average rates.
                listed = f"new:{team}:{qb_name}"
                self.info[listed] = {"name": qb_name, "position": "QB"}
            candidates.add(listed)
        usage = {pid: self.usage[(team, pid)] for pid in candidates}
        share = lambda pid, k: usage[pid].get(k) / max(usage[pid].get("n"), 1e-9) * min(1.0, usage[pid].get("n"))
        target_shares = {pid: share(pid, "target_share") for pid in candidates}
        carry_shares = {pid: share(pid, "carry_share") for pid in candidates}
        for shares in (target_shares, carry_shares):
            total = sum(shares.values())
            if total > 1:
                for pid in shares:
                    shares[pid] /= total

        # the expected starting quarterback throws the passes
        qbs = [pid for pid in candidates if self.info.get(pid, {}).get("position") == "QB"]
        starter = listed
        if starter is None and qbs:
            starter = max(qbs, key=lambda pid: self.skill[pid].get("att"))

        out = []
        for pid in candidates:
            pos = self.info.get(pid, {}).get("position")
            if pos not in POSITIONS:
                continue
            pr, sk = rates[pos], self.skill[pid]
            row = {"player_id": pid, "name": self.info[pid]["name"], "position": pos, "team": team,
                   "pass_att": 0.0, "pass_yds": 0.0, "pass_tds": 0.0, "ints": 0.0}
            if pid == starter:
                ypa = rate(sk.get("pass_yds"), sk.get("att"), pr["yds_per_att"], s.shrink_attempts)
                row.update(pass_att=pass_att, pass_yds=pass_att * ypa * pass_factor,
                           pass_tds=pass_att * rate(sk.get("pass_tds"), sk.get("att"), pr["pass_td_per_att"],
                                                    s.shrink_attempts) * td_scale,
                           ints=pass_att * rate(sk.get("ints"), sk.get("att"), pr["int_per_att"], s.shrink_attempts))
            elif pos == "QB":
                continue                                     # backup quarterbacks: not projected
            targets = target_shares.get(pid, 0.0) * pass_att
            car = carry_shares.get(pid, 0.0) * carries
            row.update(
                targets=targets,
                receptions=targets * rate(sk.get("receptions"), sk.get("targets"), pr["catch"], s.shrink_targets),
                rec_yds=targets * rate(sk.get("rec_yds"), sk.get("targets"), pr["yds_per_target"], s.shrink_targets)
                * pass_factor,
                rec_tds=targets * rate(sk.get("rec_tds"), sk.get("targets"), pr["rec_td_per_target"],
                                       s.shrink_targets) * td_scale,
                carries=car,
                rush_yds=car * rate(sk.get("rush_yds"), sk.get("carries"), pr["yds_per_carry"], s.shrink_carries)
                * rush_factor,
                rush_tds=car * rate(sk.get("rush_tds"), sk.get("carries"), pr["rush_td_per_carry"],
                                    s.shrink_carries) * td_scale)
            row["anytime_td"] = 1 - math.exp(-(row["rec_tds"] + row["rush_tds"]))
            if max(row["pass_att"], targets, car) >= 0.5:
                out.append(row)
        return out

    def run(self, targets: dict) -> pd.DataFrame:
        """Project the games in `targets` ({game id: (home points, away points) or None}), playing
        through every finished game before each one's date."""
        rows = []
        by_date = self.games.groupby("date")
        wanted = defaultdict(list)
        for gid in targets:
            wanted[self.games.at[gid, "date"]].append(gid)
        for day, group in by_date:
            if day in wanted:
                rates = self.position_rates()
                for gid in wanted[day]:
                    g = self.games.loc[gid]
                    pts = targets[gid]
                    for side, team, opp, qb in (("home", g.home, g.away, g.home_qb), ("away", g.away, g.home, g.away_qb)):
                        exp_pts = margin = None
                        if pts is not None:
                            exp_pts = pts[0] if side == "home" else pts[1]
                            margin = (pts[0] - pts[1]) * (1 if side == "home" else -1)
                        for r in self.project_team(team, opp, exp_pts, margin, qb if isinstance(qb, str) else None, rates):
                            rows.append({"game_id": gid, **r})
            for gid in group.index[group["status"] == "final"]:
                self.update(gid)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------- testing on past seasons

STATS = {  # projection column -> actual column
    "pass_yds": "passing_yards", "pass_tds": "passing_tds", "rush_yds": "rushing_yards",
    "receptions": "receptions", "rec_yds": "receiving_yards",
}


def evaluate(projections: pd.DataFrame, players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Average miss (mean absolute error) of the projections and of two simple benchmarks: each player's
    average in his previous 3 games, and his average so far that season (previous season if none)."""
    played = players.merge(games[["id", "season", "date"]], left_on="game_id", right_on="id")
    played = played.sort_values(["player_id", "date"])
    for proj, actual in STATS.items():
        g = played.groupby("player_id")[actual]
        played[f"last3_{proj}"] = g.transform(lambda s: s.fillna(0).shift(1).rolling(3, min_periods=1).mean())
        season_avg = played.groupby(["player_id", "season"])[actual].transform(
            lambda s: s.fillna(0).shift(1).expanding().mean())
        played[f"season_{proj}"] = season_avg.fillna(played[f"last3_{proj}"])
    projections = projections.rename(columns={c: f"proj_{c}" for c in projections.columns
                                              if c not in ("game_id", "player_id")})
    df = projections.merge(played, on=["game_id", "player_id"])
    rows = []
    for proj, actual in STATS.items():
        sub = df[df[f"proj_{proj}"] > 0.5] if proj in ("pass_yds", "pass_tds") else df
        sub = sub.dropna(subset=[f"last3_{proj}"])
        truth = sub[actual].fillna(0)
        rows.append({"stat": proj, "players": len(sub),
                     "model": (sub[f"proj_{proj}"] - truth).abs().mean(),
                     "last 3 games": (sub[f"last3_{proj}"] - truth).abs().mean(),
                     "season average": (sub[f"season_{proj}"] - truth).abs().mean()})
    td = df.dropna(subset=["proj_anytime_td"])
    scored = ((td["rushing_tds"].fillna(0) + td["receiving_tds"].fillna(0)) > 0).astype(float)
    p = td["proj_anytime_td"].clip(0.01, 0.99)
    rows.append({"stat": "anytime TD (log loss)", "players": len(td),
                 "model": float(-(scored * np.log(p) + (1 - scored) * np.log(1 - p)).mean()),
                 "last 3 games": np.nan, "season average": float(-(scored * np.log(scored.mean()) + (1 - scored)
                                                                   * np.log(1 - scored.mean())).mean())})
    return pd.DataFrame(rows)


def backtest_targets(conn, first: int, last: int) -> dict:
    """Finished games in these seasons, with the team model's backtest expected points."""
    from nfl.run_backtest import BACKTEST_NAME, BACKTEST_VERSION
    df = pd.read_sql("""SELECT g.id, pg.exp_home, pg.exp_away FROM games g
                        JOIN leagues l ON l.id = g.league_id AND l.code = 'NFL'
                        LEFT JOIN prediction_goals pg ON pg.game_id = g.id AND pg.model_name = ? AND pg.model_version = ?
                        WHERE g.status = 'final' AND g.season BETWEEN ? AND ?""",
                     conn, params=(BACKTEST_NAME, BACKTEST_VERSION, first, last))
    return {r.id: (None if pd.isna(r.exp_home) else (r.exp_home, r.exp_away)) for r in df.itertuples()}


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    games, teams, players = load(conn)
    if players.empty:
        raise SystemExit("No player stats found. Run python -m nfl.import_nfl_stats first.")
    for label, (first, last) in (("Tuning seasons 2012-2017", (2012, 2017)), ("Held-out seasons 2018-2025", (2018, 2025))):
        projections = Projector(games, teams, players).run(backtest_targets(conn, first, last))
        result = evaluate(projections, players, games)
        print(f"\n{label}: average miss per player-game (lower is better)")
        print(result.round(3).to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
