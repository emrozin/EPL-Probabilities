"""Save a snapshot of every Premier League player's availability (injuries, suspensions, doubts).

Usage:
    python fetch_availability.py

Source: the official Fantasy Premier League data feed, which lists each player's status
(available, doubtful, injured, suspended...) and chance of playing. It only ever shows the
current situation, so this script saves a dated copy each time it runs. Once enough weeks
of history have built up, we can test whether adjusting for missing players improves the
model. Until then, nothing here affects predictions.

Snapshots go to records/availability/YYYY-MM-DD.csv (one per day; a later run that day
replaces it). If the feed can't be reached, this prints a warning and exits normally, so
it never stops the weekly update.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
OUT_DIR = Path("records/availability")
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
STATUS = {"a": "available", "d": "doubtful", "i": "injured", "s": "suspended",
          "u": "unavailable", "n": "not eligible"}
COLUMNS = ["player_id", "name", "team", "position", "status", "chance_this_round", "chance_next_round",
           "news", "news_added", "minutes", "starts", "expected_goals", "expected_assists", "price"]


def snapshot(payload: dict) -> pd.DataFrame:
    """One row per player from the feed's JSON, keeping only what availability analysis needs."""
    teams = {t["id"]: t["name"] for t in payload.get("teams", [])}
    rows = []
    for p in payload.get("elements", []):
        rows.append({
            "player_id": p.get("id"),
            "name": f"{p.get('first_name', '')} {p.get('second_name', '')}".strip() or p.get("web_name"),
            "team": teams.get(p.get("team"), p.get("team")),
            "position": POSITIONS.get(p.get("element_type"), p.get("element_type")),
            "status": STATUS.get(p.get("status"), p.get("status")),
            "chance_this_round": p.get("chance_of_playing_this_round"),
            "chance_next_round": p.get("chance_of_playing_next_round"),
            "news": p.get("news") or "",
            "news_added": p.get("news_added"),
            "minutes": p.get("minutes"),
            "starts": p.get("starts"),
            "expected_goals": p.get("expected_goals"),
            "expected_assists": p.get("expected_assists"),
            "price": (p.get("now_cost") or 0) / 10 or None,
        })
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["team", "position", "name"])


def main() -> None:
    try:
        response = requests.get(URL, timeout=30, headers={"User-Agent": "Mozilla/5.0 (personal analytics project)"})
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as err:
        print(f"Couldn't fetch player availability ({err}). Skipping this snapshot; nothing else is affected.")
        return

    players = snapshot(payload)
    if players.empty:
        print("The availability feed had no players in it. Skipping this snapshot.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{datetime.now(timezone.utc):%Y-%m-%d}.csv"
    players.to_csv(path, index=False)
    out = players[players["status"] != "available"]
    print(f"Saved {len(players)} players to {path}: {len(out)} not fully available "
          f"({', '.join(f'{n} {s}' for s, n in out['status'].value_counts().items())}).")


if __name__ == "__main__":
    main()
