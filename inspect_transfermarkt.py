"""Download the Transfermarkt tables needed for squad values, and show what's in them.

Usage:
    python inspect_transfermarkt.py

Source: transfermarkt-datasets (https://github.com/dcaribou/transfermarkt-datasets),
published under the CC0 license (free for any use). Its updates paused in mid-July 2026,
so it covers past seasons but not this season's squads.

Files are saved to data/transfermarkt/ (not committed to Git) and only downloaded once.
"""

from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = ["clubs", "competitions", "games", "player_valuations", "transfers", "appearances"]
OUT_DIR = Path("data/transfermarkt")


def download(table: str) -> Path:
    path = OUT_DIR / f"{table}.csv.gz"
    if path.exists():
        return path
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {table}...", flush=True)
    with requests.get(f"{BASE_URL}/{table}.csv.gz", stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return path


def describe(table: str, df: pd.DataFrame) -> None:
    size_mb = (OUT_DIR / f"{table}.csv.gz").stat().st_size / 1e6
    print(f"\n=== {table}: {len(df):,} rows, {size_mb:.1f} MB ===")
    print("columns:", ", ".join(df.columns))
    for col in [c for c in df.columns if "date" in c.lower()][:2]:
        dates = pd.to_datetime(df[col], errors="coerce")
        print(f"{col}: {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 30):
        print(df.head(3).to_string())


def main() -> None:
    frames = {}
    for table in TABLES:
        try:
            frames[table] = pd.read_csv(download(table), compression="gzip", low_memory=False)
        except requests.RequestException as err:
            print(f"Couldn't download {table}: {err}")
            continue
        describe(table, frames[table])

    # Premier League coverage: which competition id it uses, and how many games per season.
    if "competitions" in frames:
        comps = frames["competitions"]
        epl = comps[comps.astype(str).apply(lambda r: r.str.contains("premier-league|Premier League", case=False)).any(axis=1)]
        print("\n=== competitions matching 'Premier League' ===")
        print(epl.head(10).to_string())
    if "games" in frames and "competition_id" in frames["games"].columns:
        games = frames["games"]
        gb1 = games[games["competition_id"] == "GB1"]
        if "season" in gb1.columns:
            print("\n=== GB1 games per season ===")
            print(gb1.groupby("season").size().to_string())
    if "clubs" in frames:
        clubs = frames["clubs"]
        comp_col = next((c for c in clubs.columns if "competition" in c), None)
        if comp_col:
            print("\n=== clubs currently in GB1 ===")
            print(clubs[clubs[comp_col] == "GB1"].iloc[:, :6].to_string())


if __name__ == "__main__":
    main()
