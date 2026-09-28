-- Sports probabilities database (SQLite).
-- Designed so other leagues (La Liga, NFL, ...) fit without schema changes.
-- Every statement uses IF NOT EXISTS, so running this file twice is harmless.

CREATE TABLE IF NOT EXISTS leagues (
    id     INTEGER PRIMARY KEY,
    code   TEXT NOT NULL UNIQUE,          -- 'EPL' now; later 'LALIGA', 'NFL', ...
    name   TEXT NOT NULL,
    sport  TEXT NOT NULL                  -- 'soccer', 'american_football', ...
);

CREATE TABLE IF NOT EXISTS teams (
    id         INTEGER PRIMARY KEY,
    league_id  INTEGER NOT NULL REFERENCES leagues(id),
    name       TEXT NOT NULL,
    UNIQUE (league_id, name)
);

CREATE TABLE IF NOT EXISTS games (
    id            INTEGER PRIMARY KEY,
    league_id     INTEGER NOT NULL REFERENCES leagues(id),
    season        INTEGER NOT NULL,       -- start year: 2024 means 2024/25
    date          TEXT NOT NULL,          -- ISO format 'YYYY-MM-DD'
    home_team_id  INTEGER NOT NULL REFERENCES teams(id),
    away_team_id  INTEGER NOT NULL REFERENCES teams(id),
    home_score    INTEGER,                -- NULL until the game is played
    away_score    INTEGER,
    status        TEXT NOT NULL DEFAULT 'scheduled'
                  CHECK (status IN ('scheduled', 'final')),
    UNIQUE (league_id, season, date, home_team_id, away_team_id)
);

-- Soccer-specific match stats. Other sports get their own stats table later.
-- Any column is NULL when the source doesn't have it for that season.
CREATE TABLE IF NOT EXISTS soccer_match_stats (
    game_id               INTEGER PRIMARY KEY REFERENCES games(id),
    ht_home_score         INTEGER,
    ht_away_score         INTEGER,
    home_shots            INTEGER,
    away_shots            INTEGER,
    home_shots_on_target  INTEGER,
    away_shots_on_target  INTEGER,
    home_fouls            INTEGER,
    away_fouls            INTEGER,
    home_corners          INTEGER,
    away_corners          INTEGER,
    home_yellows          INTEGER,
    away_yellows          INTEGER,
    home_reds             INTEGER,
    away_reds             INTEGER,
    referee               TEXT
);

CREATE TABLE IF NOT EXISTS odds (
    id          INTEGER PRIMARY KEY,
    game_id     INTEGER NOT NULL REFERENCES games(id),
    bookmaker   TEXT NOT NULL,            -- 'Pinnacle', 'Bet365', 'Market average', ...
    market      TEXT NOT NULL,            -- '1x2', 'total', 'asian_handicap' (NFL: 'spread')
    outcome     TEXT NOT NULL,            -- 'home'/'draw'/'away', 'over'/'under'
    price       REAL NOT NULL,            -- decimal odds, e.g. 2.50
    line        REAL,                     -- NULL for 1x2; 2.5 for totals;
                                          -- asian_handicap: the HOME team's handicap on both rows
    is_closing  INTEGER NOT NULL CHECK (is_closing IN (0, 1))
);

-- NULLs never count as equal in a UNIQUE constraint, so a plain
-- UNIQUE(..., line, ...) would let duplicate 1x2 rows in.
-- Indexing COALESCE(line, 0) instead makes re-imports skip existing rows.
CREATE UNIQUE INDEX IF NOT EXISTS odds_unique
    ON odds (game_id, bookmaker, market, outcome, COALESCE(line, 0), is_closing);

-- Every original CSV row, stored as JSON, so no column from the source is ever lost.
-- Query any field with json_extract, e.g. json_extract(data, '$.HS').
CREATE TABLE IF NOT EXISTS raw_source_rows (
    game_id  INTEGER PRIMARY KEY REFERENCES games(id),
    source   TEXT NOT NULL,               -- e.g. 'football-data.co.uk E0 1993'
    data     TEXT NOT NULL                -- JSON object of the original row
);

CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY,
    game_id        INTEGER NOT NULL REFERENCES games(id),
    model_name     TEXT NOT NULL,         -- e.g. 'poisson'
    model_version  TEXT NOT NULL,         -- e.g. 'v1', so model upgrades can be compared
    outcome        TEXT NOT NULL,
    probability    REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (game_id, model_name, model_version, outcome)
);
