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
    kickoff       TEXT,                   -- UK local time 'HH:MM', when known (from fixtures.csv)
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
    referee               TEXT,
    home_xg               REAL,           -- football-data.co.uk's xG (2026/27 on);
                                          -- Understat's xG is in game_xg
    away_xg               REAL
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

-- Maps the team names other data sources use to our teams,
-- e.g. Understat's 'Manchester United' -> our 'Man United'. Filled from team_aliases.sql.
CREATE TABLE IF NOT EXISTS team_aliases (
    source       TEXT NOT NULL,           -- e.g. 'understat'
    league_id    INTEGER NOT NULL REFERENCES leagues(id),
    source_name  TEXT NOT NULL,           -- the name as that source writes it
    team_id      INTEGER NOT NULL REFERENCES teams(id),
    PRIMARY KEY (source, league_id, source_name)
);

-- Expected goals per game, kept per source because providers use different xG models.
CREATE TABLE IF NOT EXISTS game_xg (
    game_id          INTEGER NOT NULL REFERENCES games(id),
    source           TEXT NOT NULL,       -- e.g. 'understat'
    home_xg          REAL NOT NULL,
    away_xg          REAL NOT NULL,
    source_match_id  TEXT,                -- the source's own id for the match
    PRIMARY KEY (game_id, source)
);

-- Each prediction's expected goals. Every goals market (over/under, exact scores,
-- both teams to score) can be recalculated exactly from these two numbers.
CREATE TABLE IF NOT EXISTS prediction_goals (
    game_id        INTEGER NOT NULL REFERENCES games(id),
    model_name     TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    exp_home       REAL NOT NULL,
    exp_away       REAL NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (game_id, model_name, model_version)
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
