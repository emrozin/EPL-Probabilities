-- Sanity checks and first exploration. Run against data/sports.db.

-- 1. Games per season. Expect 462 for 1993/94 and 1994/95 (22 teams), 380 after.
SELECT season, COUNT(*) AS games
FROM games
GROUP BY season
ORDER BY season;

-- 2. What data exists for each season? (Shows when stats and each odds market begin.)
SELECT g.season,
       COUNT(DISTINCT g.id)                                  AS games,
       COUNT(DISTINCT s.game_id)                             AS with_stats,
       COUNT(DISTINCT CASE WHEN o.market = '1x2' THEN o.bookmaker END)            AS bookmakers_1x2,
       MAX(o.market = 'total')                               AS has_totals,
       MAX(o.market = 'asian_handicap')                      AS has_asian_handicap,
       MAX(o.bookmaker = 'Pinnacle')                         AS has_pinnacle,
       MAX(o.is_closing)                                     AS has_closing_odds
FROM games g
LEFT JOIN soccer_match_stats s ON s.game_id = g.id
LEFT JOIN odds o               ON o.game_id = g.id
GROUP BY g.season
ORDER BY g.season;

-- 3. Result split and scoring by season. How big is home advantage?
SELECT season,
       ROUND(AVG(home_score > away_score), 3) AS home_win,
       ROUND(AVG(home_score = away_score), 3) AS draw,
       ROUND(AVG(home_score < away_score), 3) AS away_win,
       ROUND(AVG(home_score + away_score), 2) AS goals_per_game
FROM games
WHERE status = 'final'
GROUP BY season
ORDER BY season;

-- 4. Bookmaker margin: implied probabilities (1 / price) sum to more than 1.
--    The excess is the bookmaker's built-in edge.
SELECT bookmaker,
       CASE is_closing WHEN 1 THEN 'closing' ELSE 'pre-match' END AS snapshot,
       COUNT(*) AS games,
       ROUND(AVG(margin) * 100, 2) AS avg_margin_pct
FROM (
    SELECT game_id, bookmaker, is_closing, SUM(1.0 / price) - 1 AS margin
    FROM odds
    WHERE market = '1x2'
    GROUP BY game_id, bookmaker, is_closing
    HAVING COUNT(*) = 3
)
GROUP BY bookmaker, is_closing
ORDER BY avg_margin_pct;

-- 5. Spot-check one team's season against your own memory.
SELECT g.date, h.name AS home, a.name AS away, g.home_score, g.away_score,
       s.home_shots, s.away_shots, s.referee
FROM games g
JOIN teams h ON h.id = g.home_team_id
JOIN teams a ON a.id = g.away_team_id
LEFT JOIN soccer_match_stats s ON s.game_id = g.id
WHERE g.season = 2024
  AND 'Arsenal' IN (h.name, a.name)
ORDER BY g.date;

-- 6. Any original CSV column is still available from the raw JSON,
--    e.g. hit-woodwork or offside counts if a season has them.
SELECT g.date, json_extract(r.data, '$.HomeTeam') AS home, json_extract(r.data, '$.AwayTeam') AS away,
       r.data
FROM raw_source_rows r
JOIN games g ON g.id = r.game_id
WHERE g.season = 2024
LIMIT 3;
