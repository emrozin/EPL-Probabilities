-- NFL data checks and exploration. Run with:
--   sqlite3 data/sports.db
--   .mode box
--   .headers on
--   .read nfl/checks.sql
-- or copy one query at a time into the sqlite3 prompt.

-- 1. Games per season: played, still to play, and how many have betting prices.
SELECT g.season,
       COUNT(*)                                            AS games,
       SUM(g.status = 'final')                             AS played,
       SUM(g.status = 'scheduled')                         AS to_play,
       SUM(i.game_type <> 'REG')                           AS playoff_games,
       SUM(EXISTS (SELECT 1 FROM odds o WHERE o.game_id = g.id AND o.market = 'moneyline')) AS with_moneyline
FROM games g
JOIN leagues l ON l.id = g.league_id AND l.code = 'NFL'
JOIN nfl_game_info i ON i.game_id = g.id
GROUP BY g.season
ORDER BY g.season;

-- 2. Home advantage by era: home win rate (ties count as half), and average home margin.
SELECT CASE WHEN g.season < 2010 THEN '1999-2009'
            WHEN g.season < 2020 THEN '2010-2019'
            ELSE '2020-now' END                                         AS era,
       COUNT(*)                                                         AS games,
       ROUND(100.0 * AVG((g.home_score > g.away_score) + 0.5 * (g.home_score = g.away_score)), 1) AS home_win_pct,
       ROUND(AVG(g.home_score - g.away_score), 2)                       AS avg_home_margin
FROM games g
JOIN leagues l ON l.id = g.league_id AND l.code = 'NFL'
JOIN nfl_game_info i ON i.game_id = g.id
WHERE g.status = 'final' AND i.neutral_site = 0
GROUP BY era;

-- 3. How good is the spread? Average error should be near 0 if it's unbiased.
SELECT ROUND(AVG((g.home_score - g.away_score) - i.spread_line), 2)              AS avg_error,
       ROUND(AVG(ABS((g.home_score - g.away_score) - i.spread_line)), 2)         AS avg_miss,
       ROUND(100.0 * AVG((g.home_score - g.away_score) > i.spread_line), 1)      AS home_covered_pct,
       ROUND(100.0 * AVG((g.home_score - g.away_score) = i.spread_line), 1)      AS push_pct
FROM games g
JOIN nfl_game_info i ON i.game_id = g.id
WHERE g.status = 'final' AND i.spread_line IS NOT NULL;

-- 4. Most common final margins. Football scores come in 3s and 7s, so some margins are far more common.
SELECT ABS(g.home_score - g.away_score) AS margin,
       COUNT(*)                         AS games,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM games g2 JOIN leagues l2 ON l2.id = g2.league_id
                                 WHERE l2.code = 'NFL' AND g2.status = 'final'), 1) AS pct
FROM games g
JOIN leagues l ON l.id = g.league_id AND l.code = 'NFL'
WHERE g.status = 'final'
GROUP BY margin
ORDER BY games DESC
LIMIT 10;

-- 5. How often does the favorite win, by size of the spread?
SELECT CASE WHEN ABS(i.spread_line) < 3   THEN 'a: under 3'
            WHEN ABS(i.spread_line) < 7   THEN 'b: 3 to 6.5'
            WHEN ABS(i.spread_line) < 10  THEN 'c: 7 to 9.5'
            ELSE 'd: 10 or more' END                                           AS spread,
       COUNT(*)                                                                AS games,
       ROUND(100.0 * AVG(CASE WHEN i.spread_line > 0 THEN g.home_score > g.away_score
                              ELSE g.away_score > g.home_score END), 1)        AS favorite_won_pct
FROM games g
JOIN nfl_game_info i ON i.game_id = g.id
WHERE g.status = 'final' AND i.spread_line <> 0
GROUP BY spread
ORDER BY spread;

-- 6. This season so far: each team's record and points for/against.
WITH results AS (
    SELECT g.home_team_id AS team_id, g.home_score AS pf, g.away_score AS pa FROM games g
    JOIN leagues l ON l.id = g.league_id AND l.code = 'NFL'
    WHERE g.status = 'final' AND g.season = (SELECT MAX(season) FROM games g2 JOIN leagues l2
                                             ON l2.id = g2.league_id WHERE l2.code = 'NFL')
    UNION ALL
    SELECT g.away_team_id, g.away_score, g.home_score FROM games g
    JOIN leagues l ON l.id = g.league_id AND l.code = 'NFL'
    WHERE g.status = 'final' AND g.season = (SELECT MAX(season) FROM games g2 JOIN leagues l2
                                             ON l2.id = g2.league_id WHERE l2.code = 'NFL')
)
SELECT t.name AS team,
       SUM(pf > pa) AS w, SUM(pf < pa) AS l, SUM(pf = pa) AS t,
       SUM(pf) AS points_for, SUM(pa) AS points_against, SUM(pf - pa) AS diff
FROM results r
JOIN teams t ON t.id = r.team_id
GROUP BY t.name
ORDER BY w DESC, diff DESC;
