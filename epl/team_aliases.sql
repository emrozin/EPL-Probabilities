-- Team names that differ between data sources and our teams table
-- (which uses football-data.co.uk's names). Names that are identical in both
-- sources don't need an entry.
--
-- To add one: add a line to the VALUES list, then re-run import_understat.py.
-- The importer lists any source name it can't match, so you'll know what's missing.

WITH aliases (source, league_code, source_name, our_name) AS (
    VALUES
        ('understat', 'EPL', 'Manchester United',       'Man United'),
        ('understat', 'EPL', 'Manchester City',         'Man City'),
        ('understat', 'EPL', 'Newcastle United',        'Newcastle'),
        ('understat', 'EPL', 'Wolverhampton Wanderers', 'Wolves'),
        ('understat', 'EPL', 'Nottingham Forest',       'Nott''m Forest'),
        ('understat', 'EPL', 'West Bromwich Albion',    'West Brom'),
        ('understat', 'EPL', 'Queens Park Rangers',     'QPR')
)
INSERT OR REPLACE INTO team_aliases (source, league_id, source_name, team_id)
SELECT a.source, l.id, a.source_name, t.id
FROM aliases a
JOIN leagues l ON l.code = a.league_code
JOIN teams t   ON t.league_id = l.id AND t.name = a.our_name;
