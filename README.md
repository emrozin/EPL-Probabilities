# Sports Probabilities: Premier League and NFL

Statistical models that estimate win probabilities for English Premier League matches and NFL games, evaluated honestly against the sharpest benchmark available: betting-market odds. Predictions for upcoming games are published to this repository before kickoff, so each model's live track record can be checked by anyone. Both sports share one database, one automated update and one website, with a sport menu.

The Premier League model came first and is described in most detail below; the NFL section follows.

The project covers the full pipeline: collecting and validating 30+ seasons of data from four sources, building and tuning a Poisson goals model with expected goals (xG), testing it with walk-forward backtesting on seasons it never saw during development, and serving it through a web app with an automated, tested weekly update.

**Tech:** Python, SQL (SQLite), pandas, NumPy, SciPy, FastAPI, Jinja, pytest, GitHub Actions

## Premier League results

All numbers below are from **held-out seasons (2021/22 to 2025/26, 1,921 matches)** that played no part in choosing the model's settings. Lower log loss is better; guessing one-third for every outcome scores 1.0986.

| Model | Log loss |
|---|---|
| Goals only, initial settings | 0.9824 |
| Goals only, tuned | 0.9813 |
| + expected goals (xG) blend | 0.9778 |
| + xG with re-tuned recency weighting (**current model**) | **0.9754** |
| Closing market (Pinnacle / Betfair Exchange, margin removed) | 0.9592 |

Adding xG and re-tuning closed **27% of the gap** between the goals-only model and the closing market. Put another way, the current model gives the actual result a typical probability of **37.7%**, against the market's **38.3%**.

The model does not beat the market, and isn't expected to: closing odds reflect injuries, confirmed lineups and professional bettors' information, while this model only sees past results. The goal is to measure how close a transparent, results-based model can get, and which changes actually help.

### Live track record

Since October 2026, every prediction for an upcoming match is committed to [`records/live_predictions.csv`](records/live_predictions.csv) before kickoff by an automated job, and never changed afterwards; the Git history shows when each one was made. The website's Track record page scores these predictions, and only these, once the matches are played. A few rounds is far too small a sample to judge the model by, so the backtest above is the main evidence until the record grows.

## Data

| Source | Coverage | Contents |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/) | 1993/94 onward | Results, half-time scores, match stats (from 2000/01), 1X2 / over-under / Asian handicap odds from 20+ bookmakers, pre-match and closing; upcoming fixtures |
| [Understat](https://understat.com/) | 2014/15 onward | Expected goals (xG) per team per match; results before football-data.co.uk posts them |
| [Fantasy Premier League](https://fantasy.premierleague.com/) | 2016/17 onward | Player prices (via the [FPL Historical Dataset](https://github.com/vaastav/Fantasy-Premier-League)), current injuries and availability |
| [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | 2013/14 to June 2026 | Player market valuations, used in the squad-value experiments |

The database holds 13,000+ Premier League matches. Some of the data-quality work involved:

- **Inconsistent file formats across 30+ seasons.** Column names change over time (e.g. `BbAv` became `Avg`), and some seasons have rows with more fields than the header. The importer handles both, and every original row is also stored as JSON so no source column is ever lost.
- **Unmapped column report.** Each import lists any source columns not mapped to a table, which is how new bookmakers and a new xG column were discovered.
- **Cross-source validation.** Each source spells team names differently (e.g. "Manchester United" vs "Man United"), resolved through an alias table. All 4,610 matches linked between Understat and football-data.co.uk were checked against both sources' final scores, with 100% agreement.
- **Source changes and delays.** Midway through development, Understat stopped embedding data in its pages; the importer failed loudly rather than importing nothing, and the fix was isolated to one function. football-data.co.uk can take days to post results, so results are filled in from Understat in the meantime and confirmed when football-data.co.uk catches up, without creating duplicate matches.
- **No look-ahead.** Squad values for any week are built only from information available by that date, so a backtest never benefits from knowing, for example, a player's later transfer.

## Model

Each team has an **attack** and a **defense** rating, plus a league-wide **home advantage**. A team's expected goals in a match are:

```
expected goals = base rate × home advantage (if at home) × attack ÷ opponent's defense
```

Each team's goals are modeled as a Poisson distribution around its expected goals, which gives the probability of every exact scoreline. From those come home/draw/away probabilities, over/under totals, both teams to score, and the likeliest scores.

Ratings are fitted by **maximum likelihood** (SciPy's L-BFGS-B with an analytic gradient, about 30× faster than numerical differentiation) using the last three years of matches, with:

- **Recency weighting:** a match's weight halves every 180 days.
- **Shrinkage (L2 regularization):** ratings are pulled toward average, so teams with few matches, such as newly promoted sides, don't get extreme ratings from a handful of results.
- **xG blend:** ratings are learned from a 50/50 blend of actual goals and xG.

A **season simulator** plays out the rest of the season 10,000 times with the model to estimate each team's chances of winning the title, finishing in the top four, or being relegated.

## Evaluation

- **Walk-forward backtest:** for every week since 2016/17, the model is trained only on matches played before that week, then predicts that week's games. It never sees a result it is predicting.
- **Validation / test split:** all settings were tuned on 2016/17–2020/21, and final results are reported on 2021/22–2025/26.
- **Metrics:** log loss and Brier score, compared against closing odds from Pinnacle (Betfair Exchange where Pinnacle is unavailable), with the bookmaker margin removed.

## Findings

- **The gap to the market is spread evenly.** An error analysis split the backtest by team, stage of the season, favorite strength and outcome. The model is well calibrated overall (slightly too cautious on strong favorites), and no single group accounts for the shortfall. That pattern points to missing information, mainly team news, rather than a flaw in the math.
- **Squad value helps early in the season, and only then.** See below.
- **Promoted teams aren't a weak spot.** A dedicated check found the model's error on newly promoted teams' matches only slightly worse than usual, so importing second-division data to rate them wasn't worth the added complexity.
- **Home advantage is shrinking.** Home teams won around 45–50% of matches in the 1990s and 2000s, and closer to 41–48% recently.
- **Crowds matter.** In 2020/21, played without fans, away teams won more often than home teams (40.3% vs 37.9%), the only season where that happened.
- **Scoring jumped in 2023/24** to 3.28 goals per game, the highest on record, coinciding with a crackdown on time-wasting that added stoppage time. The goals-only model was slow to adapt that season.
- **Goals and xG work best together.** On validation seasons, goals only scored 0.9571, a 50/50 blend 0.9532, and pure xG 0.9552. Actual goals carry information xG misses, such as finishing and goalkeeping quality.
- **xG allows a shorter memory.** With xG, the best recency half-life dropped from 365 to 180 days, and less shrinkage was needed.
- **Dixon–Coles didn't help.** This well-known correction for low-scoring draws made predictions slightly worse: the model already predicted slightly more draws than occurred (23.7% vs 22.5%). The apparent excess of low scores in the raw data came from pooling strong and weak teams, which team-specific ratings already account for.

### Squad value

Pulling ratings toward each squad's value (FPL prices or Transfermarkt valuations) improved early-season predictions but made later ones slightly worse, since by then results already show how good a team is. Making its influence fade as the season goes on (halving every 12 rounds) kept nearly all of the early gain. Log loss on the 1,873 held-out matches with squad values available:

| | Current model | Squad value, full strength | Squad value, fading | Market |
|---|---|---|---|---|
| All matches | 0.9735 | 0.9737 | **0.9718** | 0.9570 |
| Rounds 1–10 | 0.9558 | **0.9492** | 0.9500 | 0.9473 |
| Rounds 11–38 | 0.9800 | 0.9826 | **0.9797** | 0.9605 |

The fading version closes 11% of the overall gap to the market, and about two-thirds of the gap in rounds 1–10. One caveat: on the tuning seasons the full-strength version scored slightly better, so the held-out seasons decided between two finalists. The fading version is preferred because, by design, it can't cause the late-season loss. It isn't used for live predictions yet.

## NFL

Data: every game since 1999 from [nflverse](https://github.com/nflverse/nfldata) (scores, spreads, totals, moneylines from 2006, starting quarterbacks, rest days, weather), plus weekly team and player stats with expected points added (EPA) from [nflverse-data](https://github.com/nflverse/nflverse-data): about 126,000 player-games, in the same database.

**Model.** Each team has an offense rating (points scored above average) and a defense rating (points prevented), plus a home advantage fitted from recent games, so it follows the league's shrinking home edge (about 2.7 points in the 2000s, under 2 since 2020). Ratings are fitted by weighted least squares on four years of games, with a 150-day half-life and shrinkage. The expected margin becomes a win probability through a normal distribution (actual margins land about 12 points either side). **Every starting quarterback has his own rating**, fitted alongside the team ratings and pulled toward average, plus an inexperience term for quarterbacks with few career starts; live predictions use nflverse's expected starters for the current week.

**Results** on held-out seasons (2018-2025, 2,219 games, ties left out; tuned on 2010-2017), log loss:

| Model | Log loss | Share of the market's edge over a coin flip |
|---|---|---|
| Coin flip | 0.6931 | 0% |
| Elo ratings (baseline) | 0.6382 | 65% |
| Points model (offense/defense ratings) | 0.6374 | 66% |
| Points model + usual-starter quarterback flag | 0.6326 | 72% |
| **Points model + individual quarterback ratings** | **0.6298** | **75%** |
| Market moneylines | 0.6089 | 100% |

Offense and defense ratings alone did no better than Elo on the held-out seasons, despite looking better on the tuning seasons. Quarterbacks made the real difference. A simple flag for teams not starting their usual quarterback came first (the model had been overrating those teams by about two points); individual quarterback ratings then replaced it, helping most in the first four weeks of a season, when teams have new starters (held-out log loss 0.6493 to 0.6458 in weeks 1-4).

**What didn't help** (`nfl/experiments.py`): rest days (byes, short weeks), travel across time zones, domes, weather, and counting last season's games less all scored the same or worse on the tuning seasons, so none are in the model. Weather also can't be used live, since nflverse records it only after a game. Learning from **EPA** as well as the score, the NFL's equivalent of the xG blend that helped the Premier League model most, made win predictions worse at every weight tried (it slightly improved total points): once quarterbacks are rated, the score already carries most of the signal.

**Player projections.** Passing, rushing and receiving yards, touchdowns and receptions for every player in upcoming games, from (1) the team's recent passing and running volume, (2) each player's recent share of targets and carries, nudged by his latest snap count (the listed starting quarterback throws the passes), (3) the weekly injury report (Out and Doubtful players aren't projected, Questionable players get a slightly smaller role, players back from injury return), (4) his efficiency pulled toward his position's average, adjusted for the opposing defense, and (5) touchdowns scaled by the team model's expected points. Each projection comes with an 80% range. Tuned on 2012-2017; average miss per player-game on the held-out 2018-2025 seasons (about 35,000 player-games):

| | Projection | Last 3 games | Season average |
|---|---|---|---|
| Passing yards | **61.4** | 71.6 | 70.0 |
| Passing touchdowns | **0.92** | 1.00 | 0.98 |
| Rushing yards | **8.2** | 8.9 | 8.7 |
| Receptions | **1.28** | 1.38 | 1.34 |
| Receiving yards | **17.0** | 18.6 | 18.0 |

Each addition was tested with a measure that counts every player who was projected or played (`nfl/player_experiments.py`), so a version can't look better by projecting fewer players. Injury reports cut misses by 4-5% for rushing yards, receptions and receiving yards on the held-out seasons, and snap counts by about 1% more. Not kept: redistributing an injured player's share to teammates, defense by position, air yards (depth of target), and adjusting passing volume for the expected score. The first version of the injury rules looked better than it was because it was scored on a different set of players, and the fair comparison caught it: players who drop off the injury report are often on injured reserve, not returning. The 80% ranges held up on the held-out seasons (79-86% of results inside). There are no free historical betting lines for player stats, so these are compared with simple benchmarks rather than the market. Projections are saved before kickoff to `records/nfl_player_projections.csv`, and the website's track record compares them with what happened and with the benchmarks.

## Website

A FastAPI app with Jinja templates, currently run locally. A menu in the header switches between the sports. Premier League pages:

- **Upcoming:** next fixtures with model probabilities next to the bookmakers' odds
- **Match pages:** exact-score grid, over/under (with the market's view), both teams to score
- **Results** by week and **league tables** for every season since 1993/94, with team pages
- **Forecast:** season simulation with title, top-four and relegation chances
- **Track record:** live predictions only, scored against results and the closing market
- **Backtest** and **Ratings:** held-out performance and current team ratings

NFL pages (under `/nfl/`): upcoming games with win chances, expected margin and total next to the market's moneyline, spread and total; results by week; standings by division; team pages with schedule and quarterbacks; game pages with expected starters, form and every earlier meeting; track record; backtest with calibration; team ratings.

`build_site.py` can also render the whole site as static HTML for GitHub Pages.

## Automation and testing

- **One command updates everything:** `update.py` runs the results, xG, backtest, fixtures, prediction and availability steps in order, stopping at the first failure.
- **GitHub Actions** runs the test suite on every push, and twice a week (Tuesdays and Fridays) runs the update for both sports and commits new predictions to `records/` before kickoff. If one sport's data source fails, the other sport still updates.
- **129 pytest tests** cover the importers (including format quirks and source changes), the model (valid probabilities, recovering known team strengths from simulated leagues, no look-ahead in the backtest), the prediction record (nothing changes after kickoff), the simulator, the NFL importers, models and player projections (including that nothing uses the game being predicted), and both sports' websites. Tests use simulated leagues and temporary databases, never real data or the network.

## Project structure

```
update.py               Runs the weekly update, step by step
db.py                   Shared database helpers; schema.sql defines one database for every sport
app.py, web.py          The website: Premier League pages and shared helpers (NFL pages in nfl/site.py)
templates/, static/     Page templates and styles; build_site.py renders the site as static HTML
records/                Official live predictions, saved before kickoff
tests/                  pytest suite

epl/                    Premier League
  import_epl.py           football-data.co.uk results, stats and odds (cached, safe to re-run)
  import_understat.py     Understat xG; fills in results before football-data.co.uk posts them
  import_fixtures.py      Upcoming fixtures and pre-match odds
  model.py                The model, backtest and scoring
  run_backtest.py         Saves walk-forward backtest predictions
  predict_upcoming.py     Predicts upcoming matches and records them before kickoff
  records.py              The official prediction record in records/
  simulate.py             Monte Carlo season simulator
  fetch_availability.py   Daily snapshots of player injuries and availability
  error_analysis.py       Where the model loses ground to the market
  promoted_teams.py       How the model handles newly promoted teams
  import_fpl_values.py, import_transfermarkt.py, tune_squad_value.py, tune_value_fade.py
                          Squad-value data and experiments
  team_aliases.sql        Team-name mappings between data sources
  checks.sql              SQL sanity checks and data exploration queries
  poisson_model.ipynb     Model development notebook

nfl/                    NFL
  import_nfl.py           nflverse games, results, spreads, totals, moneylines and quarterbacks since 1999
  elo.py, tune_elo.py     Elo baseline and its tuning
  model.py, tune_model.py Points model with quarterback ratings, and its tuning and comparison
  experiments.py          Tests of rest, travel, domes, weather, EPA and other ideas (none helped)
  import_nfl_stats.py     Weekly team and player stats with EPA, injury reports and snap counts from nflverse-data
  players.py              Player projections, their 80% ranges (ranges.json), and their test against benchmarks
  player_experiments.py   Tests of injury reports, snap counts, air yards and defense by position
  predict_players.py      Projects players for upcoming games and records them before kickoff
  run_backtest.py         Saves walk-forward backtest predictions
  predict_upcoming.py     Predicts the next 8 days' games and records them before kickoff
  records.py              The official NFL prediction record in records/
  site.py                 The NFL web pages
  checks.sql              SQL data checks and exploration queries
```

## Running it

Requires Python 3.11+. Run everything from the project folder; each script is a module run with `python -m`.

```bash
git clone https://github.com/emrozin/sports-probabilities.git
cd sports-probabilities
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python update.py              # builds data/sports.db for both sports (the first run downloads every season)
uvicorn app:app --reload      # then open http://127.0.0.1:8000
pytest                        # run the tests
```

## Roadmap

- Betting lines for player stats going forward, to compare player projections with the market
- College football and basketball
- Player availability: adjust for injured and suspended players, the largest remaining source of the gap to the market
- An optional "model + market" forecast that blends the model with pre-match odds
- Use the fading squad-value adjustment for live predictions

## Acknowledgements

Data comes from football-data.co.uk, Understat, Fantasy Premier League and nflverse; please respect their terms of use. Historical FPL prices come from the [FPL Historical Dataset](https://github.com/vaastav/Fantasy-Premier-League) by Vaastav Anand (MIT License), and player valuations from [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) (CC0). Understat has no official API, so its importer may need updating if the site changes. This is a statistics project, not betting advice.
