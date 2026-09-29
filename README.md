Premier League Match Probabilities

A statistical model that estimates win, draw and loss probabilities for 
English Premier League matches,evaluated honestly against the sharpest 
benchmark available: closing betting-market odds.

The project covers the full pipeline: collecting and validating 30+ seasons 
of data from two independent sources, building and tuning a Poisson goals model 
with expected goals (xG), and testing it with walk-forward backtesting on seasons 
it never saw during development.

Tech: Python, SQL (SQLite), pandas, NumPy, SciPy, Jupyter

Results:

All numbers below are from held-out seasons (2021/22 to present, 1,921 matches)
that played no part in choosing the model's settings. Lower log loss is better; 
guessing one-third for every outcome scores 1.0986.

Model	                                                        | Log loss
--------------------------------------------------------------|---------
Goals only, initial settings	                                | 0.9824
Goals only, tuned	                                            | 0.9813
expected goals (xG) blend	                                    | 0.9778
xG with re-tuned recency weighting	                          | 0.9754
Closing market (Pinnacle / Betfair Exchange, margin removed)	| 0.9592

Adding xG and re-tuning closed 27% of the gap between the goals-only model
and the closing market. The model does not beat the market, and isn't 
expected to: closing odds reflect injuries, confirmed lineups, transfers
and professional bettors' information, while this model only sees past results.
The goal is to measure how close a transparent, results-based model can get, 
and which changes actually help.

Results as of September 2026; the current season updates as new matches are played.

Data:

Source	               |      Coverage	           |       Contents
-----------------------|---------------------------|----------------------------------
football-data.co.uk    |	    1993/94 onward	     |    Results, half-time scores, match stats (from 2000/01), 1X2 / over-under / Asian handicap odds from 20+ bookmakers, pre-match and closing.
Understat	             |      2014/15 onward	     |    Expected goals (xG) per team per match

The database currently holds 13,000+ matches. Some data-quality work involved:

- Inconsistent file formats across 30+ seasons. Column names change
over time (e.g. BbAv became Avg), and some seasons have rows with more
fields than the header. The importer handles both, and every original
row is also stored as JSON so no source column is ever lost.
- Unmapped column report. Each import lists any source columns not mapped
to a table, which is how new bookmakers and a new xG column were discovered.
- Cross-source validation. Understat and football-data.co.uk use different
team names (e.g. "Manchester United" vs "Man United"), resolved through an
alias table. All 4,610 linked matches were checked against both sources' final
scores, with 100% agreement.
- Source changes. Midway through development, Understat stopped embedding data
in its pages. The importer failed loudly rather than importing nothing, and the
fix was isolated to one function.

Model:

Each team has an attack and a defense rating, plus a league-wide home advantage.
A team's expected goals in a match are:

  expected goals = base rate × home advantage (if at home) × attack ÷ opponent's defense

Each team's goals are modeled as a Poisson distribution around its expected goals, 
which gives the probability of every exact scoreline. Summing the scorelines gives
the home win, draw and away win probabilities.

Ratings are fitted by maximum likelihood (SciPy's L-BFGS-B with an analytic gradient,
about 30× faster than numerical differentiation) using the last three years of matches,
with:

- Recency weighting: a match's weight halves every 180 days.
- Shrinkage (L2 regularization): ratings are pulled toward average,
so teams with few matches, such as newly promoted sides, don't get
extreme ratings from a handful of results.
- xG blend: ratings are learned from a 50/50 blend of actual goals and xG.

Evaluation:

- Walk-forward backtest: for every week since 2016/17, the model is trained only
on matches played before that week, then predicts that week's games. It never
sees a result it is predicting.
- Validation / test split: all settings were tuned on 2016/17–2020/21, and final
results are reported on 2021/22 onward.
- Metrics: log loss and Brier score, compared against closing odds from Pinnacle
(Betfair Exchange where Pinnacle is unavailable), with the bookmaker margin removed.

Findings:

- Home advantage is shrinking. Home teams won around 45–50% of matches in the 1990s and 2000s,
and closer to 41–48% recently.
- Crowds matter. In 2020/21, played without fans, away teams won more often than home teams
(40.3% vs 37.9%), the only season where that happened.
- Scoring jumped in 2023/24 to 3.28 goals per game, the highest on record, coinciding with
a crackdown on time-wasting that added stoppage time. The goals-only model was slow
to adapt that season.
- Goals and xG work best together. On validation seasons, goals only scored 0.9571,
a 50/50 blend 0.9532, and pure xG 0.9552. Actual goals carry information xG misses,
such as finishing and goalkeeping quality.
- xG allows a shorter memory. With xG, the best recency half-life dropped from 365 to 180 days,
and less shrinkage was needed.
- Dixon–Coles didn't help. This well-known correction for low-scoring draws made
predictions slightly worse: the model already predicted slightly more draws than occurred
(23.7% vs 22.5%). The apparent excess of low scores in the raw data came from pooling
strong and weak teams, which team-specific ratings already account for.


Project structure: 

import_epl.py         Imports football-data.co.uk results, stats and odds (cached, safe to re-run)

import_understat.py   Imports Understat xG and links it to existing matches

schema.sql            Database schema, designed to support more leagues and sports

team_aliases.sql      Team-name mappings between data sources

checks.sql            SQL sanity checks and data exploration queries

poisson_model.ipynb   Model, backtest, tuning and evaluation

Running it:
Requires Python 3.11+.

bash

git clone https://github.com/<erozin>/epl-probabilities.git

cd epl-probabilities

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python import_epl.py         # builds data/sports.db (first run downloads all seasons)

python import_understat.py   # adds xG

Then open poisson_model.ipynb and run all cells.

Roadmap: 

- Website showing upcoming fixtures with model probabilities next to bookmaker odds, plus live backtest results
- Championship data so promoted teams start with a realistic rating
- Error analysis: where the model loses the most ground to the market
- More leagues: La Liga, Bundesliga, Serie A, Ligue 1 and MLS, then the NFL

Notes:

Data comes from football-data.co.uk and Understat; please respect their terms of use. 
Understat has no official API, so its importer may need updating if the site changes.
This is a statistics project, not betting advice.
