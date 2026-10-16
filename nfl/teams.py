"""NFL teams: names and divisions, shared by the website and the season simulator."""

# Current names and divisions. Relocated teams appear under their current name throughout
# (Oakland Raiders -> Las Vegas, San Diego Chargers -> Los Angeles, St. Louis Rams -> Los Angeles).
TEAMS = {
    "BUF": ("Buffalo Bills", "AFC East"), "MIA": ("Miami Dolphins", "AFC East"),
    "NE": ("New England Patriots", "AFC East"), "NYJ": ("New York Jets", "AFC East"),
    "BAL": ("Baltimore Ravens", "AFC North"), "CIN": ("Cincinnati Bengals", "AFC North"),
    "CLE": ("Cleveland Browns", "AFC North"), "PIT": ("Pittsburgh Steelers", "AFC North"),
    "HOU": ("Houston Texans", "AFC South"), "IND": ("Indianapolis Colts", "AFC South"),
    "JAX": ("Jacksonville Jaguars", "AFC South"), "TEN": ("Tennessee Titans", "AFC South"),
    "DEN": ("Denver Broncos", "AFC West"), "KC": ("Kansas City Chiefs", "AFC West"),
    "LV": ("Las Vegas Raiders", "AFC West"), "LAC": ("Los Angeles Chargers", "AFC West"),
    "DAL": ("Dallas Cowboys", "NFC East"), "NYG": ("New York Giants", "NFC East"),
    "PHI": ("Philadelphia Eagles", "NFC East"), "WAS": ("Washington Commanders", "NFC East"),
    "CHI": ("Chicago Bears", "NFC North"), "DET": ("Detroit Lions", "NFC North"),
    "GB": ("Green Bay Packers", "NFC North"), "MIN": ("Minnesota Vikings", "NFC North"),
    "ATL": ("Atlanta Falcons", "NFC South"), "CAR": ("Carolina Panthers", "NFC South"),
    "NO": ("New Orleans Saints", "NFC South"), "TB": ("Tampa Bay Buccaneers", "NFC South"),
    "ARI": ("Arizona Cardinals", "NFC West"), "LA": ("Los Angeles Rams", "NFC West"),
    "SF": ("San Francisco 49ers", "NFC West"), "SEA": ("Seattle Seahawks", "NFC West"),
}
DIVISIONS_FROM = 2002   # the current eight divisions began in 2002


def conference(team: str) -> str:
    return TEAMS[team][1].split()[0]


def division(team: str) -> str:
    return TEAMS[team][1]
