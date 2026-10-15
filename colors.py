"""Team colors for the website: each team's bar, chip and header use its own colors.

Colors only (no logos, which are trademarks). When two teams' main colors are too alike to tell
apart in one bar (Liverpool and Man United are both red), the away team uses its second color.
"""

import math

# (main color, second color)
NFL = {
    "ARI": ("#97233F", "#FFB612"), "ATL": ("#A71930", "#101820"), "BAL": ("#241773", "#9E7C0C"),
    "BUF": ("#00338D", "#C60C30"), "CAR": ("#0085CA", "#101820"), "CHI": ("#0B162A", "#C83803"),
    "CIN": ("#FB4F14", "#101820"), "CLE": ("#311D00", "#FF3C00"), "DAL": ("#003594", "#869397"),
    "DEN": ("#FB4F14", "#002244"), "DET": ("#0076B6", "#B0B7BC"), "GB": ("#203731", "#FFB612"),
    "HOU": ("#03202F", "#A71930"), "IND": ("#002C5F", "#A2AAAD"), "JAX": ("#006778", "#D7A22A"),
    "KC": ("#E31837", "#FFB81C"), "LV": ("#101820", "#A5ACAF"), "LAC": ("#0080C6", "#FFC20E"),
    "LA": ("#003594", "#FFA300"), "MIA": ("#008E97", "#FC4C02"), "MIN": ("#4F2683", "#FFC62F"),
    "NE": ("#002244", "#C60C30"), "NO": ("#D3BC8D", "#101820"), "NYG": ("#0B2265", "#A71930"),
    "NYJ": ("#125740", "#101820"), "PHI": ("#004C54", "#A5ACAF"), "PIT": ("#FFB612", "#101820"),
    "SEA": ("#002244", "#69BE28"), "SF": ("#AA0000", "#B3995D"), "TB": ("#D50A0A", "#34302B"),
    "TEN": ("#0C2340", "#4B92DB"), "WAS": ("#5A1414", "#FFB612"),
}

# football-data.co.uk's team names
EPL = {
    "Arsenal": ("#EF0107", "#063672"), "Aston Villa": ("#670E36", "#95BFE5"), "Barnsley": ("#D71920", "#101820"),
    "Birmingham": ("#0000FF", "#D71920"), "Blackburn": ("#009EE0", "#101820"), "Blackpool": ("#F68712", "#101820"),
    "Bolton": ("#263C7E", "#D71920"), "Bournemouth": ("#DA291C", "#101820"), "Bradford": ("#8A0F2C", "#F5A12D"),
    "Brentford": ("#E30613", "#101820"), "Brighton": ("#0057B8", "#FFCD00"), "Burnley": ("#6C1D45", "#99D6EA"),
    "Cardiff": ("#0070B5", "#D71920"), "Charlton": ("#D4021D", "#101820"), "Chelsea": ("#034694", "#DBA111"),
    "Coventry": ("#59CBE8", "#101820"), "Crystal Palace": ("#1B458F", "#C4122E"), "Derby": ("#101820", "#7F7F7F"),
    "Everton": ("#003399", "#FFCC00"), "Fulham": ("#101820", "#CC0000"), "Huddersfield": ("#0E63AD", "#FDD116"),
    "Hull": ("#F5A12D", "#101820"), "Ipswich": ("#3A64A3", "#DE2C37"), "Leeds": ("#FFCD00", "#1D428A"),
    "Leicester": ("#003090", "#FDBE11"), "Liverpool": ("#C8102E", "#00B2A9"), "Luton": ("#F78F1E", "#002D62"),
    "Man City": ("#6CABDD", "#1C2C5B"), "Man United": ("#DA291C", "#FBE122"), "Middlesbrough": ("#E11B22", "#101820"),
    "Newcastle": ("#241F20", "#41B6E6"), "Norwich": ("#00A650", "#FFF200"), "Nott'm Forest": ("#DD0000", "#101820"),
    "Oldham": ("#004B9D", "#D71920"), "Portsmouth": ("#001489", "#D71920"), "QPR": ("#1D5BA4", "#D71920"),
    "Reading": ("#004494", "#D71920"), "Sheffield United": ("#EE2737", "#101820"),
    "Sheffield Weds": ("#4473C5", "#F5A12D"), "Southampton": ("#D71920", "#130C0E"), "Stoke": ("#E03A3E", "#1B449C"),
    "Sunderland": ("#EB172B", "#101820"), "Swansea": ("#101820", "#7F7F7F"), "Swindon": ("#D71920", "#101820"),
    "Tottenham": ("#132257", "#7F7F7F"), "Watford": ("#FBEE23", "#ED2127"), "West Brom": ("#122F67", "#7F7F7F"),
    "West Ham": ("#7A263A", "#1BB1E7"), "Wigan": ("#1D59AF", "#101820"), "Wimbledon": ("#0033A0", "#FFE600"),
    "Wolves": ("#FDB913", "#231F20"),
}

DEFAULT_HOME, DEFAULT_AWAY = ("#1D4F91", "#5A6B8C"), ("#D9A21B", "#8C6A12")
MIN_DIFFERENCE = 40   # color difference (CIE76, Lab) below which two colors read as the same in a bar


def _lab(hex_color: str) -> tuple[float, float, float]:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    x = (0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]) / 0.95047
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    z = (0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def _hex(lab: tuple[float, float, float]) -> str:
    l, a, b = lab
    fy = (l + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    inv = lambda t: t ** 3 if t ** 3 > 0.008856 else (t - 16 / 116) / 7.787
    x, y, z = inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883
    lin = (3.2406 * x - 1.5372 * y - 0.4986 * z, -0.9689 * x + 1.8758 * y + 0.0415 * z,
           0.0557 * x - 0.2040 * y + 1.0570 * z)
    srgb = [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in lin]
    return "#" + "".join(f"{round(min(1, max(0, c)) * 255):02X}" for c in srgb)


DARK_MIN_LIGHTNESS = 42   # on the dark page, team colors darker than this are lifted to it (same hue)


def for_dark(hex_color: str) -> str:
    """The same color, lifted so it reads on a near-black page (Seattle's navy becomes a brighter navy)."""
    l, a, b = _lab(hex_color)
    if l >= DARK_MIN_LIGHTNESS:
        return hex_color
    boost = 1.6   # lifting alone washes a color out; keep it as vivid as the original looks
    return _hex((DARK_MIN_LIGHTNESS, a * boost, b * boost))


def difference(a: str, b: str) -> float:
    return math.dist(_lab(a), _lab(b))


def ink(hex_color: str) -> str:
    """Text color that reads on top of this color: near-black on light colors, white on dark ones."""
    return "#141b17" if _lab(hex_color)[0] > 62 else "#ffffff"


def pair(table: dict, home: str, away: str) -> tuple[str, str]:
    """(home color, away color) for one game, the away team switching to its second color (or the
    default) when the main colors clash."""
    h = table.get(home, DEFAULT_HOME)
    a = table.get(away, DEFAULT_AWAY)
    for option in (a[0], a[1], DEFAULT_AWAY[0], DEFAULT_HOME[1]):
        if difference(h[0], option) >= MIN_DIFFERENCE:
            return h[0], option
    return h[0], a[0]


def style(table: dict, home: str, away: str) -> str:
    """CSS custom properties for a game: each team's color and the text color on it, for the light
    page (--hl, --al) and the dark one (--hd, --ad). The stylesheet picks one set as --home/--away."""
    h, a = pair(table, home, away)
    hd, ad = for_dark(h), for_dark(a)
    return (f"--hl: {h}; --hl-ink: {ink(h)}; --al: {a}; --al-ink: {ink(a)}; "
            f"--hd: {hd}; --hd-ink: {ink(hd)}; --ad: {ad}; --ad-ink: {ink(ad)};")


def team_style(table: dict, team: str) -> str:
    """CSS custom properties for one team's page: its main color as the home color."""
    c = table.get(team, DEFAULT_HOME)[0]
    d = for_dark(c)
    return f"--hl: {c}; --hl-ink: {ink(c)}; --hd: {d}; --hd-ink: {ink(d)}; --al: {c}; --al-ink: {ink(c)}; --ad: {d}; --ad-ink: {ink(d)};"


def chip(table: dict, team: str):
    """A small square in the team's main color (lifted on the dark page), as HTML."""
    from markupsafe import Markup
    c = table.get(team, DEFAULT_HOME)[0]
    return Markup(f'<span class="chip team-chip" style="--hl: {c}; --hd: {for_dark(c)}" aria-hidden="true"></span>')
