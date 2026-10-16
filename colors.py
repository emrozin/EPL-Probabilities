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

# College football: school colors from cfbfastR-data's team list (team name as in its schedules)
CFB = {
    'Air Force': ('#004A7B', '#FFFFFF'),
    'Akron': ('#00285E', '#84754E'),
    'Alabama': ('#690014', '#F1F2F3'),
    'Arizona': ('#002449', '#00205B'),
    'Arizona State': ('#942139', '#F1F2F3'),
    'Arkansas': ('#9C1831', '#000000'),
    'Arkansas State': ('#E81018', '#000000'),
    'Army': ('#CE9C00', '#231F20'),
    'Auburn': ('#03244D', '#F1F2F3'),
    'BYU': ('#001E4C', '#FFFFFF'),
    'Ball State': ('#DA0000', '#FFFFFF'),
    'Baylor': ('#004834', '#FFB81C'),
    'Boise State': ('#09347A', '#D8D9DA'),
    'Boston College': ('#88001A', '#A39161'),
    'Bowling Green': ('#2B1000', '#492000'),
    'Buffalo': ('#041A9B', '#EBEBEB'),
    'California': ('#031522', '#FFC423'),
    'Central Michigan': ('#6A0032', '#FFFFFF'),
    'Charlotte': ('#0C562D', '#CFAB7A'),
    'Cincinnati': ('#000000', '#717073'),
    'Clemson': ('#F66733', '#522D80'),
    'Coastal Carolina': ('#007073', '#876447'),
    'Colorado': ('#D1C57E', '#FFD200'),
    'Colorado State': ('#004537', '#FFC425'),
    'Connecticut': ('#001D40', '#F1F2F3'),
    'Delaware': ('#033594', '#E8CE31'),
    'Duke': ('#001A57', '#F1F2F3'),
    'East Carolina': ('#4B1869', '#F0907B'),
    'Eastern Michigan': ('#00331B', '#F0F0F0'),
    'Florida': ('#0021A5', '#0021A5'),
    'Florida Atlantic': ('#004B85', '#BB2F4C'),
    'Florida International': ('#001538', '#C5960C'),
    'Florida State': ('#782F40', '#CEB888'),
    'Fresno State': ('#C41230', '#231F20'),
    'Georgia': ('#CC0000', '#000000'),
    'Georgia Southern': ('#003775', '#F0F0F0'),
    'Georgia State': ('#1E539A', '#EBEBEB'),
    'Georgia Tech': ('#00223E', '#002C56'),
    "Hawai'i": ('#003420', '#FFFFFF'),
    'Houston': ('#C90822', '#231F20'),
    'Idaho': ('#000000', '#8C6E4A'),
    'Illinois': ('#F77329', '#FA6300'),
    'Indiana': ('#7D110C', '#EEEDEB'),
    'Iowa': ('#000000', '#FFE100'),
    'Iowa State': ('#660015', '#830B2C'),
    'Jacksonville State': ('#B50500', '#B5B7BA'),
    'James Madison': ('#450084', '#CBB778'),
    'Kansas': ('#0022B4', '#E8000D'),
    'Kansas State': ('#633194', '#E7D2AD'),
    'Kennesaw State': ('#000000', '#FDBB30'),
    'Kent State': ('#002445', '#F0B510'),
    'Kentucky': ('#005DAA', '#FFFFFF'),
    'LSU': ('#22005C', '#FDD023'),
    'Liberty': ('#071740', '#A61F21'),
    'Louisiana': ('#CE2842', '#7F7F7F'),
    'Louisiana Monroe': ('#231F20', '#B18445'),
    'Louisiana Tech': ('#002D65', '#D3313A'),
    'Louisville': ('#AD000A', '#CCCCCC'),
    'Marshall': ('#186329', '#BE854C'),
    'Maryland': ('#D5002B', '#FFCD00'),
    'Memphis': ('#002447', '#231F20'),
    'Miami': ('#004325', '#F0F0F0'),
    'Miami (OH)': ('#A4000C', '#F0F0F0'),
    'Michigan': ('#00274C', '#FFCB05'),
    'Michigan State': ('#18453B', '#FFFFFF'),
    'Middle Tennessee': ('#0079C2', '#C3C5C8'),
    'Minnesota': ('#7F011B', '#7A0019'),
    'Mississippi State': ('#762123', '#C8C8C8'),
    'Missouri': ('#000000', '#000000'),
    'Missouri State': ('#5F0000', '#E8E8E8'),
    'NC State': ('#EF1216', '#231F20'),
    'Navy': ('#131630', '#B6A77C'),
    'Nebraska': ('#F20017', '#F5F1E7'),
    'Nevada': ('#153E5F', '#8E9295'),
    'New Mexico': ('#000000', '#231F20'),
    'New Mexico State': ('#891216', '#000000'),
    'North Carolina': ('#99BFE5', '#13294B'),
    'North Dakota State': ('#01402A', '#FFFFFF'),
    'North Texas': ('#00853D', '#000000'),
    'Northern Illinois': ('#F1122C', '#CC0000'),
    'Northwestern': ('#372286', '#D6CAC1'),
    'Notre Dame': ('#00122B', '#AE9142'),
    'Ohio': ('#295A29', '#E4BB85'),
    'Ohio State': ('#DE3121', '#666666'),
    'Oklahoma': ('#7B0000', '#CCCCCC'),
    'Oklahoma State': ('#FF6500', '#FF9900'),
    'Old Dominion': ('#00507D', '#A1D2F1'),
    'Ole Miss': ('#001148', '#00205B'),
    'Oregon': ('#044520', '#FEE123'),
    'Oregon State': ('#C34500', '#DEA076'),
    'Penn State': ('#00265D', '#002E5C'),
    'Pittsburgh': ('#003263', '#231F20'),
    'Purdue': ('#B89D29', '#A4A9AD'),
    'Rice': ('#003D7D', '#D1D5D8'),
    'Rutgers': ('#D21034', '#FFFFFF'),
    'SMU': ('#E32F38', '#ECDCB9'),
    'Sacramento State': ('#00573C', '#CDB97D'),
    'San Diego State': ('#BF2C37', '#A8996E'),
    'San José State': ('#005893', '#FDBA31'),
    'South Alabama': ('#003E7E', '#7F7F7F'),
    'South Carolina': ('#670010', '#000000'),
    'South Florida': ('#004A36', '#231F20'),
    'Southern Mississippi': ('#FFAA3C', '#FFC423'),
    'Stanford': ('#A80532', '#FFFFFF'),
    'Syracuse': ('#F37321', '#0D1D37'),
    'TCU': ('#3C377D', '#F1F2F3'),
    'Temple': ('#A80532', '#A7A9AC'),
    'Tennessee': ('#EE9627', '#FFFFFF'),
    'Texas': ('#EE7524', '#F0F0F0'),
    'Texas A&M': ('#5C0025', '#FFFFFF'),
    'Texas State': ('#4E1719', '#B4975A'),
    'Texas Tech': ('#C80025', '#231F20'),
    'Toledo': ('#00488F', '#FFD51D'),
    'Troy': ('#AE0210', '#88898C'),
    'Tulane': ('#005837', '#3BAF29'),
    'Tulsa': ('#004371', '#EE3B33'),
    'UAB': ('#003B28', '#FFC845'),
    'UCF': ('#000000', '#231F20'),
    'UCLA': ('#005C8E', '#FFC72C'),
    'UMass': ('#880007', '#7F7F7F'),
    'UNLV': ('#000000', '#C2C4C6'),
    'USC': ('#AE2531', '#FFC72C'),
    'UT San Antonio': ('#002A5C', '#F47321'),
    'UTEP': ('#002843', '#FF6F12'),
    'Utah': ('#890012', '#7E8083'),
    'Utah State': ('#003263', '#949CA1'),
    'Vanderbilt': ('#000000', '#231F20'),
    'Virginia': ('#00204E', '#002D62'),
    'Virginia Tech': ('#74232D', '#C2C1BA'),
    'Wake Forest': ('#9E7E38', '#000000'),
    'Washington': ('#2B2F64', '#E8E3D3'),
    'Washington State': ('#94022A', '#6A747C'),
    'West Virginia': ('#FFC600', '#EAAA00'),
    'Western Kentucky': ('#F32026', '#B3B5B8'),
    'Western Michigan': ('#B59966', '#BFA979'),
    'Wisconsin': ('#A00001', '#F7F7F7'),
    'Wyoming': ('#533B22', '#FFFFFF'),
    'App State': ('#222222', '#FFCC00'), 'Massachusetts': ('#881C1C', '#212721'), 'Sam Houston': ('#F56A00', '#212721'),
    'Southern Miss': ('#212721', '#FFAB00'), 'UConn': ('#000E2F', '#7C878E'), 'UL Monroe': ('#800029', '#BD955A'),
    'UTSA': ('#0C2340', '#F15A22'),
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


def usable(hex_color: str) -> bool:
    """Not so close to white that it would vanish on a card."""
    return _lab(hex_color)[0] < 90


def main_color(table: dict, team: str, default: tuple) -> str:
    colors = table.get(team, default)
    return next((c for c in colors if usable(c)), default[0])


def pair(table: dict, home: str, away: str) -> tuple[str, str]:
    """(home color, away color) for one game, the away team switching to its second color (or the
    default) when the main colors clash."""
    h = main_color(table, home, DEFAULT_HOME)
    a = table.get(away, DEFAULT_AWAY)
    for option in (*a, DEFAULT_AWAY[0], DEFAULT_HOME[1]):
        if usable(option) and difference(h, option) >= MIN_DIFFERENCE:
            return h, option
    return h, main_color(table, away, DEFAULT_AWAY)


def style(table: dict, home: str, away: str) -> str:
    """CSS custom properties for a game: each team's color and the text color on it, for the light
    page (--hl, --al) and the dark one (--hd, --ad). The stylesheet picks one set as --home/--away."""
    h, a = pair(table, home, away)
    hd, ad = for_dark(h), for_dark(a)
    return (f"--hl: {h}; --hl-ink: {ink(h)}; --al: {a}; --al-ink: {ink(a)}; "
            f"--hd: {hd}; --hd-ink: {ink(hd)}; --ad: {ad}; --ad-ink: {ink(ad)};")


def team_style(table: dict, team: str) -> str:
    """CSS custom properties for one team's page: its main color as the home color."""
    c = main_color(table, team, DEFAULT_HOME)
    d = for_dark(c)
    return f"--hl: {c}; --hl-ink: {ink(c)}; --hd: {d}; --hd-ink: {ink(d)}; --al: {c}; --al-ink: {ink(c)}; --ad: {d}; --ad-ink: {ink(d)};"


def chip(table: dict, team: str):
    """A small square in the team's main color (lifted on the dark page), as HTML."""
    from markupsafe import Markup
    c = main_color(table, team, DEFAULT_HOME)
    return Markup(f'<span class="chip team-chip" style="--hl: {c}; --hd: {for_dark(c)}" aria-hidden="true"></span>')
