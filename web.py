"""What every sport's web pages share: the page templates, the site's base path and formatting helpers.

app.py (Premier League) and nfl/site.py (NFL) both build their pages with these.
"""

import math
import os
import re
from pathlib import Path

import pandas as pd
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent
BASE = os.environ.get("SITE_BASE", "").rstrip("/")  # "" locally, the repository name on GitHub Pages

templates = Jinja2Templates(directory=ROOT / "templates")
templates.env.globals["base"] = BASE
# The sport menu in the header: (label, sport code, home page)
templates.env.globals["sports"] = [("Premier League", "epl", f"{BASE}/"), ("NFL", "nfl", f"{BASE}/nfl/")]


def slugify(name: str) -> str:
    """'Nott'm Forest' -> 'nottm-forest'"""
    name = name.lower().replace("'", "").replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def day_label(iso_date: str, with_year: bool = False) -> str:
    """'2026-10-03' -> 'Saturday 3 October' (or 'Sat 3 Oct 2026' with_year)"""
    d = pd.Timestamp(iso_date)
    if with_year:
        return f"{d:%a} {d.day} {d:%b %Y}"
    return f"{d:%A} {d.day} {d:%B}"


def ordinal(n: int) -> str:
    """1 -> '1st', 12 -> '12th', 22 -> '22nd'"""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def signed_number(value: float, decimals: int = 0) -> str:
    """Goal or point difference style: +7, \u22123, 0"""
    text = f"{abs(value):.{decimals}f}"
    if round(value, decimals) > 0:
        return "+" + text
    if round(value, decimals) < 0:
        return "\u2212" + text
    return text


def typical_probability(log_loss: float) -> str:
    """Log loss as a percentage: the typical probability given to what actually happened.

    One decimal place, because the model and the market are often less than a point apart.
    """
    return f"{math.exp(-log_loss) * 100:.1f}"
