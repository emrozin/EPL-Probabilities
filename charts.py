"""Small charts for the team pages, drawn as HTML and SVG on the server (no JavaScript library).

Each function turns data into positions (percentages of the chart's width and height); the
templates in templates/charts.html draw them, so the charts stay sharp at any width.
"""

import math


def nice_limit(values, step: float = 2.0, minimum: float = 4.0) -> float:
    """A symmetric axis limit around 0 that fits every value, rounded up to a multiple of step."""
    biggest = max([abs(v) for v in values] + [minimum])
    return math.ceil(biggest / step) * step


def line_chart(points: list[dict], step: float = 2.0, minimum: float = 4.0) -> dict | None:
    """points: [{"label", "value", "tip"}] in order. Values are centered on 0 (an average team).
    Returns the SVG path and each point's position for the hover markers."""
    if len(points) < 2:
        return None
    limit = nice_limit([p["value"] for p in points], step, minimum)
    n = len(points)
    placed = []
    for i, p in enumerate(points):
        x = i / (n - 1) * 100
        y = (limit - p["value"]) / (2 * limit) * 100
        placed.append({**p, "x": round(x, 2), "y": round(y, 2)})
    path = "M " + " L ".join(f"{p['x']} {p['y']}" for p in placed)
    ticks = [{"value": v, "y": round((limit - v) / (2 * limit) * 100, 2)}
             for v in (limit, limit / 2, 0, -limit / 2, -limit)]
    # x labels: no more than about 9, always the first and last
    every = max(1, math.ceil(n / 9))
    labels = [p for i, p in enumerate(placed) if i % every == 0 or i == n - 1]
    if len(labels) > 1 and labels[-1] is not labels[-2] and placed.index(labels[-1]) - placed.index(labels[-2]) < every / 2:
        labels.pop(-2)
    return {"points": placed, "path": path, "ticks": ticks, "labels": labels, "limit": limit}


def chance_bars(games: list[dict]) -> list[dict]:
    """games: [{"label", "chance" (0-1, the model's chance of this team winning), "outcome" (W/D/L/T),
    "tip"}]. Adds the bar height in percent."""
    return [{**g, "height": round(g["chance"] * 100, 1)} for g in games]
