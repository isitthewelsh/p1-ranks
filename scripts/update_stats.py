#!/usr/bin/env python3
"""
Fetches current-season MLB Stats API data (headshot ID + hitting/pitching lines)
for every player in the Top 500 and Top 400 Dynasty lists embedded in index.html,
and writes the result to data/stats-snapshot.csv.

Run weekly by .github/workflows/update-stats.yml. Can also be run manually:
    python3 scripts/update_stats.py
"""
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "index.html"
OUT_CSV = ROOT / "data" / "stats-snapshot.csv"
OUT_META = ROOT / "data" / "meta.json"

SEASONS_TO_TRY = [2026, 2025]
MAX_WORKERS = 8
REQUEST_TIMEOUT = 10


def extract_const_array(html_text, const_name):
    pattern = rf"^const {const_name} = (\[.*\]);\s*$"
    match = re.search(pattern, html_text, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find `const {const_name} = [...]` in index.html")
    return json.loads(match.group(1))


def collect_player_names():
    html_text = INDEX_HTML.read_text(encoding="utf-8")
    top500 = extract_const_array(html_text, "TOP500")
    dynasty = extract_const_array(html_text, "DYNASTY")
    names = {p["name"] for p in top500 if p.get("name")}
    names |= {p["name"] for p in dynasty if p.get("name")}
    return sorted(names)


def fetch_player_id(session, name):
    url = "https://statsapi.mlb.com/api/v1/people/search?names=" + quote(name)
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        people = r.json().get("people", [])
    except Exception:
        return None
    match = next((p for p in people if p.get("fullName", "").lower() == name.lower()), None)
    if not match and people:
        match = people[0]
    return match["id"] if match else None


def fetch_season_stat(session, player_id, group, season):
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=season&group={group}&season={season}"
    )
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        splits = (r.json().get("stats") or [{}])[0].get("splits")
    except Exception:
        return None
    if splits:
        return splits[-1]["stat"]
    return None


def fetch_player_row(session, name):
    player_id = fetch_player_id(session, name)
    row = {
        "name": name,
        "mlb_id": player_id or "",
        "season": "",
        "h_avg": "", "h_hr": "", "h_rbi": "", "h_r": "", "h_sb": "",
        "p_w": "", "p_era": "", "p_whip": "", "p_k": "", "p_ip": "",
    }
    if not player_id:
        return row

    for season in SEASONS_TO_TRY:
        hitting = fetch_season_stat(session, player_id, "hitting", season)
        pitching = fetch_season_stat(session, player_id, "pitching", season)
        if hitting or pitching:
            row["season"] = season
            if hitting:
                row["h_avg"] = hitting.get("avg", "")
                row["h_hr"] = hitting.get("homeRuns", "")
                row["h_rbi"] = hitting.get("rbi", "")
                row["h_r"] = hitting.get("runs", "")
                row["h_sb"] = hitting.get("stolenBases", "")
            if pitching:
                row["p_w"] = pitching.get("wins", "")
                row["p_era"] = pitching.get("era", "")
                row["p_whip"] = pitching.get("whip", "")
                row["p_k"] = pitching.get("strikeOuts", "")
                row["p_ip"] = pitching.get("inningsPitched", "")
            break
    return row


def main():
    names = collect_player_names()
    print(f"Found {len(names)} unique players across TOP500 + DYNASTY", file=sys.stderr)

    rows = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "ProspectOneStatsSnapshot/1.0"})
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(fetch_player_row, session, name): name for name in names}
            done = 0
            for future in as_completed(futures):
                rows.append(future.result())
                done += 1
                if done % 50 == 0:
                    print(f"  ...{done}/{len(names)}", file=sys.stderr)

    rows.sort(key=lambda r: r["name"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "mlb_id", "season",
                  "h_avg", "h_hr", "h_rbi", "h_r", "h_sb",
                  "p_w", "p_era", "p_whip", "p_k", "p_ip"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    matched = sum(1 for r in rows if r["mlb_id"])
    with_stats = sum(1 for r in rows if r["season"])
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}: {matched}/{len(rows)} matched an MLB ID, "
          f"{with_stats} have season stats on file", file=sys.stderr)

    OUT_META.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "player_count": len(rows),
        "matched_count": matched,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_META.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
