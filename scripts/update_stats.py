#!/usr/bin/env python3
"""
Fetches current-season MLB Stats API data (headshot ID + hitting/pitching lines)
for every player in the Top 500 and Top 400 Dynasty lists embedded in index.html,
and writes the result to data/stats-snapshot.csv.

Checks MLB first, then falls back through affiliated minor-league levels
(Triple-A -> Double-A -> High-A -> Single-A -> Rookie) so prospects who
haven't debuted yet still get their current-level stat line, since the MLB
Stats API covers the full minor-league system under different sportIds.

Player ID resolution has a few failure modes the naive "exact name match on
the first search result" approach gets wrong, so fetch_player_id() layers
several strategies:
  1. data/id-overrides.json -- manual name -> id map for players confirmed
     missing from MLB's search index entirely (checked via direct /people/{id}
     lookup), e.g. very recent draftees the index hasn't caught up on yet.
  2. Full-name search, matched with diacritics stripped (our source names are
     plain ASCII; MLB's are properly accented, e.g. "Pena" vs "Pena").
  3. If a name search returns multiple/no confident matches, disambiguate (or
     retry with a last-name-only search) using age, since common names can
     collide with unrelated retired players decades apart in age.
  4. Cross-reference the MLB Draft API (which reliably links draft picks to
     their person id) for players still unresolved after the above.

Run weekly by .github/workflows/update-stats.yml. Can also be run manually:
    python3 scripts/update_stats.py
"""
import csv
import json
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "index.html"
OUT_CSV = ROOT / "data" / "stats-snapshot.csv"
OUT_META = ROOT / "data" / "meta.json"
ID_OVERRIDES_FILE = ROOT / "data" / "id-overrides.json"

SEASONS_TO_TRY = [2026, 2025]
# (sportId, display label), checked in order -- MLB first, then most advanced
# minor-league level down to Rookie ball.
LEVELS_TO_TRY = [
    (1, "MLB"),
    (11, "AAA"),
    (12, "AA"),
    (13, "A+"),
    (14, "A"),
    (16, "ROK"),
]
DRAFT_YEARS_TO_INDEX = range(2019, 2026)
# MLB + every affiliated minor-league level, for building a roster-based
# name index -- far more complete than name search for current players.
ROSTER_SPORT_IDS = [1, 11, 12, 13, 14, 16]
ROSTER_SEASON = 2026
AGE_MATCH_TOLERANCE_YEARS = 5
MAX_WORKERS = 16
REQUEST_TIMEOUT = 10


NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(s):
    """Lowercase, strip diacritics ('Pena' == 'Pena'), drop periods from
    initials ('J.D. Dix' == 'JD Dix', which is how MLB's own records store
    it), and drop a trailing generational suffix (our source data has
    "Eric Booth Jr", MLB has "Eric Booth") -- any of these otherwise breaks
    both the search query and the exact-name comparison."""
    nfkd = unicodedata.normalize("NFKD", s or "")
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    stripped = stripped.replace(".", "")
    stripped = re.sub(r"\s+", " ", stripped).lower().strip()
    parts = stripped.split()
    if len(parts) > 1 and parts[-1] in NAME_SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts)


def age_from_birthdate(birth_date_str):
    if not birth_date_str:
        return None
    try:
        y, m, d = (int(x) for x in birth_date_str.split("-"))
    except ValueError:
        return None
    today = date.today()
    return (today - date(y, m, d)).days / 365.25


def extract_const_array(html_text, const_name):
    pattern = rf"^const {const_name} = (\[.*\]);\s*$"
    match = re.search(pattern, html_text, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find `const {const_name} = [...]` in index.html")
    return json.loads(match.group(1))


def collect_players():
    """Returns {name: age_hint_or_None}, deduped across TOP500 + DYNASTY."""
    html_text = INDEX_HTML.read_text(encoding="utf-8")
    top500 = extract_const_array(html_text, "TOP500")
    dynasty = extract_const_array(html_text, "DYNASTY")
    players = {}
    for p in top500 + dynasty:
        name = p.get("name")
        if not name:
            continue
        if name not in players or players[name] is None:
            players[name] = p.get("age")
    return players


def load_id_overrides():
    if not ID_OVERRIDES_FILE.exists():
        return {}
    data = json.loads(ID_OVERRIDES_FILE.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def fetch_team_ids(session, sport_id):
    try:
        r = session.get(f"https://statsapi.mlb.com/api/v1/teams?sportId={sport_id}", timeout=15)
        r.raise_for_status()
        return [t["id"] for t in r.json().get("teams", [])]
    except Exception:
        return []


def fetch_team_roster(session, team_id):
    url = (f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
           f"?rosterType=fullSeason&season={ROSTER_SEASON}&hydrate=person(birthDate)")
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        return r.json().get("roster", [])
    except Exception:
        return []


def build_roster_index(session):
    """normalized full name -> list of (id, birthDate), across every MLB +
    affiliated minor-league roster. Far more reliable than name search for
    currently-active players, since the search index visibly lags for some
    recently-added prospects (see module docstring)."""
    team_ids = []
    for sport_id in ROSTER_SPORT_IDS:
        team_ids += fetch_team_ids(session, sport_id)

    index = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(fetch_team_roster, session, tid) for tid in team_ids]
        for future in as_completed(futures):
            for entry in future.result():
                person = entry.get("person") or {}
                pid = person.get("id")
                full_name = person.get("fullName")
                if not pid or not full_name:
                    continue
                key = normalize_name(full_name)
                index.setdefault(key, []).append((pid, person.get("birthDate")))
    return index


def build_draft_index(session):
    """normalized full name -> list of (id, birthDate), across DRAFT_YEARS_TO_INDEX."""
    index = {}
    for year in DRAFT_YEARS_TO_INDEX:
        try:
            r = session.get(f"https://statsapi.mlb.com/api/v1/draft/{year}", timeout=30)
            r.raise_for_status()
            rounds = r.json().get("drafts", {}).get("rounds", [])
        except Exception:
            continue
        for rnd in rounds:
            for pick in rnd.get("picks", []):
                person = pick.get("person") or {}
                pid = person.get("id")
                full_name = person.get("fullName")
                if not pid or not full_name:
                    continue
                key = normalize_name(full_name)
                index.setdefault(key, []).append((pid, person.get("birthDate")))
    return index


def pick_best_candidate(candidates, age_hint):
    """candidates: list of (id, birthDate). Returns id or None.

    Whenever an age hint is available, EVERY candidate must pass the age-
    tolerance check -- including when there's only one candidate, since a
    lone search result can still be an unrelated same-name player decades
    apart in age (observed: a common name returning only a retired 1980s
    player when the real prospect isn't indexed under that name at all).
    Only trust an unverified single candidate when no age hint exists.
    """
    if not candidates:
        return None
    if age_hint is not None:
        scored = []
        for pid, birth_date in candidates:
            candidate_age = age_from_birthdate(birth_date)
            if candidate_age is None:
                continue
            diff = abs(candidate_age - age_hint)
            if diff <= AGE_MATCH_TOLERANCE_YEARS:
                scored.append((diff, pid))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        return scored[0][1]
    return candidates[0][0] if len(candidates) == 1 else None


def search_people(session, query):
    url = "https://statsapi.mlb.com/api/v1/people/search?names=" + quote(query)
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json().get("people", [])
    except Exception:
        return []


def _plausible_first_name(query_first, candidate_first):
    if not query_first or not candidate_first:
        return False
    return (
        candidate_first == query_first
        or candidate_first.startswith(query_first)
        or query_first.startswith(candidate_first)
    )


def fetch_player_id(session, name, age_hint, overrides, draft_index, roster_index):
    if name in overrides:
        return overrides[name]

    normalized_query = normalize_name(name)
    query_parts = normalized_query.split()
    query_first = query_parts[0] if query_parts else ""
    query_last = query_parts[-1] if query_parts else ""

    # Merge exact-name candidates from every source: roster (most reliable --
    # every affiliated MLB + minor-league team's current-season roster), the
    # draft-pick index (catches recent draftees search hasn't indexed yet),
    # and name search itself.
    people = search_people(session, name.replace(".", ""))
    exact = list(roster_index.get(normalized_query, []))
    exact += draft_index.get(normalized_query, [])
    exact += [(p["id"], p.get("birthDate")) for p in people
              if normalize_name(p.get("fullName", "")) == normalized_query]
    result = pick_best_candidate(exact, age_hint)
    if result:
        return result

    # Fall back to last-name-only matching across the same sources,
    # disambiguated by age -- catches nickname mismatches (e.g. "Josh" vs
    # "Joshua") and cases the exact-name pass didn't surface at all. Still
    # requires the first name to at least plausibly match (prefix either
    # direction) -- last name alone isn't enough, or e.g. "Luis Hernandez"
    # can silently match an unrelated "Miguel Hernandez" of similar age.
    if query_last and query_last != normalized_query:
        same_last = []
        for key, candidates in roster_index.items():
            parts = key.split()
            if parts and parts[-1] == query_last and _plausible_first_name(query_first, parts[0]):
                same_last += candidates
        for key, candidates in draft_index.items():
            parts = key.split()
            if parts and parts[-1] == query_last and _plausible_first_name(query_first, parts[0]):
                same_last += candidates

        people2 = search_people(session, query_last)
        for p in people2:
            if normalize_name(p.get("lastName", "")) != query_last:
                continue
            if _plausible_first_name(query_first, normalize_name(p.get("firstName", ""))):
                same_last.append((p["id"], p.get("birthDate")))

        result = pick_best_candidate(same_last, age_hint)
        if result:
            return result

    return None


def fetch_season_stat(session, player_id, group, season, sport_id):
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=season&group={group}&season={season}&sportId={sport_id}"
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


def fetch_player_row(session, name, age_hint, overrides, draft_index, roster_index):
    player_id = fetch_player_id(session, name, age_hint, overrides, draft_index, roster_index)
    row = {
        "name": name,
        "mlb_id": player_id or "",
        "season": "",
        "level": "",
        "h_avg": "", "h_hr": "", "h_rbi": "", "h_r": "", "h_sb": "",
        "p_w": "", "p_era": "", "p_whip": "", "p_k": "", "p_ip": "",
    }
    if not player_id:
        return row

    for season in SEASONS_TO_TRY:
        for sport_id, level_label in LEVELS_TO_TRY:
            hitting = fetch_season_stat(session, player_id, "hitting", season, sport_id)
            pitching = fetch_season_stat(session, player_id, "pitching", season, sport_id)
            if hitting or pitching:
                row["season"] = season
                row["level"] = level_label
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
                return row
    return row


def main():
    players = collect_players()
    overrides = load_id_overrides()
    print(f"Found {len(players)} unique players across TOP500 + DYNASTY "
          f"({len(overrides)} manual ID overrides loaded)", file=sys.stderr)

    with requests.Session() as session:
        session.headers.update({"User-Agent": "ProspectOneStatsSnapshot/1.0"})

        print(f"Indexing MLB drafts {DRAFT_YEARS_TO_INDEX.start}-{DRAFT_YEARS_TO_INDEX.stop - 1}...",
              file=sys.stderr)
        draft_index = build_draft_index(session)
        print(f"  ...{len(draft_index)} drafted players indexed", file=sys.stderr)

        print("Indexing every MLB + affiliated minor-league roster...", file=sys.stderr)
        roster_index = build_roster_index(session)
        print(f"  ...{len(roster_index)} rostered players indexed", file=sys.stderr)

        rows = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(fetch_player_row, session, name, age_hint, overrides, draft_index, roster_index): name
                for name, age_hint in players.items()
            }
            done = 0
            for future in as_completed(futures):
                rows.append(future.result())
                done += 1
                if done % 50 == 0:
                    print(f"  ...{done}/{len(players)}", file=sys.stderr)

    rows.sort(key=lambda r: r["name"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "mlb_id", "season", "level",
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
