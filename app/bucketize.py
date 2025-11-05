from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

Bucket = Literal["team_offense", "opp_offense", "special_teams"]


def _bucket_for_play(play: dict, team_norm: str) -> Bucket:
    pteam = (play.get("posteam") or "").strip().lower()
    dteam = (play.get("defteam") or "").strip().lower()
    st = bool(play.get("specialTeams") or play.get("specialTeamsPlayType") or play.get("special_teams"))
    if st:
        return "special_teams"
    if pteam and team_norm in pteam:
        return "team_offense"
    if dteam and team_norm in dteam:
        return "opp_offense"
    return "team_offense" if pteam else "opp_offense"


def _play_time_hint(play: dict) -> Optional[float]:
    return play.get("_aligned_sec")


def _score_weight(play: dict) -> float:
    if play.get("scoringPlay") or play.get("_score_changed"):
        return 3.0
    return 1.0


def _down_distance_weight(play: dict) -> float:
    down = play.get("down")
    to_go = play.get("distance") or play.get("yardsToGo")
    yardline = play.get("yardline_100")
    weight = 1.0
    if down in (3, 4):
        weight += 0.5
    if isinstance(to_go, (int, float)):
        if to_go <= 3:
            weight += 0.4
        elif to_go >= 10:
            weight += 0.2
    if isinstance(yardline, (int, float)) and yardline <= 20:
        weight += 0.3
    return weight


def score_play(play: dict) -> float:
    return _score_weight(play) * _down_distance_weight(play)


def build_guided_windows(
    plays: List[dict],
    *,
    team_name: str,
    period_clock_to_video,
    pre_pad: float,
    post_pad: float,
) -> Dict[Bucket, List[Tuple[float, float, float, dict]]]:
    team_norm = team_name.strip().lower()
    out: Dict[Bucket, List[Tuple[float, float, float, dict]]] = {
        "team_offense": [],
        "opp_offense": [],
        "special_teams": [],
    }

    prev_home, prev_away = None, None
    for play in plays:
        home_score = play.get("homeScore")
        away_score = play.get("awayScore")
        changed = (
            prev_home is not None
            and prev_away is not None
            and (home_score, away_score) != (prev_home, prev_away)
        )
        play["_score_changed"] = bool(changed)
        prev_home, prev_away = home_score, away_score

        period_raw = play.get("period") or play.get("quarter") or 0
        clock_val = play.get("clock")
        if isinstance(clock_val, dict):
            clock_str = clock_val.get("displayValue") or "00:00"
        else:
            clock_str = clock_val or "00:00"
        try:
            aligned = period_clock_to_video(int(period_raw), str(clock_str))
        except Exception:
            aligned = None
        play["_aligned_sec"] = aligned

    for play in plays:
        aligned = _play_time_hint(play)
        if aligned is None:
            continue
        bucket = _bucket_for_play(play, team_norm)
        start = max(0.0, float(aligned) - float(pre_pad))
        end = float(aligned) + float(post_pad)
        weight = score_play(play)
        out[bucket].append((start, end, weight, play))

    merged: Dict[Bucket, List[Tuple[float, float, float, dict]]] = {k: [] for k in out}
    for bucket, items in out.items():
        items.sort(key=lambda item: item[0])
        for start, end, weight, play in items:
            if merged[bucket] and start - merged[bucket][-1][1] <= 2.0:
                prev_start, prev_end, prev_weight, prev_play = merged[bucket][-1]
                merged[bucket][-1] = (
                    prev_start,
                    max(prev_end, end),
                    max(prev_weight, weight),
                    play if weight >= prev_weight else prev_play,
                )
            else:
                merged[bucket].append((start, end, weight, play))
    return merged
