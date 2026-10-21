"""'Key out' for an upcoming game: players the injury report has OUT who normally play real minutes."""
from __future__ import annotations

KEY_MINUTES = 20.0        # a starter or a main rotation piece; a 6-minute bench player sitting is not news
PER_TEAM = 2


def key_out_for_game(game: dict, rosters: dict[str, list[dict]] | None, minutes_by_player: dict[str, float]) -> list[dict]:
    """[{team, player_name}], home team first, at most PER_TEAM per team, most minutes first. Empty when the game is
    finished or the rosters are unknown (unknown is not "nobody is out")."""
    if game.get("completed") or not rosters:
        return []
    out = []
    for team in (game.get("home_team"), game.get("away_team")):
        outs = [p for p in rosters.get(team, []) if p.get("injury_status") == "Out" and minutes_by_player.get(p["player_id"], 0.0) >= KEY_MINUTES]
        outs.sort(key=lambda p: -minutes_by_player[p["player_id"]])
        out += [{"team": team, "player_name": p["player_name"]} for p in outs[:PER_TEAM]]
    return out
