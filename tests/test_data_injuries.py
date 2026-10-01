"""data/injuries.py must never serve anything again.

The measured landmine: on main this module returned a literal fabricated
injury for a named real player ("LeBron James", Questionable, "Lower
extremity injury"), a fabricated injury history ("Ankle sprain",
2023-01-15), and _fetch_player_stats returned {} -- behind a docstring that
read like a real fetch from the official injury report. It was not merely
unused: nba_predictor/data/__init__.py imported it, so importing the data
package reached the fabrications, and features/injuries.py called into it by
default.

If any production path had used it, the site would report an injury that does
not exist, attributed to a real player. Every entry point now raises, and
these tests pin that, so restoring the old behaviour fails loudly rather than
quietly shipping a fake.

The real source is data/espn.py::get_injuries().
"""

import pytest


CALLS = [
    ("get_current_injuries", ()),
    ("get_player_injury_history", (203999,)),
    ("get_missing_player_value", (203999, 2024)),
]


@pytest.mark.parametrize("name,args", CALLS, ids=[name for name, _ in CALLS])
def test_the_module_refuses_to_answer_instead_of_inventing_data(name, args):
    from nba_predictor.data import injuries

    with pytest.raises(NotImplementedError):
        getattr(injuries, name)(*args)


def test_no_fabricated_player_name_survives_in_the_executable_source():
    # A hardcoded real name is what made this dangerous: it would read as a
    # fact about a person. Asserted on what Python would actually execute, so
    # the module's docstring is free to document what it used to do.
    import ast
    import inspect

    from nba_predictor.data import injuries

    tree = ast.parse(inspect.getsource(injuries))
    executable = "\n".join(
        ast.unparse(node)
        for node in tree.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    )
    assert "LeBron James" not in executable
    assert "Ankle sprain" not in executable
    assert "Lower extremity" not in executable
    assert "2023-01-15" not in executable


def test_the_module_no_longer_ships_the_fabricating_helpers():
    from nba_predictor.data import injuries

    for helper in ("_fetch_injury_report", "_fetch_player_stats", "_write_cache", "_read_cache"):
        assert not hasattr(injuries, helper), f"{helper} only existed to back the fabrications"


def test_the_refusal_points_readers_at_the_real_feed():
    import inspect

    from nba_predictor.data import injuries

    with pytest.raises(NotImplementedError, match="espn.get_injuries"):
        injuries.get_current_injuries()
    assert "espn.get_injuries" in inspect.getsource(injuries)


def test_the_real_feed_is_the_espn_module_not_this_one():
    from nba_predictor.data import espn

    assert hasattr(espn, "get_injuries")


def test_features_still_sum_an_injected_value_fn():
    # features/injuries.py's only real use is a caller-supplied value fn; that
    # path is untouched. Only the default (which reached the stub) now raises.
    from nba_predictor.features.injuries import missing_players_value

    total = missing_players_value([101, 102], season=2026, value_fn=lambda pid, season: 2.5)

    assert total == 5.0


def test_features_default_path_raises_rather_than_reporting_a_zero():
    # _fetch_player_stats returned {}, so this used to sum a fabricated 0.0
    # over every missing player -- a number that read as measured.
    from nba_predictor.features.injuries import missing_players_value

    with pytest.raises(NotImplementedError):
        missing_players_value([101], season=2026)
