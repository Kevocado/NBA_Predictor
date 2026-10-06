"""Data modules for fetching NBA data from external sources.

`balldontlie`, `nba_api` and `odds_api` were removed here (spec section 10,
gap G7). Nothing in `src/` imported any of them outside this re-export, and:

* `odds_api` (The Odds API) was never wired -- the pipeline reads
  `sportsbook_api` only. Its predecessor called a fabricated path, and the spec
  is explicit that The Odds API is not a fallback here ($0 data spend, ESPN
  keyless plus the already-keyed RapidAPI sportsbook feed).
* `nba_api` and `balldontlie` had no production caller. ESPN keyless is the only
  real source, and no keys are spent.

`get_player_props` in `sportsbook_api` is kept despite having no production
caller: it is written against a schema *signal* (`participantKey` populated on
the market) rather than a guessed market-type name, and it answers a live
question -- whether the books have started posting player markets -- which is
worth more than the maintenance of the function.
"""

from nba_predictor.data import espn
from nba_predictor.data import injuries
from nba_predictor.data import sportsbook_api
from nba_predictor.data import team_reference

__all__ = [
    "espn",
    "injuries",
    "sportsbook_api",
    "team_reference",
]