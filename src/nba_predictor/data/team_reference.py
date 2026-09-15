from dataclasses import dataclass


@dataclass(frozen=True)
class TeamInfo:
    nba_api_id: int
    abbreviation: str
    name: str
    conference: str
    division: str
    arena_lat: float
    arena_lon: float
    timezone: str
    altitude_ft: int


TEAMS: tuple[TeamInfo, ...] = (
    # --- Eastern Conference ---
    # Atlantic
    TeamInfo(1610612738, "BOS", "Boston Celtics", "East", "Atlantic", 42.3662, -71.0621, "America/New_York", 0),
    TeamInfo(1610612751, "BKN", "Brooklyn Nets", "East", "Atlantic", 40.6826, -73.9754, "America/New_York", 0),
    TeamInfo(1610612752, "NYK", "New York Knicks", "East", "Atlantic", 40.7505, -73.9934, "America/New_York", 0),
    TeamInfo(1610612755, "PHI", "Philadelphia 76ers", "East", "Atlantic", 39.9012, -75.1720, "America/New_York", 0),
    TeamInfo(1610612761, "TOR", "Toronto Raptors", "East", "Atlantic", 43.6435, -79.3791, "America/Toronto", 0),
    # Central
    TeamInfo(1610612741, "CHI", "Chicago Bulls", "East", "Central", 41.8807, -87.6742, "America/Chicago", 0),
    TeamInfo(1610612739, "CLE", "Cleveland Cavaliers", "East", "Central", 41.4965, -81.6882, "America/New_York", 0),
    TeamInfo(1610612765, "DET", "Detroit Pistons", "East", "Central", 42.3410, -83.0550, "America/Detroit", 0),
    TeamInfo(1610612754, "IND", "Indiana Pacers", "East", "Central", 39.7640, -86.1555, "America/Indiana/Indianapolis", 0),
    TeamInfo(1610612749, "MIL", "Milwaukee Bucks", "East", "Central", 43.0451, -87.9172, "America/Chicago", 0),
    # Southeast
    TeamInfo(1610612737, "ATL", "Atlanta Hawks", "East", "Southeast", 33.7573, -84.3963, "America/New_York", 0),
    TeamInfo(1610612766, "CHA", "Charlotte Hornets", "East", "Southeast", 35.2251, -80.8392, "America/New_York", 0),
    TeamInfo(1610612748, "MIA", "Miami Heat", "East", "Southeast", 25.7814, -80.1870, "America/New_York", 0),
    TeamInfo(1610612753, "ORL", "Orlando Magic", "East", "Southeast", 28.5392, -81.3839, "America/New_York", 0),
    TeamInfo(1610612764, "WAS", "Washington Wizards", "East", "Southeast", 38.8981, -77.0209, "America/New_York", 0),
    # --- Western Conference ---
    # Northwest
    TeamInfo(1610612743, "DEN", "Denver Nuggets", "West", "Northwest", 39.7487, -105.0077, "America/Denver", 5280),
    TeamInfo(1610612750, "MIN", "Minnesota Timberwolves", "West", "Northwest", 44.9795, -93.2760, "America/Chicago", 0),
    TeamInfo(1610612760, "OKC", "Oklahoma City Thunder", "West", "Northwest", 35.4634, -97.5151, "America/Chicago", 0),
    TeamInfo(1610612757, "POR", "Portland Trail Blazers", "West", "Northwest", 45.5316, -122.6668, "America/Los_Angeles", 0),
    TeamInfo(1610612762, "UTA", "Utah Jazz", "West", "Northwest", 40.7683, -111.9011, "America/Denver", 0),
    # Pacific
    TeamInfo(1610612744, "GSW", "Golden State Warriors", "West", "Pacific", 37.7680, -122.3877, "America/Los_Angeles", 0),
    TeamInfo(1610612746, "LAC", "LA Clippers", "West", "Pacific", 33.9450, -118.3410, "America/Los_Angeles", 0),
    TeamInfo(1610612747, "LAL", "Los Angeles Lakers", "West", "Pacific", 34.0430, -118.2673, "America/Los_Angeles", 0),
    TeamInfo(1610612756, "PHX", "Phoenix Suns", "West", "Pacific", 33.4457, -112.0712, "America/Phoenix", 0),
    TeamInfo(1610612758, "SAC", "Sacramento Kings", "West", "Pacific", 38.5802, -121.4997, "America/Los_Angeles", 0),
    # Southwest
    TeamInfo(1610612742, "DAL", "Dallas Mavericks", "West", "Southwest", 32.7905, -96.8103, "America/Chicago", 0),
    TeamInfo(1610612745, "HOU", "Houston Rockets", "West", "Southwest", 29.7508, -95.3621, "America/Chicago", 0),
    TeamInfo(1610612763, "MEM", "Memphis Grizzlies", "West", "Southwest", 35.1382, -90.0505, "America/Chicago", 0),
    TeamInfo(1610612740, "NOP", "New Orleans Pelicans", "West", "Southwest", 29.9490, -90.0821, "America/Chicago", 0),
    TeamInfo(1610612759, "SAS", "San Antonio Spurs", "West", "Southwest", 29.4269, -98.4375, "America/Chicago", 0),
)

_BY_ABBREVIATION = {t.abbreviation: t for t in TEAMS}


def get_team(abbreviation: str) -> TeamInfo:
    return _BY_ABBREVIATION[abbreviation]
