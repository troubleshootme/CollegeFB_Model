-- Normalized, leakage-safe warehouse for the CFB prediction model.

CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER,
    season_type TEXT,
    start_date TEXT,
    completed INTEGER,
    neutral_site INTEGER,
    conference_game INTEGER,
    venue_id INTEGER,
    venue TEXT,
    home_id INTEGER,
    home_team TEXT,
    home_conference TEXT,
    home_classification TEXT,
    home_points REAL,
    home_pregame_elo REAL,
    home_postgame_elo REAL,
    home_postgame_wp REAL,
    away_id INTEGER,
    away_team TEXT,
    away_conference TEXT,
    away_classification TEXT,
    away_points REAL,
    away_pregame_elo REAL,
    away_postgame_elo REAL,
    away_postgame_wp REAL,
    excitement_index REAL
);

CREATE INDEX IF NOT EXISTS idx_games_season_week ON games (season, week);
CREATE INDEX IF NOT EXISTS idx_games_start ON games (start_date);
CREATE INDEX IF NOT EXISTS idx_games_home ON games (home_team, season);
CREATE INDEX IF NOT EXISTS idx_games_away ON games (away_team, season);

CREATE TABLE IF NOT EXISTS lines (
    game_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    spread REAL,
    spread_open REAL,
    over_under REAL,
    over_under_open REAL,
    home_moneyline REAL,
    away_moneyline REAL,
    PRIMARY KEY (game_id, provider)
);

CREATE TABLE IF NOT EXISTS team_game_stats (
    game_id INTEGER NOT NULL,
    team_id INTEGER,
    team TEXT NOT NULL,
    home_away TEXT,
    conference TEXT,
    points REAL,
    first_downs REAL,
    third_down_att REAL,
    third_down_conv REAL,
    fourth_down_att REAL,
    fourth_down_conv REAL,
    possession_seconds REAL,
    total_yards REAL,
    rushing_yards REAL,
    rushing_attempts REAL,
    net_passing_yards REAL,
    yards_per_pass REAL,
    yards_per_rush REAL,
    completions REAL,
    pass_attempts REAL,
    turnovers REAL,
    interceptions REAL,
    fumbles_lost REAL,
    sacks REAL,
    tackles_for_loss REAL,
    PRIMARY KEY (game_id, team)
);

CREATE TABLE IF NOT EXISTS ppa_games (
    game_id INTEGER NOT NULL,
    season INTEGER,
    week INTEGER,
    season_type TEXT,
    team TEXT NOT NULL,
    conference TEXT,
    opponent TEXT,
    off_overall REAL,
    off_passing REAL,
    off_rushing REAL,
    off_first_down REAL,
    off_second_down REAL,
    off_third_down REAL,
    def_overall REAL,
    def_passing REAL,
    def_rushing REAL,
    PRIMARY KEY (game_id, team)
);

CREATE TABLE IF NOT EXISTS advanced_game_stats (
    game_id INTEGER NOT NULL,
    season INTEGER,
    week INTEGER,
    season_type TEXT,
    team TEXT NOT NULL,
    opponent TEXT,
    off_success_rate REAL,
    off_explosiveness REAL,
    off_ppa REAL,
    off_stuff_rate REAL,
    off_line_yards REAL,
    off_plays INTEGER,
    def_success_rate REAL,
    def_explosiveness REAL,
    def_ppa REAL,
    def_stuff_rate REAL,
    def_line_yards REAL,
    def_plays INTEGER,
    PRIMARY KEY (game_id, team)
);

CREATE TABLE IF NOT EXISTS havoc_games (
    game_id INTEGER NOT NULL,
    season INTEGER,
    week INTEGER,
    team TEXT NOT NULL,
    opponent TEXT,
    off_havoc REAL,
    def_havoc REAL,
    PRIMARY KEY (game_id, team)
);

CREATE TABLE IF NOT EXISTS elo_weekly (
    year INTEGER NOT NULL,
    week INTEGER NOT NULL,
    season_type TEXT NOT NULL,
    team TEXT NOT NULL,
    conference TEXT,
    elo REAL,
    PRIMARY KEY (year, week, season_type, team)
);

CREATE TABLE IF NOT EXISTS core_ratings (
    year INTEGER NOT NULL,
    through_week INTEGER,
    through_season_type TEXT,
    team TEXT NOT NULL,
    conference TEXT,
    overall REAL,
    offense REAL,
    defense REAL,
    model_version TEXT,
    PRIMARY KEY (year, team, through_week)
);

CREATE TABLE IF NOT EXISTS sp_ratings (
    year INTEGER NOT NULL,
    team TEXT NOT NULL,
    rating REAL,
    ranking INTEGER,
    offense REAL,
    defense REAL,
    special_teams REAL,
    PRIMARY KEY (year, team)
);

CREATE TABLE IF NOT EXISTS talent (
    year INTEGER NOT NULL,
    team TEXT NOT NULL,
    talent REAL,
    PRIMARY KEY (year, team)
);

CREATE TABLE IF NOT EXISTS recruiting_teams (
    year INTEGER NOT NULL,
    team TEXT NOT NULL,
    rank INTEGER,
    points REAL,
    PRIMARY KEY (year, team)
);

CREATE TABLE IF NOT EXISTS returning_production (
    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    conference TEXT,
    percent_ppa REAL,
    percent_passing_ppa REAL,
    percent_rushing_ppa REAL,
    usage REAL,
    PRIMARY KEY (season, team)
);

CREATE TABLE IF NOT EXISTS venues (
    id INTEGER PRIMARY KEY,
    name TEXT,
    city TEXT,
    state TEXT,
    timezone TEXT,
    latitude REAL,
    longitude REAL,
    elevation REAL,
    capacity REAL,
    grass INTEGER,
    dome INTEGER
);

CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    school TEXT,
    conference TEXT,
    classification TEXT,
    venue_id INTEGER,
    latitude REAL,
    longitude REAL,
    elevation REAL,
    capacity REAL,
    grass INTEGER,
    dome INTEGER
);

CREATE TABLE IF NOT EXISTS coaches_seasons (
    coach_id INTEGER,
    first_name TEXT,
    last_name TEXT,
    school TEXT NOT NULL,
    year INTEGER NOT NULL,
    games INTEGER,
    wins INTEGER,
    losses INTEGER,
    sp_overall REAL,
    sp_offense REAL,
    sp_defense REAL,
    srs REAL,
    PRIMARY KEY (school, year, last_name)
);

CREATE TABLE IF NOT EXISTS wepa_season (
    year INTEGER NOT NULL,
    team TEXT NOT NULL,
    conference TEXT,
    epa_total REAL,
    epa_passing REAL,
    epa_rushing REAL,
    epa_allowed_total REAL,
    success_rate REAL,
    success_rate_allowed REAL,
    explosiveness REAL,
    explosiveness_allowed REAL,
    PRIMARY KEY (year, team)
);

CREATE TABLE IF NOT EXISTS pregame_wp (
    game_id INTEGER PRIMARY KEY,
    season INTEGER,
    week INTEGER,
    season_type TEXT,
    home_team TEXT,
    away_team TEXT,
    spread REAL,
    home_win_probability REAL
);

CREATE TABLE IF NOT EXISTS weather (
    game_id INTEGER PRIMARY KEY,
    temperature REAL,
    precipitation REAL,
    wind_speed REAL,
    indoor INTEGER
);

CREATE TABLE IF NOT EXISTS ingest_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
