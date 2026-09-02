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
    home_q1 REAL,
    home_q2 REAL,
    home_q3 REAL,
    home_q4 REAL,
    home_pregame_elo REAL,
    home_postgame_elo REAL,
    home_postgame_wp REAL,
    away_id INTEGER,
    away_team TEXT,
    away_conference TEXT,
    away_classification TEXT,
    away_points REAL,
    away_q1 REAL,
    away_q2 REAL,
    away_q3 REAL,
    away_q4 REAL,
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

CREATE TABLE IF NOT EXISTS fpi_ratings (
    year INTEGER NOT NULL,
    team TEXT NOT NULL,
    conference TEXT,
    fpi REAL,
    fpi_rank INTEGER,
    offense REAL,
    defense REAL,
    special_teams REAL,
    overall_eff REAL,
    source TEXT NOT NULL DEFAULT 'cfbd',
    as_of TEXT,
    PRIMARY KEY (year, team, source)
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

CREATE TABLE IF NOT EXISTS player_season_stats (
    season INTEGER NOT NULL,
    player_id TEXT,
    player TEXT NOT NULL,
    position TEXT,
    team TEXT NOT NULL,
    conference TEXT,
    category TEXT NOT NULL DEFAULT 'all',
    games REAL,
    attempts REAL,
    completions REAL,
    yards REAL,
    touchdowns REAL,
    interceptions REAL,
    yards_per_attempt REAL,
    yards_per_carry REAL,
    rating REAL,
    PRIMARY KEY (season, player, team, category)
);

CREATE TABLE IF NOT EXISTS player_ppa (
    season INTEGER NOT NULL,
    player_id TEXT,
    player TEXT NOT NULL,
    position TEXT,
    team TEXT NOT NULL,
    conference TEXT,
    average_ppa REAL,
    passing REAL,
    rushing REAL,
    usage REAL,
    PRIMARY KEY (season, player, team)
);

CREATE TABLE IF NOT EXISTS transfer_portal (
    season INTEGER NOT NULL,
    player_id TEXT,
    player TEXT NOT NULL,
    position TEXT,
    origin TEXT,
    destination TEXT,
    stars REAL,
    eligibility TEXT,
    transfer_date TEXT,
    PRIMARY KEY (season, player, origin)
);

CREATE TABLE IF NOT EXISTS injury_reports (
    snapshot_id TEXT NOT NULL,
    as_of TEXT NOT NULL,
    season INTEGER,
    week INTEGER,
    espn_team_id TEXT,
    team TEXT,
    team_raw TEXT NOT NULL,
    player TEXT,
    player_id TEXT NOT NULL,
    position TEXT,
    status TEXT,
    status_weight REAL,
    position_weight REAL,
    load REAL,
    report_date TEXT,
    comment TEXT,
    source TEXT NOT NULL DEFAULT 'espn',
    PRIMARY KEY (snapshot_id, player_id, team_raw)
);

CREATE INDEX IF NOT EXISTS idx_injury_as_of ON injury_reports (as_of, team);
CREATE INDEX IF NOT EXISTS idx_player_stats_team ON player_season_stats (season, team);
CREATE INDEX IF NOT EXISTS idx_portal_season ON transfer_portal (season, destination);

CREATE TABLE IF NOT EXISTS ingest_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
