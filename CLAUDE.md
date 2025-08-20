# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a college football prediction model project that uses machine learning to forecast game outcomes. The project collects data from the College Football Data API, stores it in a SQLite database, and builds predictive models using historical game statistics, betting lines, and team information.

## Database Structure

The project uses a SQLite database (`collegeFootball.db`) containing 18 tables:

### Core Tables
- **Games{year}** (2020-2024): Game results with basic info (teams, scores, dates)
- **AdvancedSeasonGameStat{year}** (2020-2024): Detailed game statistics (rushing/passing yards, turnovers, etc.)
- **bets{year}** (2020-2024): Betting lines from multiple providers (spreads, over/under)
- **fbsTeams**: Team metadata (stadium info, elevation, capacity, coordinates, grass/dome)
- **coachHist**: Coach performance history with SP+ ratings
- **talent20s**: Team talent composite scores (2020-2024) as proxy for recruiting/transfers

### Key ID Relationships
- `game_id` links Games tables to AdvancedSeasonGameStat and bets tables
- `team_id` in fbsTeams corresponds to `away_id`/`home_id` in Games tables
- AdvancedSeasonGameStat tables need regular updates (missing late Saturday games)

## Data Collection Workflow

### Jupyter Notebooks (data_collection_preparation/)
1. **import_modules.ipynb**: Common imports and API configuration
2. **get_games.ipynb**: Fetches game data from CFBD API
3. **get_advanced_box_score.ipynb**: Collects detailed game statistics
4. **get_bets.ipynb**: Gathers betting line data
5. **get_coaches.ipynb**: Coach history and performance data
6. **get_talent.ipynb**: Team talent composite scores

### API Configuration
All notebooks use College Football Data API with Bearer token authentication:
```python
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = 'API_KEY_HERE'
configuration.api_key_prefix['Authorization'] = 'Bearer'
```

## Modeling Structure

### Current Model (modelling/model_1.0)
- Proof of concept linear model
- Training data: 2020-2023 + completed 2024 games (weeks 1-8)
- Test data: Upcoming 2024 games (week 9+)
- Features include team stats, venue info, betting lines
- Uses pandas for data manipulation and merging

### Data Processing Pattern
1. Load games data and filter by completion status and week
2. Merge with fbsTeams for stadium/location features
3. Join with AdvancedSeasonGameStat for performance metrics
4. Incorporate betting data for market expectations
5. Apply feature engineering (one-hot encoding for conferences)

## Development Commands

### Database Operations
```bash
# Connect to database and list tables
sqlite3 collegeFootball.db ".tables"

# View table structure
sqlite3 collegeFootball.db ".schema TABLE_NAME"
```

### Jupyter Notebook Usage
- Notebooks expect to be run from the root directory
- Use `%run` magic commands to import shared modules
- Database connection: `sqlite3.connect('collegeFootball.db')`

## Data Update Requirements

Regular updates needed for current season:
- **AdvancedSeasonGameStat24**: Updated after each week (missing late Saturday games)
- **bets24**: Betting lines updated regularly throughout season
- **Games24**: Game results updated as games complete

## Planned Enhancements

Future development areas identified in documentation:
- Weather data integration using stadium coordinates
- Flight arrival times instead of distance for travel impact
- Stadium noise/intensity categorization
- Injury data incorporation
- ARIMA modeling for optimal historical data window
- Feature selection using PCA, Lasso, and decision trees
- Hedging strategy development
- Sharp vs. public betting movement detection