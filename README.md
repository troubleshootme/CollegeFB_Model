# College Football Prediction Model

College Football database currently contains 16 tables

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Games20
### Games21
### Games22
### Games23
## Games 24 is still needed

Columns: index|id|home_team|away_team|away_points|home_points

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### AdvancedSeasonGameStat20
### AdvancedSeasonGameStat21
### AdvancedSeasonGameStat22
### AdvancedSeasonGameStat23
### AdvancedSeasonGameStat24
AdvancedSeasonGameStat24 has to be updated regularly. It's last update was on Saturday night so it is missing games that ended late Saturday.
Columns: index|game_id|team_id|team_name|home_away|conference|points|rushingTDs|puntReturnYards|puntReturnTDs|puntReturns|passingTDs|kickReturnYards|kickReturnTDs|kickReturns|kickingPoints|interceptionYards|interceptionTDs|passesIntercepted|fumblesRecovered|totalFumbles|tacklesForLoss|defensiveTDs|tackles|sacks|qbHurries|passesDeflected|possessionTime|interceptions|fumblesLost|turnovers|totalPenaltiesYards|yardsPerRushAttempt|rushingAttempts|rushingYards|yardsPerPass|completionAttempts|netPassingYards|totalYards|fourthDownEff|thirdDownEff|firstDowns

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### fbsTeams

Columns: index|id|school|conference|division|elevation|capacity|latitude|longitude|grass|dome

-----------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------
### bets20
### bets21
### bets22
### bets23
### bets24

Columns:
bets24 has to be updated regularly. 
index|game_id|season|season_type|week|start_date|away_team|away_conference|away_score|home_team|home_conference|home_score|provider0|formatted_spread0|spread0|over_under0|provider1|formatted_spread1|spread1|over_under1|provider2|formatted_spread2|spread2|over_under2

-----------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------

### coachHist

Columns: 
index|INTEGER|0||0
1|first_name|TEXT|0||0
2|hire_date|TEXT|0||0
3|season_type|TEXT|0||0
4|year|REAL|0||0
5|games|REAL|0||0
6|losses|REAL|0||0
7|postseason_rank|REAL|0||0
8|preseason_rank|REAL|0||0
9|school|TEXT|0||0
10|sp_defense|REAL|0||0
11|sp_offense|REAL|0||0
12|sp_overall|REAL|0||0
13|games0|REAL|0||0
14|losses0|REAL|0||0
15|postseason_rank0|REAL|0||0
16|preseason_rank0|REAL|0||0
17|school0|TEXT|0||0
18|sp_defense0|REAL|0||0
19|sp_offense0|REAL|0||0
20|sp_overall0|REAL|0||0
21|games1|REAL|0||0
22|losses1|REAL|0||0
23|postseason_rank1|REAL|0||0
24|preseason_rank1|REAL|0||0
25|school1|TEXT|0||0
26|sp_defense1|REAL|0||0
27|sp_offense1|REAL|0||0
28|sp_overall1|REAL|0||0

----------------------------------------------------------------------------------------------------------------------------
