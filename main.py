import pandas as pd
import numpy as np

df_main = pd.read_csv("data/Games.csv")

# sort chronologically by game date
df_main['gameDateTimeEst'] = pd.to_datetime(df_main['gameDateTimeEst'])
df_main = df_main.sort_values('gameDateTimeEst')

# create a new column 'homeWin' which is 1 if the home team won, and 0 otherwise
df_main['homeWin'] = (df_main['homeScore'] > df_main['awayScore']).astype(int)
print(df_main.tail(5))


# calculate offensive, defensive, and net ratings for each team
df_teamstats = pd.read_csv("data/TeamStatistics.csv")

# sort chronologically by game date
df_teamstats['gameDateTimeEst'] = pd.to_datetime(df_teamstats['gameDateTimeEst'])
df_teamstats = df_teamstats.sort_values('gameDateTimeEst')

# estimate possessions based on standard formula:
# Basic Possession Formula=0.96*[(Field Goal Attempts)+(Turnovers)+0.44*(Free Throw Attempts)-(Offensive Rebounds)]
# Source: https://www.nbastuffer.com/analytics101/possession/

df_teamstats['possessions'] = 0.96 * (df_teamstats['fieldGoalsAttempted'] + df_teamstats['turnovers'] + 0.44 * df_teamstats['freeThrowsAttempted'] - df_teamstats['reboundsOffensive'])

# calculate offensive rating (points scored per 100 possessions) 
df_teamstats['offRating'] = (df_teamstats['teamScore'] / df_teamstats['possessions']) * 100

# calculate defensive rating (points allowed per 100 possessions)
df_teamstats['defRating'] = (df_teamstats['opponentScore'] / df_teamstats['possessions']) * 100

# calculate net rating (offensive rating - defensive rating)
df_teamstats['netRating'] = df_teamstats['offRating'] - df_teamstats['defRating']
print(df_teamstats.tail(5))


# combine the ratings back into the main dataframe
# home
df_main = df_main.merge(
    df_teamstats,
    left_on=["hometeamName", "gameDateTimeEst"],
    right_on=["teamName", "gameDateTimeEst"],
    how="left"
)

df_main = df_main.rename(columns={
    "offRating": "homeOffRating",
    "defRating": "homeDefRating",
    "netRating": "homeNetRating"
})

df_main = df_main.drop(columns=["teamName"])

# away
df_main = df_main.merge(
    df_teamstats,
    left_on=["awayteamName", "gameDateTimeEst"],
    right_on=["teamName", "gameDateTimeEst"],
    how="left"
)

df_main = df_main.rename(columns={
    "offRating": "awayOffRating",
    "defRating": "awayDefRating",
    "netRating": "awayNetRating"
})

df_main = df_main.drop(columns=["teamName"])

# calculate rating differences
df_main['offRatingDiff'] = df_main['homeOffRating'] - df_main['awayOffRating']
df_main['defRatingDiff'] = df_main['homeDefRating'] - df_main['awayDefRating']
df_main['netRatingDiff'] = df_main['homeNetRating'] - df_main['awayNetRating']

# calculate rolling averages of the rating differences for each team
df_main['homeOffRatingAvg'] = (
    df_main.groupby('hometeamName')['homeOffRating']
    .rolling(10, min_periods=1)
    .mean()
    .reset_index(0, drop=True)
)

print(df_main.tail(5))