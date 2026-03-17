import pandas as pd
import numpy as np

df_main = pd.read_csv("data/Games.csv")

# sort chronologically by game date
df_main['gameDateTimeEst'] = pd.to_datetime(df_main['gameDateTimeEst'])
df_main = df_main.sort_values('gameDateTimeEst')

# create a new column 'homeWin' which is 1 if the home team won, and 0 otherwise
df_main['homeWin'] = (df_main['homeScore'] > df_main['awayScore']).astype(int)
#print(df_main.tail(5))


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
#print(df_teamstats.tail(5))


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
                            # Rolling 10-game average of offensive rating
                            # df['home_off_rating_avg'] = (
                            #     df.groupby('home_team')['home_off_rating']
                            #     .rolling(10, min_periods=1)
                            #     .mean()
                            #     .reset_index(0, drop=True)
                            # )

                            # df['away_off_rating_avg'] = (
                            #     df.groupby('away_team')['away_off_rating']
                            #     .rolling(10, min_periods=1)
                            #     .mean()
                            #     .reset_index(0, drop=True)
                            # )

                            # # Differential feature
                            # df['off_rating_diff'] = df['home_off_rating_avg'] - df['away_off_rating_avg']
                            # print(df_main.tail(5))

# calculate rest days rest_diff
# calculate back to back games b2b_diff

df_main = df_main.sort_values('gameDateTimeEst').reset_index(drop=True)
home = df_main[['gameId', 'gameDateTimeEst', 'hometeamName']].rename(columns={'hometeamName': 'teamId'})
away = df_main[['gameId', 'gameDateTimeEst', 'awayteamName']].rename(columns={'awayteamName': 'teamId'})

all_games = pd.concat([home, away]).sort_values(['teamId', 'gameDateTimeEst']).reset_index(drop=True)
# Rest days = days since last game minus 1 (so 0 = back-to-back)
all_games['prevGameDate'] = all_games.groupby('teamId')['gameDateTimeEst'].shift(1)
all_games['restDays'] = (all_games['gameDateTimeEst'] - all_games['prevGameDate']).dt.days
all_games['isBackToBack'] = (all_games['restDays'] <= 1).astype(int)


# Merge home team stats back
home_stats = all_games.merge(
    df_main[['gameId', 'hometeamName']].rename(columns={'hometeamName': 'teamId'}),
    on=['gameId', 'teamId']
).rename(columns={'restDays': 'homeRestDays', 'isBackToBack': 'homeBackToBack'})

# Merge away team stats back
away_stats = all_games.merge(
    df_main[['gameId', 'awayteamName']].rename(columns={'awayteamName': 'teamId'}),
    on=['gameId', 'teamId']
).rename(columns={'restDays': 'awayRestDays', 'isBackToBack': 'awayBackToBack'})

df_main = df_main.merge(home_stats[['gameId', 'homeRestDays', 'homeBackToBack']], on='gameId', how='left')
df_main = df_main.merge(away_stats[['gameId', 'awayRestDays', 'awayBackToBack']], on='gameId', how='left')

df_main["restDiff"] = df_main["homeRestDays"] - df_main["awayRestDays"]
df_main["b2bDiff"] = df_main["homeBackToBack"] - df_main["awayBackToBack"]

# print(df_main.tail(5))

# feature engineering done

features = [
    'offRatingDiff',   # Home ORtg - Away ORtg
    'defRatingDiff',   # Home DRtg - Away DRtg
    'netRatingDiff',   # Home NetRating - Away NetRating
    'restDiff',         # Home Rest - Away Rest
    'b2bDiff',          # Home B2B - Away B2B
]

target = 'homeWin'  # 1 if home team wins, 0 if loses

# Example: train on games before Jan 1, 2022, test on 2022 and after
train = df_main[df_main['gameDateTimeEst'] < '2022-01-01']
test = df_main[df_main['gameDateTimeEst'] >= '2022-01-01']

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]