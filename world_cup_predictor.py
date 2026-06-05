import pandas as pd
import numpy as np

df = pd.read_csv('international_results/results.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df["date"] = pd.to_datetime(df["date"])

today = pd.Timestamp.today().normalize()
past_games = df[df["date"] < today]

def get_result(row):
    if row['home_score'] > row['away_score']:
        return "H"
    elif row['home_score'] < row['away_score']:
        return "A"
    else:
        return "D"

df['result'] = df.apply(get_result, axis=1)

def last_10_results_before_date(df, team, match_date, n=10):
    previous_matches = df[
        (((df["home_team"] == team) | (df["away_team"] == team)) &
         (df["date"] < match_date))
    ].sort_values("date").tail(n)

    score = 0

    for _, row in previous_matches.iterrows():
        if row["home_team"] == team:
            if row["home_score"] > row["away_score"]:
                score += 3
            elif row["home_score"] == row["away_score"]:
                score += 1
        else:
            if row["away_score"] > row["home_score"]:
                score += 3
            elif row["away_score"] == row["home_score"]:
                score += 1

    if len(previous_matches) == 0:
        return 0

    return score / len(previous_matches)

home_recent_points_list = []
away_recent_points_list = []
form_diff_list = []

for _, row in past_games.iterrows():
    match_date = row["date"]
    home_team = row["home_team"]
    away_team = row["away_team"]

    home_recent = last_10_results_before_date(past_games, home_team, match_date)
    away_recent = last_10_results_before_date(past_games, away_team, match_date)

    form_diff = home_recent - away_recent

    home_recent_points_list.append(home_recent)
    away_recent_points_list.append(away_recent)
    form_diff_list.append(form_diff)

past_games["home_recent_points"] = home_recent_points_list
past_games["away_recent_points"] = away_recent_points_list
past_games["form_diff"] = form_diff_list

print(past_games[[
    "date", "home_team", "away_team",
    "home_recent_points", "away_recent_points", "form_diff"
]].head(10))

