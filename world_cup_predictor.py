import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from scipy.stats import poisson
from THIRD_PLACE_LOOKUP import THIRD_PLACE_LOOKUP
from collections import Counter

df = pd.read_csv('international_results/results.csv')
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

today = pd.Timestamp.today().normalize()

df["neutral"] = df["neutral"].fillna(False).astype(int)

past_games = df[df["date"] < today].copy().tail(500)

def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def actual_score(home_score, away_score):
    if home_score > away_score:
        return 1.0, 0.0
    elif home_score < away_score:
        return 0.0, 1.0
    else:
        return 0.5, 0.5

def add_elo_features(df, k=30, base_rating=1500, home_advantage=100):
    ratings = defaultdict(lambda: base_rating)

    home_elo_list = []
    away_elo_list = []
    elo_diff_list = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_elo = ratings[home]
        away_elo = ratings[away]

        home_elo_list.append(home_elo)
        away_elo_list.append(away_elo)
        elo_diff_list.append(home_elo - away_elo)

        adj_home_elo = home_elo + (0 if row["neutral"] else home_advantage)

        exp_home = expected_score(adj_home_elo, away_elo)
        exp_away = 1 - exp_home

        act_home, act_away = actual_score(row["home_score"], row["away_score"])

        ratings[home] = home_elo + k * (act_home - exp_home)
        ratings[away] = away_elo + k * (act_away - exp_away)

    df = df.copy()
    df["home_elo"] = home_elo_list
    df["away_elo"] = away_elo_list
    df["elo_diff"] = elo_diff_list

    return df, ratings

def get_result(row):
    if row['home_score'] > row['away_score']:
        return "H"
    elif row['home_score'] < row['away_score']:
        return "A"
    else:
        return "D"

df['result'] = df.apply(get_result, axis=1)
df, current_elos = add_elo_features(df)

past_games = df[df["date"] < today].copy().tail(500)
past_games['result'] = past_games.apply(get_result, axis=1)

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
        return 0, previous_matches

    return score / len(previous_matches), previous_matches

home_recent_points_list = []
away_recent_points_list = []
form_diff_list = []

for _, row in past_games.iterrows():
    match_date = row["date"]
    home_team = row["home_team"]
    away_team = row["away_team"]

    home_recent, _ = last_10_results_before_date(past_games, home_team, match_date)
    away_recent, _ = last_10_results_before_date(past_games, away_team, match_date)

    form_diff = home_recent - away_recent

    home_recent_points_list.append(home_recent)
    away_recent_points_list.append(away_recent)
    form_diff_list.append(form_diff)

past_games["home_recent_points"] = home_recent_points_list
past_games["away_recent_points"] = away_recent_points_list
past_games["form_diff"] = form_diff_list

X_score = past_games[[
    "home_recent_points",
    "away_recent_points",
    "form_diff",
    "neutral",
    "home_elo",
    "away_elo",
    "elo_diff"
]]

y_home_goals = past_games["home_score"]
y_away_goals = past_games["away_score"]

home_goals_model = Pipeline([
    ("scaler", StandardScaler()),
    ("poisson", PoissonRegressor(alpha=0.1, max_iter=1000))
])

away_goals_model = Pipeline([
    ("scaler", StandardScaler()),
    ("poisson", PoissonRegressor(alpha=0.1, max_iter=1000))
])

home_goals_model.fit(X_score, y_home_goals)
away_goals_model.fit(X_score, y_away_goals)

y = past_games["result"]
X = past_games[[
    "home_recent_points",
    "away_recent_points",
    "form_diff",
    "neutral",
    "home_elo",
    "away_elo",
    "elo_diff"
]]

split_index = int(len(past_games) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=5000))
])

model.fit(X_train, y_train)

#print("Model class order:", model.classes_)

def predict_match(team1, team2, match_date, neutral=True):
    home_recent, home_matches = last_10_results_before_date(past_games, team1, match_date)
    away_recent, away_matches = last_10_results_before_date(past_games, team2, match_date)
    form_diff = home_recent - away_recent

    home_elo = current_elos.get(team1, 1500)
    away_elo = current_elos.get(team2, 1500)
    elo_diff = home_elo - away_elo

    input_df = pd.DataFrame({
        "home_recent_points": [home_recent],
        "away_recent_points": [away_recent],
        "form_diff": [form_diff],
        "neutral": [neutral],
        "home_elo": [home_elo],
        "away_elo": [away_elo],
        "elo_diff": [elo_diff],
    })

    probs = model.predict_proba(input_df)[0]
    class_probs = dict(zip(model.classes_, probs))

    p_H = class_probs["H"]
    p_D = class_probs["D"]
    p_A = class_probs["A"]

    return {
        "team1_win_prob": p_H,
        "draw_prob": p_D,
        "team2_win_prob": p_A,
        "team1_elo": home_elo,
        "team2_elo": away_elo,
        "elo_diff": elo_diff,
        "team1_last_10": home_matches,
        "team2_last_10": away_matches,
    }

def predict_expected_goals(team1, team2, match_date, neutral=1):
    home_recent, home_matches = last_10_results_before_date(past_games, team1, match_date)
    away_recent, away_matches = last_10_results_before_date(past_games, team2, match_date)

    form_diff = home_recent - away_recent

    home_elo = current_elos.get(team1, 1500)
    away_elo = current_elos.get(team2, 1500)
    elo_diff = home_elo - away_elo

    input_df = pd.DataFrame({
        "home_recent_points": [home_recent],
        "away_recent_points": [away_recent],
        "form_diff": [form_diff],
        "neutral": [neutral],
        "home_elo": [home_elo],
        "away_elo": [away_elo],
        "elo_diff": [elo_diff],
    })

    expected_home_goals = home_goals_model.predict(input_df)[0]
    expected_away_goals = away_goals_model.predict(input_df)[0]

    return {
        "team1_xg": expected_home_goals,
        "team2_xg": expected_away_goals,
        "team1_last_10": home_matches,
        "team2_last_10": away_matches,
        "team1_elo": home_elo,
        "team2_elo": away_elo,
    }

def predict_score_matrix(team1, team2, match_date, neutral=1, max_goals=5):
    result = predict_expected_goals(team1, team2, match_date, neutral)

    lambda_home = result["team1_xg"]
    lambda_away = result["team2_xg"]

    score_matrix = np.zeros((max_goals + 1, max_goals + 1))

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            p_home = poisson.pmf(home_goals, lambda_home)
            p_away = poisson.pmf(away_goals, lambda_away)
            score_matrix[home_goals, away_goals] = p_home * p_away

    return score_matrix, result

def predict_most_likely_score(team1, team2, match_date, neutral=1, max_goals=5):
    score_matrix, result = predict_score_matrix(team1, team2, match_date, neutral, max_goals)

    max_index = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    home_goals, away_goals = max_index
    prob = score_matrix[home_goals, away_goals]

    return {
        "team1": team1,
        "team2": team2,
        "predicted_score": f"{home_goals}-{away_goals}",
        "score_probability": prob,
        "team1_xg": result["team1_xg"],
        "team2_xg": result["team2_xg"],
        "score_matrix": score_matrix
    }

def simulate_match(team1, team2, match_date, neutral=1):
    result = predict_expected_goals(team1, team2, match_date, neutral)

    lambda_home = result["team1_xg"]
    lambda_away = result["team2_xg"]

    home_goals = np.random.poisson(lambda_home)
    away_goals = np.random.poisson(lambda_away)

    if home_goals > away_goals:
        outcome = team1
    elif away_goals > home_goals:
        outcome = team2
    else:
        outcome = "Draw"

    return {
        "team1": team1,
        "team2": team2,
        "team1_goals": home_goals,
        "team2_goals": away_goals,
        "winner": outcome
    }

def simulate_match_many(team1, team2, match_date, neutral=1, n_simulations=10000):
    team1_wins = 0
    team2_wins = 0
    draws = 0

    score_counts = {}

    for _ in range(n_simulations):
        sim = simulate_match(team1, team2, match_date, neutral)

        score = f"{sim['team1_goals']}-{sim['team2_goals']}"
        score_counts[score] = score_counts.get(score, 0) + 1

        if sim["winner"] == team1:
            team1_wins += 1
        elif sim["winner"] == team2:
            team2_wins += 1
        else:
            draws += 1

    most_common_score = max(score_counts, key=score_counts.get)

    return {
        "team1_win_prob": team1_wins / n_simulations,
        "draw_prob": draws / n_simulations,
        "team2_win_prob": team2_wins / n_simulations,
        "most_common_score": most_common_score,
        "most_common_score_prob": score_counts[most_common_score] / n_simulations,
        "score_counts": score_counts
    }

def simulate_whole_group(group_teams, fixtures, match_dates):
    table = {
        team: {"points": 0, "gf": 0, "ga": 0, "gd": 0}
        for team in group_teams
    }
    for fixture in fixtures:
        home_team = fixture["team1"]
        away_team = fixture["team2"]
        simmed = simulate_match(home_team, away_team, fixture["match_date"], fixture["neutral"])
        if simmed["winner"] == home_team:
            table[home_team]["points"] += 3
        elif simmed["winner"] == away_team:
            table[away_team]["points"] += 3
        else:
            table[home_team]["points"] += 1
            table[away_team]["points"] += 1
        table[home_team]["gf"] += simmed["team1_goals"]
        table[home_team]["ga"] += simmed["team2_goals"]
        table[away_team]["gf"] += simmed["team2_goals"]
        table[away_team]["ga"] += simmed["team1_goals"]

        table[home_team]["gd"] = table[home_team]["gf"] - table[home_team]["ga"]
        table[away_team]["gd"] = table[away_team]["gf"] - table[away_team]["ga"]

    standings = sorted(
        table.items(),
        key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["gf"]),
        reverse=True
    )

    return standings

def simulate_knockout_match(team1, team2, match_date, neutral=1):
    result = predict_expected_goals(team1, team2, match_date, neutral)

    team1_goals = np.random.poisson(result["team1_xg"])
    team2_goals = np.random.poisson(result["team2_xg"])

    if team1_goals > team2_goals:
        winner = team1
    elif team2_goals > team1_goals:
        winner = team2
    else:
        winner = np.random.choice([team1, team2])  # simple tiebreak placeholder

    return {
        "team1": team1,
        "team2": team2,
        "team1_goals": int(team1_goals),
        "team2_goals": int(team2_goals),
        "winner": winner
    }

def simulate_round(fixtures, match_date, neutral=1):
    results = []

    for team1, team2 in fixtures:
        match_result = simulate_knockout_match(team1, team2, match_date, neutral)
        results.append(match_result)

    return results

def pair_next_round(results):
    winners = [result["winner"] for result in results]
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]



y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

"""
team1 = "England"
team2 = "France"

match_date = pd.Timestamp("2026-06-20")
result = predict_match(team1, team2, match_date, neutral=1)

print("Probabilities:")
print(f"{team1} win: {result['team1_win_prob']:.3f}")
print(f"Draw:        {result['draw_prob']:.3f}")
print(f"{team2} win: {result['team2_win_prob']:.3f}")

print("\nElo:")
print(f"{team1} Elo: {result['team1_elo']:.1f}")
print(f"{team2} Elo: {result['team2_elo']:.1f}")
print(f"Elo diff:    {result['elo_diff']:.1f}")

print(f"\n{team1} last matches:")
print(result["team1_last_10"][["date", "home_team", "away_team", "home_score", "away_score"]].to_string(index=False))

print(f"\n{team2} last matches:")
print(result["team2_last_10"][["date", "home_team", "away_team", "home_score", "away_score"]].to_string(index=False))

score_result = predict_most_likely_score(team1, team2, match_date, neutral=1, max_goals=5)

print("\nExpected goals:")
print(f"{team1} xG: {score_result['team1_xg']:.2f}")
print(f"{team2} xG: {score_result['team2_xg']:.2f}")

print("\nMost likely exact score:")
print(f"{team1} vs {team2}: {score_result['predicted_score']}")
print(f"Probability: {score_result['score_probability']:.3f}")

sim_result = simulate_match(team1, team2, match_date, neutral=1)

print("\nOne simulated match:")
print(f"{sim_result['team1']} {sim_result['team1_goals']} - {sim_result['team2_goals']} {sim_result['team2']}")
print(f"Winner: {sim_result['winner']}")

many_result = simulate_match_many(team1, team2, match_date, neutral=1, n_simulations=10000)

print("\nMonte Carlo simulation:")
print(f"{team1} win: {many_result['team1_win_prob']:.3f}")
print(f"Draw:        {many_result['draw_prob']:.3f}")
print(f"{team2} win: {many_result['team2_win_prob']:.3f}")
print(f"Most common score: {many_result['most_common_score']}")
print(f"Score probability: {many_result['most_common_score_prob']:.3f}")


group_teams = ["Mexico", "South Africa", "South Korea", "Czech Republic"]
fixtures = [
    {"team1": "Mexico", "team2": "South Africa", "match_date": pd.Timestamp("2026-06-11"), "neutral": 1},
    {"team1": "South Korea", "team2": "Czech Republic", "match_date": pd.Timestamp("2026-06-11"), "neutral": 1},
    {"team1": "Czech Republic", "team2": "South Africa", "match_date": pd.Timestamp("2026-06-16"), "neutral": 1},
    {"team1": "Mexico", "team2": "South Korea", "match_date": pd.Timestamp("2026-06-16"), "neutral": 1},
    {"team1": "Czech Republic", "team2": "Mexico", "match_date": pd.Timestamp("2026-06-21"), "neutral": 1},
    {"team1": "South Africa", "team2": "South Korea", "match_date": pd.Timestamp("2026-06-21"), "neutral": 1},
]

group_result = simulate_whole_group(group_teams, fixtures)
for position, (team, stats) in enumerate(group_result, start=1):
    print(position, team, stats)
"""
"""
all_groups = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

group_dates = {
    "A": [pd.Timestamp("2026-06-11"), pd.Timestamp("2026-06-18"), pd.Timestamp("2026-06-24")],
    "B": [pd.Timestamp("2026-06-12"), pd.Timestamp("2026-06-18"), pd.Timestamp("2026-06-24")],
    "C": [pd.Timestamp("2026-06-13"), pd.Timestamp("2026-06-19"), pd.Timestamp("2026-06-24")],
    "D": [pd.Timestamp("2026-06-12"), pd.Timestamp("2026-06-19"), pd.Timestamp("2026-06-25")],
    "E": [pd.Timestamp("2026-06-14"), pd.Timestamp("2026-06-20"), pd.Timestamp("2026-06-25")],
    "F": [pd.Timestamp("2026-06-14"), pd.Timestamp("2026-06-20"), pd.Timestamp("2026-06-25")],
    "G": [pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-21"), pd.Timestamp("2026-06-26")],
    "H": [pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-21"), pd.Timestamp("2026-06-26")],
    "I": [pd.Timestamp("2026-06-16"), pd.Timestamp("2026-06-22"), pd.Timestamp("2026-06-26")],
    "J": [pd.Timestamp("2026-06-16"), pd.Timestamp("2026-06-22"), pd.Timestamp("2026-06-27")],
    "K": [pd.Timestamp("2026-06-17"), pd.Timestamp("2026-06-23"), pd.Timestamp("2026-06-27")],
    "L": [pd.Timestamp("2026-06-17"), pd.Timestamp("2026-06-23"), pd.Timestamp("2026-06-27")],
}

def build_group_fixtures(group_teams, dates):
    t1, t2, t3, t4 = group_teams
    d1, d2, d3 = dates

    fixtures = [
        {"team1": t1, "team2": t2, "match_date": d1, "neutral": 1},
        {"team1": t3, "team2": t4, "match_date": d1, "neutral": 1},
        {"team1": t1, "team2": t3, "match_date": d2, "neutral": 1},
        {"team1": t4, "team2": t2, "match_date": d2, "neutral": 1},
        {"team1": t4, "team2": t1, "match_date": d3, "neutral": 1},
        {"team1": t2, "team2": t3, "match_date": d3, "neutral": 1},
    ]

    return fixtures

all_group_results = {}

for group_name, teams in all_groups.items():
    dates = group_dates[group_name]
    fixtures = build_group_fixtures(teams, dates)

    group_result = simulate_whole_group(teams, fixtures, None)
    all_group_results[group_name] = group_result

for group_name, result in all_group_results.items():
    print(f"\nGroup {group_name}")
    for position, (team, stats) in enumerate(result, start=1):
        print(position, team, stats)

qualified_teams = {}

for group_name, result in all_group_results.items():
    winner = result[0][0]
    runner_up = result[1][0]
    qualified_teams[group_name] = {
        "winner": winner,
        "runner_up": runner_up
    }

third_place_teams = []

for group_name, result in all_group_results.items():
    team, stats = result[2]

    third_place_teams.append({
        "group": group_name,
        "team": team,
        "points": stats["points"],
        "gf": stats["gf"],
        "ga": stats["ga"],
        "gd": stats["gd"]
    })

third_place_teams = sorted(
    third_place_teams,
    key=lambda x: (x["points"], x["gd"], x["gf"]),
    reverse=True
)

qualified_third_placed = third_place_teams[:8]
eliminated_third_placed = third_place_teams[8:]

print("\nBest 8 third-placed teams that qualify:")
for team_data in qualified_third_placed:
    print(f"Group {team_data['group']} - {team_data['team']}")

best_third_place_teams = [team_data["team"] for team_data in qualified_third_placed]

qualified_teams = []

for group_name, result in all_group_results.items():
    winner = result[0][0]
    runner_up = result[1][0]
    qualified_teams.append(winner)
    qualified_teams.append(runner_up)

for team_data in qualified_third_placed:
    qualified_teams.append(team_data["team"])

group_winners = {}
group_runners_up = {}

for group_name, result in all_group_results.items():
    group_winners[group_name] = result[0][0]
    group_runners_up[group_name] = result[1][0]

qualified_third_by_group = {
    team_data["group"]: team_data["team"]
    for team_data in qualified_third_placed
}

qualified_key = ''.join(sorted(team_data["group"] for team_data in qualified_third_placed))
print("\nQualified third-place key:", qualified_key)

slot_to_group_slot = THIRD_PLACE_LOOKUP[qualified_key]

third_place_slot_teams = {
    slot: qualified_third_by_group[group_slot[1]]
    for slot, group_slot in slot_to_group_slot.items()
}

fixtures = [
    (group_runners_up["A"], group_runners_up["B"]),
    (group_winners["E"], third_place_slot_teams["1E"]),
    (group_winners["F"], group_runners_up["C"]),
    (group_winners["C"], group_runners_up["F"]),
    (group_winners["I"], third_place_slot_teams["1I"]),
    (group_runners_up["E"], group_runners_up["I"]),
    (group_winners["A"], third_place_slot_teams["1A"]),
    (group_winners["L"], third_place_slot_teams["1L"]),
    (group_winners["D"], third_place_slot_teams["1D"]),
    (group_winners["G"], third_place_slot_teams["1G"]),
    (group_runners_up["K"], group_runners_up["L"]),
    (group_winners["H"], group_runners_up["J"]),
    (group_winners["B"], third_place_slot_teams["1B"]),
    (group_winners["J"], group_runners_up["H"]),
    (group_winners["K"], third_place_slot_teams["1K"]),
    (group_runners_up["D"], group_runners_up["G"]),
]

all_teams_in_fixtures = [team for match in fixtures for team in match]
print("Unique teams in Round of 32:", len(set(all_teams_in_fixtures)))

print("\nRound of 32 fixtures:")
for i, (team1, team2) in enumerate(fixtures, start=1):
    print(f"{i}. {team1} vs {team2}")

def build_round_of_32_from_groups(all_group_results):
    group_winners = {}
    group_runners_up = {}

    for group_name, result in all_group_results.items():
        group_winners[group_name] = result[0][0]
        group_runners_up[group_name] = result[1][0]

    third_place_teams = []

    for group_name, result in all_group_results.items():
        team, stats = result[2]
        third_place_teams.append({
            "group": group_name,
            "team": team,
            "points": stats["points"],
            "gf": stats["gf"],
            "ga": stats["ga"],
            "gd": stats["gd"],
        })

    third_place_teams = sorted(
        third_place_teams,
        key=lambda x: (x["points"], x["gd"], x["gf"]),
        reverse=True
    )

    qualified_third_placed = third_place_teams[:8]

    qualified_third_by_group = {
        team_data["group"]: team_data["team"]
        for team_data in qualified_third_placed
    }

    qualified_key = ''.join(
        sorted(team_data["group"] for team_data in qualified_third_placed)
    )

    slot_to_group_slot = THIRD_PLACE_LOOKUP[qualified_key]

    third_place_slot_teams = {
        slot: qualified_third_by_group[group_slot[1]]
        for slot, group_slot in slot_to_group_slot.items()
    }

    round_of_32_fixtures = [
        (group_runners_up["A"], group_runners_up["B"]),
        (group_winners["E"], third_place_slot_teams["1E"]),
        (group_winners["F"], group_runners_up["C"]),
        (group_winners["C"], group_runners_up["F"]),
        (group_winners["I"], third_place_slot_teams["1I"]),
        (group_runners_up["E"], group_runners_up["I"]),
        (group_winners["A"], third_place_slot_teams["1A"]),
        (group_winners["L"], third_place_slot_teams["1L"]),
        (group_winners["D"], third_place_slot_teams["1D"]),
        (group_winners["G"], third_place_slot_teams["1G"]),
        (group_runners_up["K"], group_runners_up["L"]),
        (group_winners["H"], group_runners_up["J"]),
        (group_winners["B"], third_place_slot_teams["1B"]),
        (group_winners["J"], group_runners_up["H"]),
        (group_winners["K"], third_place_slot_teams["1K"]),
        (group_runners_up["D"], group_runners_up["G"]),
    ]

    return round_of_32_fixtures, qualified_third_placed

def get_winners(results):
    return [result["winner"] for result in results]

def build_round_of_16_fixtures(round_of_32_results):
    winners = get_winners(round_of_32_results)
    return [
        (winners[0], winners[2]),   # W1 vs W3
        (winners[1], winners[4]),   # W2 vs W5
        (winners[5], winners[7]),   # W6 vs W8
        (winners[6], winners[3]),   # W7 vs W4
        (winners[8], winners[10]),  # W9 vs W11
        (winners[9], winners[11]),  # W10 vs W12
        (winners[12], winners[15]), # W13 vs W16
        (winners[13], winners[14]), # W14 vs W15
    ]

def build_quarterfinal_fixtures(round_of_16_results):
    winners = get_winners(round_of_16_results)
    return [
        (winners[0], winners[1]),   # W17 vs W18
        (winners[4], winners[5]),   # W21 vs W22
        (winners[2], winners[3]),   # W19 vs W20
        (winners[6], winners[7]),   # W23 vs W24
    ]

def build_semifinal_fixtures(quarterfinal_results):
    winners = get_winners(quarterfinal_results)
    return [
        (winners[0], winners[1]),   # W25 vs W26
        (winners[2], winners[3]),   # W27 vs W28
    ]

def build_final_fixtures(semifinal_results):
    winners = get_winners(semifinal_results)
    return [
        (winners[0], winners[1])    # W29 vs W30
    ]

def print_round_results(round_name, results):
    print(f"\n{round_name}")
    for i, result in enumerate(results, start=1):
        print(
            f"{i}. {result['team1']} {result['team1_goals']} - "
            f"{result['team2_goals']} {result['team2']} | "
            f"Winner: {result['winner']}"
        )

round_of_32_results = simulate_round(fixtures, pd.Timestamp("2026-06-28"))
print_round_results("Round of 32", round_of_32_results)

round_of_16_fixtures = build_round_of_16_fixtures(round_of_32_results)
round_of_16_results = simulate_round(round_of_16_fixtures, pd.Timestamp("2026-07-04"))
print_round_results("Round of 16", round_of_16_results)

quarterfinal_fixtures = build_quarterfinal_fixtures(round_of_16_results)
quarterfinal_results = simulate_round(quarterfinal_fixtures, pd.Timestamp("2026-07-09"))
print_round_results("Quarterfinals", quarterfinal_results)

semifinal_fixtures = build_semifinal_fixtures(quarterfinal_results)
semifinal_results = simulate_round(semifinal_fixtures, pd.Timestamp("2026-07-14"))
print_round_results("Semifinals", semifinal_results)

final_fixtures = build_final_fixtures(semifinal_results)
final_results = simulate_round(final_fixtures, pd.Timestamp("2026-07-19"))
print_round_results("Final", final_results)

champion = final_results[0]["winner"]
print("\nChampion:", champion)
"""
all_groups = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

group_dates = {
    "A": [pd.Timestamp("2026-06-11"), pd.Timestamp("2026-06-18"), pd.Timestamp("2026-06-24")],
    "B": [pd.Timestamp("2026-06-12"), pd.Timestamp("2026-06-18"), pd.Timestamp("2026-06-24")],
    "C": [pd.Timestamp("2026-06-13"), pd.Timestamp("2026-06-19"), pd.Timestamp("2026-06-24")],
    "D": [pd.Timestamp("2026-06-12"), pd.Timestamp("2026-06-19"), pd.Timestamp("2026-06-25")],
    "E": [pd.Timestamp("2026-06-14"), pd.Timestamp("2026-06-20"), pd.Timestamp("2026-06-25")],
    "F": [pd.Timestamp("2026-06-14"), pd.Timestamp("2026-06-20"), pd.Timestamp("2026-06-25")],
    "G": [pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-21"), pd.Timestamp("2026-06-26")],
    "H": [pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-21"), pd.Timestamp("2026-06-26")],
    "I": [pd.Timestamp("2026-06-16"), pd.Timestamp("2026-06-22"), pd.Timestamp("2026-06-26")],
    "J": [pd.Timestamp("2026-06-16"), pd.Timestamp("2026-06-22"), pd.Timestamp("2026-06-27")],
    "K": [pd.Timestamp("2026-06-17"), pd.Timestamp("2026-06-23"), pd.Timestamp("2026-06-27")],
    "L": [pd.Timestamp("2026-06-17"), pd.Timestamp("2026-06-23"), pd.Timestamp("2026-06-27")],
}

def build_group_fixtures(group_teams, dates):
    t1, t2, t3, t4 = group_teams
    d1, d2, d3 = dates

    return [
        {"team1": t1, "team2": t2, "match_date": d1, "neutral": 1},
        {"team1": t3, "team2": t4, "match_date": d1, "neutral": 1},
        {"team1": t1, "team2": t3, "match_date": d2, "neutral": 1},
        {"team1": t4, "team2": t2, "match_date": d2, "neutral": 1},
        {"team1": t4, "team2": t1, "match_date": d3, "neutral": 1},
        {"team1": t2, "team2": t3, "match_date": d3, "neutral": 1},
    ]

def build_round_of_32_from_groups(all_group_results):
    group_winners = {group_name: result[0][0] for group_name, result in all_group_results.items()}
    group_runners_up = {group_name: result[1][0] for group_name, result in all_group_results.items()}

    third_place_teams = []
    for group_name, result in all_group_results.items():
        team, stats = result[2]
        third_place_teams.append({
            "group": group_name,
            "team": team,
            "points": stats["points"],
            "gf": stats["gf"],
            "ga": stats["ga"],
            "gd": stats["gd"],
        })

    third_place_teams = sorted(
        third_place_teams,
        key=lambda x: (x["points"], x["gd"], x["gf"]),
        reverse=True
    )

    qualified_third_placed = third_place_teams[:8]

    qualified_third_by_group = {
        team_data["group"]: team_data["team"]
        for team_data in qualified_third_placed
    }

    qualified_key = ''.join(sorted(team_data["group"] for team_data in qualified_third_placed))

    if qualified_key not in THIRD_PLACE_LOOKUP:
        raise ValueError(f"Missing third-place lookup for combination: {qualified_key}")

    slot_to_group_slot = THIRD_PLACE_LOOKUP[qualified_key]

    third_place_slot_teams = {
        slot: qualified_third_by_group[group_slot[1]]
        for slot, group_slot in slot_to_group_slot.items()
    }

    round_of_32_fixtures = [
        (group_runners_up["A"], group_runners_up["B"]),
        (group_winners["E"], third_place_slot_teams["1E"]),
        (group_winners["F"], group_runners_up["C"]),
        (group_winners["C"], group_runners_up["F"]),
        (group_winners["I"], third_place_slot_teams["1I"]),
        (group_runners_up["E"], group_runners_up["I"]),
        (group_winners["A"], third_place_slot_teams["1A"]),
        (group_winners["L"], third_place_slot_teams["1L"]),
        (group_winners["D"], third_place_slot_teams["1D"]),
        (group_winners["G"], third_place_slot_teams["1G"]),
        (group_runners_up["K"], group_runners_up["L"]),
        (group_winners["H"], group_runners_up["J"]),
        (group_winners["B"], third_place_slot_teams["1B"]),
        (group_winners["J"], group_runners_up["H"]),
        (group_winners["K"], third_place_slot_teams["1K"]),
        (group_runners_up["D"], group_runners_up["G"]),
    ]

    return round_of_32_fixtures, qualified_third_placed, qualified_key

def get_winners(results):
    return [result["winner"] for result in results]

def build_round_of_16_fixtures(round_of_32_results):
    winners = get_winners(round_of_32_results)
    return [
        (winners[0], winners[2]),
        (winners[1], winners[4]),
        (winners[5], winners[7]),
        (winners[6], winners[3]),
        (winners[8], winners[10]),
        (winners[9], winners[11]),
        (winners[12], winners[15]),
        (winners[13], winners[14]),
    ]

def build_quarterfinal_fixtures(round_of_16_results):
    winners = get_winners(round_of_16_results)
    return [
        (winners[0], winners[1]),
        (winners[4], winners[5]),
        (winners[2], winners[3]),
        (winners[6], winners[7]),
    ]

def build_semifinal_fixtures(quarterfinal_results):
    winners = get_winners(quarterfinal_results)
    return [
        (winners[0], winners[1]),
        (winners[2], winners[3]),
    ]

def build_final_fixtures(semifinal_results):
    winners = get_winners(semifinal_results)
    return [
        (winners[0], winners[1])
    ]

def simulate_full_tournament():
    all_group_results = {}

    for group_name, teams in all_groups.items():
        group_fixtures = build_group_fixtures(teams, group_dates[group_name])
        all_group_results[group_name] = simulate_whole_group(teams, group_fixtures, None)

    round_of_32_fixtures, qualified_third_placed, qualified_key = build_round_of_32_from_groups(all_group_results)

    round_of_32_results = simulate_round(round_of_32_fixtures, pd.Timestamp("2026-06-28"))
    round_of_16_results = simulate_round(build_round_of_16_fixtures(round_of_32_results), pd.Timestamp("2026-07-04"))
    quarterfinal_results = simulate_round(build_quarterfinal_fixtures(round_of_16_results), pd.Timestamp("2026-07-09"))
    semifinal_results = simulate_round(build_semifinal_fixtures(quarterfinal_results), pd.Timestamp("2026-07-14"))
    final_results = simulate_round(build_final_fixtures(semifinal_results), pd.Timestamp("2026-07-19"))

    return {
        "champion": final_results[0]["winner"],
        "qualified_key": qualified_key,
        "qualified_third_placed": qualified_third_placed,
        "round_of_32_fixtures": round_of_32_fixtures,
    }

def simulate_many_tournaments(n_simulations=1000):
    champion_counts = Counter()

    for _ in range(n_simulations):
        sim_result = simulate_full_tournament()
        champion_counts[sim_result["champion"]] += 1

    all_teams = sorted({team for teams in all_groups.values() for team in teams})

    champion_probs = pd.DataFrame({
        "team": all_teams,
        "champion_prob": [champion_counts[team] / n_simulations for team in all_teams]
    }).sort_values("champion_prob", ascending=False).reset_index(drop=True)

    return champion_probs

champion_probs = simulate_many_tournaments(n_simulations=1000)

print("\nWorld Cup winner probabilities:")
for _, row in champion_probs.iterrows():
    print(f"{row['team']}: {row['champion_prob']:.3%}")

