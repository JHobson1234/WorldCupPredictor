import tkinter as tk
from tkinter import ttk, scrolledtext
import pandas as pd
from world_cup_predictor import (
    predict_match,
    predict_most_likely_score,
    simulate_match,
    simulate_full_tournament,
    most_likely_tournament_run,
    all_groups,
    simulate_many_tournaments
)

teams = sorted({team for group in all_groups.values() for team in group})

def write_output(text):
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, text)

def simulate_game():
    team1 = team1_combo.get()
    team2 = team2_combo.get()

    if not team1 or not team2:
        write_output("Please select both teams.")
        return

    if team1 == team2:
        write_output("Please choose two different teams.")
        return

    result = simulate_match(team1, team2, pd.Timestamp("2026-06-20"), neutral=1)

    write_output(
        f"SIMULATED MATCH\n\n"
        f"{result['team1']} {result['team1_goals']} - {result['team2_goals']} {result['team2']}\n"
        f"Winner: {result['winner']}"
    )

def predict_game():
    team1 = team1_combo.get()
    team2 = team2_combo.get()

    if not team1 or not team2:
        write_output("Please select both teams.")
        return

    if team1 == team2:
        write_output("Please choose two different teams.")
        return

    match_date = pd.Timestamp("2026-06-20")
    result = predict_match(team1, team2, match_date, neutral=1)
    score_result = predict_most_likely_score(team1, team2, match_date, neutral=1, max_goals=5)

    write_output(
        f"MATCH PREDICTION\n\n"
        f"{team1} win: {result['team1_win_prob']:.3%}\n"
        f"Draw: {result['draw_prob']:.3%}\n"
        f"{team2} win: {result['team2_win_prob']:.3%}\n\n"
        f"Predicted score: {score_result['predicted_score']}\n"
        f"Score probability: {score_result['score_probability']:.3%}\n"
        f"{team1} xG: {score_result['team1_xg']:.2f}\n"
        f"{team2} xG: {score_result['team2_xg']:.2f}"
    )

def predict_tournament():
    result = simulate_full_tournament()

    text = "WORLD CUP TOURNAMENT PREDICTION\n\nGROUP STAGE\n"

    for group_name, standings in result["group_results"].items():
        text += f"\nGroup {group_name}\n"
        for pos, (team, stats) in enumerate(standings, start=1):
            text += (
                f"{pos}. {team} - "
                f"{stats['points']} pts, GF {stats['gf']}, "
                f"GA {stats['ga']}, GD {stats['gd']}\n"
            )

    text += "\nBEST THIRD-PLACED TEAMS\n"
    for team_data in result["qualified_third_placed"]:
        text += (
            f"Group {team_data['group']} - {team_data['team']} "
            f"({team_data['points']} pts, GD {team_data['gd']})\n"
        )

    for round_name, round_key in [
        ("ROUND OF 32", "round_of_32_results"),
        ("ROUND OF 16", "round_of_16_results"),
        ("QUARTERFINALS", "quarterfinal_results"),
        ("SEMIFINALS", "semifinal_results"),
        ("FINAL", "final_results"),
    ]:
        text += f"\n{round_name}\n"
        for match in result[round_key]:
            text += (
                f"{match['team1']} {match['team1_goals']} - "
                f"{match['team2_goals']} {match['team2']} | "
                f"Winner: {match['winner']}\n"
            )

    text += f"\nCHAMPION: {result['champion']}"
    write_output(text)

def show_most_likely_tournament():
    result = most_likely_tournament_run(1000)

    text = "MOST LIKELY TOURNAMENT RUN FROM 1000 SIMULATIONS\n\n"

    m

    final_match, final_count = result["final"]
    text += (
        f"Most common final: {final_match[0]} vs {final_match[1]} "
        f"| Winner: {final_match[2]} ({final_count}/1000)\n\n"
    )

    text += "Most common semifinals\n"
    for match, count in result["semifinals"]:
        text += f"{match[0]} vs {match[1]} | Winner: {match[2]} ({count}/1000)\n"

    text += "\nMost common quarterfinals\n"
    for match, count in result["quarterfinals"]:
        text += f"{match[0]} vs {match[1]} | Winner: {match[2]} ({count}/1000)\n"

    write_output(text)

def ten_most_likely_winners():
    result = simulate_many_tournaments(1000)
    top_10 = result.head(10)
    text = "10 most likely winners\n\n"
    for _, row in top_10.iterrows():
        team = row["team"]
        chance = row["champion_prob"]
        text += f"{team}: {chance:.2%}\n"
    write_output(text)



ws = tk.Tk()
ws.title("World Cup Simulator")
ws.geometry("850x800")
ws.resizable(False, False)

title_label = tk.Label(ws, text="World Cup Simulator", font=("Arial", 16))
title_label.pack(pady=20)

team1_combo = ttk.Combobox(ws, values=teams, state="readonly", width=30)
team1_combo.pack(pady=10)

team2_combo = ttk.Combobox(ws, values=teams, state="readonly", width=30)
team2_combo.pack(pady=10)

simulate_button = tk.Button(ws, text="Simulate Match", command=simulate_game)
simulate_button.pack(pady=10)

predict_button = tk.Button(ws, text="Predict Match", command=predict_game)
predict_button.pack(pady=10)

tournament_button = tk.Button(ws, text="Predict Tournament", command=predict_tournament)
tournament_button.pack(pady=10)

most_likely_button = tk.Button(ws, text="Most Likely Tournament", command=show_most_likely_tournament)
most_likely_button.pack(pady=10)

tournament_odds_button = tk.Button(ws, text="Tournament Odds", command=ten_most_likely_winners)
tournament_odds_button.pack(pady=10)

output_box = scrolledtext.ScrolledText(ws, width=90, height=28, font=("Courier New", 10))
output_box.pack(pady=20)

ws.mainloop()