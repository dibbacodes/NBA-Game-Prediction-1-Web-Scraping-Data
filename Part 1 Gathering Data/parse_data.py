import os
import pandas as pd
from bs4 import BeautifulSoup

# Directory where the box score HTML files are stored
SCORE_DIR = "data/scores"

# List all files in the SCORE_DIR
box_scores = os.listdir(SCORE_DIR)

# Print the number of box score files found
print(f"Number of box score files: {len(box_scores)}")

# Create full file paths for box score HTML files and filter for HTML files only
box_scores = [os.path.join(SCORE_DIR, f) for f in box_scores if f.endswith(".html")]


# Function to parse HTML file and return BeautifulSoup object
def parse_html(box_score):
    with open(box_score) as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Remove unnecessary rows for easier parsing later
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr.thread")]

    return soup


# Function to read the line score table from the parsed HTML
def read_line_score(soup):
    line_score = pd.read_html(str(soup), attrs={"id": "line_score"})[0]

    # Rename columns for consistency
    cols = list(line_score.columns)
    cols[0] = "team"
    cols[-1] = "total"
    line_score.columns = cols

    # Select relevant columns
    line_score = line_score[["team", "total"]]
    return line_score


# Function to read statistics table (basic or advanced) for a given team
def read_stats(soup, team, stat):
    df = pd.read_html(str(soup), attrs={"id": f"box-{team}-game-{stat}"}, index_col=0)[
        0
    ]

    # Convert all values to numeric, setting non-numeric values to NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


# Function to read the season information from the HTML
def read_season_info(soup):
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all("a")]
    season = os.path.basename(hrefs[1]).split("_")[0]
    return season


# Initialize variables
base_cols = None
games = []

# Process each box score file
for box_score in box_scores:
    # Parse the HTML file
    soup = parse_html(box_score)

    # Read the line score table
    line_score = read_line_score(soup)
    teams = list(line_score["team"])

    summaries = []
    for team in teams:
        # Read basic and advanced statistics for the team
        basic = read_stats(soup, team, "basic")
        advanced = read_stats(soup, team, "advanced")

        # Get totals and maximums for the stats
        totals = pd.concat([basic.iloc[-1, :], advanced.iloc[-1, :]])
        totals.index = totals.index.str.lower()

        maxes = pd.concat([basic.iloc[:-1, :].max(), advanced.iloc[:-1, :].max()])
        maxes.index = maxes.index.str.lower() + "_max"

        # Combine totals and maximums into a single summary
        summary = pd.concat([totals, maxes])

        # Define base columns for the summary
        if base_cols is None:
            base_cols = list(summary.index.drop_duplicates(keep="first"))
            base_cols = [b for b in base_cols if "bpm" not in b]

        summary = summary[base_cols]

        summaries.append(summary)

    # Combine summaries for both teams
    summary = pd.concat(summaries, axis=1).T

    # Combine line score and team summaries
    game = pd.concat([summary, line_score], axis=1)

    # Assign home/away labels (0 for away, 1 for home)
    game["home"] = [0, 1]

    # Create opponent data by reversing the game DataFrame and resetting the index
    game_opp = game.iloc[::-1].reset_index()
    game_opp.columns += "_opp"

    # Combine game data with opponent data
    full_game = pd.concat([game, game_opp], axis=1)

    # Add season and date information
    full_game["season"] = read_season_info(soup)
    full_game["date"] = os.path.basename(box_score)[:8]
    full_game["date"] = pd.to_datetime(full_game["date"], format="%Y%m%d")

    # Add a win indicator (1 if total score is greater than opponent's total, else 0)
    full_game["won"] = full_game["total"] > full_game["total_opp"]

    # Add the full game data to the list of games
    games.append(full_game)

    # Print progress every 100 games
    if len(games) % 100 == 0:
        print(f"{len(games)} / {len(box_scores)}")

# Combine all games into a single DataFrame
games_df = pd.concat(games, ignore_index=True)

# Save the combined DataFrame to a CSV file
games_df.to_csv("nba_games.csv")
