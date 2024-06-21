import pandas as pd

# Load the dataset
df = pd.read_csv("nba_games.csv", index_col=0)

# Sort values by date and reset the index
df = df.sort_values("date")
df = df.reset_index(drop=True)

# Delete duplicate columns
del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]


def add_target(team):
    # Add a new target column containing the value of the "won" column from the next row
    team["target"] = team["won"].shift(-1)
    return team


# Apply the add_target function grouped by team
df = df.groupby("team", group_keys=False).apply(add_target)

# Handle NULL values in the target column by setting them to 2 and convert to integers
df["target"][pd.isnull(df["target"])] = 2
df["target"] = df["target"].astype(int, errors="ignore")

# Remove columns with NULL values
nulls = pd.isnull(df).sum()
valid_columns = df.columns[~df.columns.isin(nulls[nulls > 0].index)]
df = df[valid_columns].copy()

# -------------------------------------------------------------

# Machine Learning section

from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier

# Initialize Ridge Classifier and TimeSeriesSplit
rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)

# Initialize Sequential Feature Selector
sfs = SequentialFeatureSelector(
    rr, n_features_to_select=30, direction="forward", cv=split
)

# Define columns to be removed
removed_columns = ["season", "date", "won", "target", "team", "team_opp"]

# Select columns excluding the removed columns
selected_columns = df.columns[~df.columns.isin(removed_columns)]

from sklearn.preprocessing import MinMaxScaler

# Scale selected columns
scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])

# Fit the feature selector to find the best predictors
sfs.fit(df[selected_columns], df["target"])
predictors = list(selected_columns[sfs.get_support()])


def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []

    # Get a sorted list of unique seasons
    seasons = sorted(data["season"].unique())

    for i in range(start, len(seasons), step):
        season = seasons[i]

        # Split the data into training and testing sets based on seasons
        train = data[data["season"] < season]
        test = data[data["season"] == season]

        # Fit the model and make predictions
        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)

        # Combine actual and predicted values
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]

        all_predictions.append(combined)

    return pd.concat(all_predictions)


# Backtest the model
predictions = backtest(df, rr, predictors)

from sklearn.metrics import accuracy_score

# Filter out predictions with target value 2
predictions = predictions[predictions["actual"] != 2]

# Calculate accuracy score
accuracy = accuracy_score(predictions["actual"], predictions["prediction"])

# Calculate home win percentage
home_win_percentage = df.groupby("home").apply(
    lambda x: x[x["won"] == 1].shape[0] / x.shape[0]
)

# Improving Model with Rolling Averages


# Define function to calculate rolling averages for the last 10 games
def find_team_averages(team):
    numeric_columns = team.select_dtypes(include="number")
    rolling = numeric_columns.rolling(10).mean()
    return rolling


# Apply rolling average function grouped by team and season
df_rolling = df.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

# Rename columns to indicate rolling averages
rolling_cols = [f"{col}_10" for col in df_rolling.columns]
df_rolling.columns = rolling_cols

# Concatenate the original and rolling average DataFrames
df = pd.concat([df, df_rolling], axis=1)

# Drop rows with NaN values resulting from the rolling averages
df = df.dropna()


# Define function to shift columns for the next game
def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col


# Add columns for the next game's home, team_opp, and date
def add_col(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))


df["home_next"] = add_col(df, "home")
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df, "date")

# Merge the DataFrame with itself to add the rolling average columns of the opposing team
full = df.merge(
    df[rolling_cols + ["team_opp_next", "date_next", "team"]],
    left_on=["team", "date_next"],
    right_on=["team_opp_next", "date_next"],
)

# Define columns to be removed
removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns

# Select columns excluding the removed columns
selected_columns = full.columns[~full.columns.isin(removed_columns)]

# Fit the feature selector to find the best predictors in the full DataFrame
sfs.fit(full[selected_columns], full["target"])
predictors = list(selected_columns[sfs.get_support()])

# Backtest the model with the updated predictors
predictions = backtest(full, rr, predictors)

# Calculate the final accuracy score
final_accuracy = accuracy_score(predictions["actual"], predictions["prediction"])

print(f"Initial Accuracy: {accuracy}")
print(f"Final Accuracy after using Rolling Averages: {final_accuracy}")
print(f"Home Win Percentage: {home_win_percentage}")
