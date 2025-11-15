import pandas as pd

INPUT_FILE = "steam_reviews.csv"

# Remove not needed columns
cols = ["author.steamid", "app_id", "app_name", "recommended"]
df = pd.read_csv(INPUT_FILE, usecols=cols)

# We remove users who have less then 1 review
user_counts = df["author.steamid"].value_counts()
active_users = user_counts[user_counts > 1].index
df = df[df["author.steamid"].isin(active_users)]

MIN_GAMES_PER_USER = 10
MIN_ITEM_RATINGS = 100

# Filter users
user_game_counts = df.groupby("author.steamid")["app_id"].nunique()
active_users = user_game_counts[user_game_counts >= MIN_GAMES_PER_USER].index
df = df[df["author.steamid"].isin(active_users)]

# Filter items
item_counts = df["app_id"].value_counts()
popular_items = item_counts[item_counts >= MIN_ITEM_RATINGS].index
df = df[df["app_id"].isin(popular_items)]

df.to_csv("steam_reviews_trimmed.csv", index=False)

print("Done")
