from flask import Flask, render_template, abort
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data from file
CSV_FILE = "steam_reviews_trimmed.csv"

df = pd.read_csv(CSV_FILE, usecols=["author.steamid", "app_id", "app_name", "recommended"])

# Make app name
games_df = df[["app_id", "app_name"]].drop_duplicates().set_index("app_id")
games_df = games_df.sort_values("app_name")

# Rating-tabel for CF
ratings = df.rename(columns={
    "author.steamid": "user_id",
    "app_id": "item_id"
}).copy()
ratings["rating"] = ratings["recommended"].astype(int)
ratings = ratings[["user_id", "item_id", "rating"]]

# item–item similarity
user_item_matrix = ratings.pivot_table(
    index="user_id",
    columns="item_id",
    values="rating",
    aggfunc="mean"
)

ratings_filled = user_item_matrix.fillna(0)
item_matrix = ratings_filled.T  # rows = item_id, cols = user_id

item_sim_matrix = cosine_similarity(item_matrix)
item_ids = item_matrix.index
item_sim = pd.DataFrame(item_sim_matrix, index=item_ids, columns=item_ids)


def recommend_similar_items(item_id, top_k=10):
    """Returner liste over lignende spill basert på item-item similarity."""
    if item_id not in item_sim.index:
        return []

    sims = item_sim.loc[item_id].drop(item_id).sort_values(ascending=False).head(top_k)

    recs = []
    for other_id, score in sims.items():
        if other_id in games_df.index:
            name = games_df.loc[other_id, "app_name"]
        else:
            name = str(other_id)

        recs.append({
            "app_id": int(other_id),
            "app_name": name,
            "similarity": float(score),
        })
    return recs

# Flask Routes
@app.route("/")
def index():
    games = [
        {"app_id": int(app_id), "app_name": row.app_name}
        for app_id, row in games_df.iterrows()
    ]
    return render_template("index.html", games=games)


@app.route("/game/<int:app_id>")
def game(app_id):
    if app_id not in games_df.index:
        abort(404)

    selected = {
        "app_id": int(app_id),
        "app_name": games_df.loc[app_id, "app_name"],
    }
    recs = recommend_similar_items(app_id, top_k=10)

    return render_template("game.html", game=selected, recommendations=recs)


if __name__ == "__main__":
    app.run(debug=True)
