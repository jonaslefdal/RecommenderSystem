from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("steam_reviews_trimmed.csv")

df["rating"] = df["recommended"].astype(int)

ratings = df.rename(columns={
    "author.steamid": "user_id",
    "app_id": "item_id"
})[["user_id", "item_id", "rating"]]

train_df, test_df = train_test_split(ratings, test_size=0.1, random_state=42)

print("Train size:", len(train_df))
print("Test size:", len(test_df))

# Pivotér kun train-set for å trene anbefalingsmodellen
train_matrix = train_df.pivot_table(
    index="user_id",
    columns="item_id",
    values="rating",
    aggfunc="mean"
).fillna(0)

item_matrix_train = train_matrix.T
item_sim_matrix_train = cosine_similarity(item_matrix_train)
item_sim_train = pd.DataFrame(item_sim_matrix_train,
                              index=item_matrix_train.index,
                              columns=item_matrix_train.index)

def recommend_for_user(user_id, top_k=10):
    # hent brukerens ratings i train
    user_ratings = train_matrix.loc[user_id]

    # spill brukeren likte (rating=1)
    liked_items = user_ratings[user_ratings > 0].index

    # accumuler score fra andre items basert på similarity
    scores = np.zeros(len(item_sim_train))

    for item in liked_items:
        scores += item_sim_train.loc[item].values

    # sorter og fjern spill brukeren allerede liker
    recommended_indices = np.argsort(scores)[::-1]
    recommended_items = item_sim_train.index[recommended_indices]

    recommended_items = [i for i in recommended_items if i not in liked_items]

    return recommended_items[:top_k]

def precision_at_k(user_id, k=10):
    recommended = recommend_for_user(user_id, k)
    actual = test_df[test_df["user_id"] == user_id]["item_id"].values

    if len(actual) == 0:
        return np.nan  # ikke evaluer brukere uten test-rating

    hits = len(set(recommended) & set(actual))
    return hits / k


def evaluate_precision(k=10, sample_size=500):
    sample_users = np.random.choice(train_matrix.index, sample_size, replace=False)
    scores = []

    for user in sample_users:
        score = precision_at_k(user, k)
        if not np.isnan(score):
            scores.append(score)

    return np.mean(scores)

# Finn globale mest populære items (basert på positive ratings i train)
positive_train = train_df[train_df["rating"] == 1]
item_popularity = positive_train["item_id"].value_counts()
popular_items_global = list(item_popularity.index)

def recommend_popular_for_user(user_id, top_k=10):
    # samme liste for alle, uavhengig av bruker
    return popular_items_global[:top_k]


def precision_at_k_popular(user_id, k=10):
    recommended = recommend_popular_for_user(user_id, k)
    actual = test_df[test_df["user_id"] == user_id]["item_id"].values

    if len(actual) == 0:
        return np.nan

    hits = len(set(recommended) & set(actual))
    return hits / k

def evaluate_precision_popular(k=10, sample_size=500):
    sample_users = np.random.choice(train_matrix.index, sample_size, replace=False)
    scores = []

    for user in sample_users:
        score = precision_at_k_popular(user, k)
        if not np.isnan(score):
            scores.append(score)

    return np.mean(scores)


all_items = list(train_matrix.columns)

def recommend_random_for_user(user_id, top_k=10):
    # velg K tilfeldige items (uten å bry deg om brukeren)
    if len(all_items) < top_k:
        return all_items
    return list(np.random.choice(all_items, size=top_k, replace=False))


def precision_at_k_random(user_id, k=10):
    recommended = recommend_random_for_user(user_id, k)
    actual = test_df[test_df["user_id"] == user_id]["item_id"].values

    if len(actual) == 0:
        return np.nan

    hits = len(set(recommended) & set(actual))
    return hits / k

def evaluate_precision_random(k=10, sample_size=500):
    sample_users = np.random.choice(train_matrix.index, sample_size, replace=False)
    scores = []

    for user in sample_users:
        score = precision_at_k_random(user, k)
        if not np.isnan(score):
            scores.append(score)

    return np.mean(scores)


print("Evaluating Precision@10 for CF ...")
avg_precision_cf = evaluate_precision(k=10, sample_size=300)
print("CF Precision@10:", avg_precision_cf)

print("\nEvaluating Precision@10 for Popularity baseline ...")
avg_precision_pop = evaluate_precision_popular(k=10, sample_size=300)
print("Popularity Precision@10:", avg_precision_pop)

print("\nEvaluating Precision@10 for Random baseline ...")
avg_precision_rand = evaluate_precision_random(k=10, sample_size=300)
print("Random Precision@10:", avg_precision_rand)
