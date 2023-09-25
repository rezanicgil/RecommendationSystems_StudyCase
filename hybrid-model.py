import numpy as np
import pandas as pd

pd.set_option('display.max_columns',20)
movie = pd.read_csv('./kaggle/input/movie.csv');
rating = pd.read_csv('./kaggle/input/rating.csv')

df = movie.merge(rating, how="left", on="movieId")

comment_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = comment_counts[comment_counts["count"] <= 1000].index

# Let's get access to movies with over 1000 reviews:
common_movies = df[~df["title"].isin(rare_movies)]


# Let's create the User Movie Df:
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

print("--------------------Item Based Movie Recommendation-------------------")

# item-based movie recommendation example:
movie_name = "Matrix, The (1999)"
movie_name = user_movie_df[movie_name]
print(user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10))

print("---------------------------------------------------------------------")

# Let's determine the movies that the user watched.

# Let's choose random user:
# random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
random_user = 28491

# Let's reduce the #dataset to user 28491:
random_user_df = user_movie_df[user_movie_df.index == random_user]


# Let's choose non-NaN. Movies watched by all 28491:
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# we have reduced the dataset based on movies watched by user 28491:
movies_watched_df = user_movie_df[movies_watched]

# information on how many movies each user watched in total:
user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId","movie_count"]

# 60% of movies watched by 28491:
perc = len(movies_watched) * 60 / 100

# People who have watched more than 60% movies together with 28491 users:
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

final_df_no_duplicates = final_df.apply(lambda x: x.drop_duplicates())

# correlation data frames for all users:
corr_df = final_df_no_duplicates.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index()

# Users with a correlation of %65 or more with 28491 users:
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)


# let's see the ratings of users:
rating = pd.read_csv("./kaggle/input/rating.csv")
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

# Calculate the Weighted Average Recommendation Score and keep the first 5 movies.

# Let's do a single score with the most similar by corr * rating:
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()


# Movies 28491 will like:
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.7].sort_values("weighted_rating", ascending=False)

movies_to_be_recommend.merge(movie[["movieId", "title"]])

# ▪ 5 suggestions user-based
movies_to_be_recommend.merge(movie[["movieId", "title"]])[:5]['title'].to_list()

# Make an item-based suggestion based on the name of the movie that the user has watched with the highest score.


# The last highly-rated movie by user 108170:

user = 108170
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]


# ▪ 5 suggestions user-based
movies_to_be_recommend.merge(movie[["movieId", "title"]])[:5]['title'].to_list()

print("--------------------5 Suggestion Item based-------------------------")
# ▪ 5 suggestions item-based
movie_name = movie[movie['movieId'] == movie_id]['title'].values[0]
movie_name = user_movie_df[movie_name]
movies_from_item_based = user_movie_df.corrwith(movie_name).sort_values(ascending=False)
movies_from_item_based[1:6].index.to_list()
