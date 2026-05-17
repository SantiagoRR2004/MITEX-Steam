from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import dataHandling
import pandas as pd
import bertopic
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.representation import MaximalMarginalRelevance
import nltk
import json
import os

nltk.download("stopwords", quiet=True)


def basicTopicModeling() -> None:
    """
    Perform basic topic modeling on the cleaned reviews using BERTopic.

    This just simply concatenates all the reviews for each game
    (positive and negative as different documents) and then
    applies basic BERTopic.

    Args:
        - None

    Returns:
        - None
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    reviewPaths = dataHandling.obtainReviewStructure(
        os.path.join(currentDirectory, "cleanData")
    )

    data_list = []

    for game, reviewTypes in reviewPaths.items():

        for reviewType, path in reviewTypes.items():

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    review = eval(line)["review"]
                    data_list.append({"game": f"{game} {reviewType}", "review": review})

    df = pd.DataFrame(data_list)
    # min_df=5 to ignore words that appear in less than 5 reviews
    gaming_stopwords = stopwords.words("english") + [
        "game",
        "games",
        "play",
        "playing",
        "played",
        "like",
        "fun",
        "would",
        "get",
        "good",
        "bad",
    ]
    # min_df=5 to ignore words that appear in less than 5 reviews
    vectorizerModel = CountVectorizer(
        stop_words=gaming_stopwords, min_df=5, ngram_range=(1, 2)
    )
    representation_model = MaximalMarginalRelevance(diversity=0.3)

    umap_model = UMAP(
        n_neighbors=40,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=80,
        min_samples=10,
        cluster_selection_epsilon=0.0,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    model = bertopic.BERTopic(
        language="english",
        vectorizer_model=vectorizerModel,
        representation_model=representation_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=50,
        nr_topics="auto",  # number of topics to reduce to
    )

    docs = df["review"].tolist()
    pred, prob = model.fit_transform(docs)

    df["cluster"] = pred

    keyWords = model.get_topics()
    clusters = {}

    print(df.groupby("cluster").size())
    for cluster_id, group in df.groupby("cluster"):
        c_id = int(cluster_id)
        if c_id == -1:
            keywords = ["outliers"]
        else:
            keywords = [word for word, _ in keyWords[cluster_id]]

        # How many reviews per game in this cluster
        game_counts = group["game"].value_counts()
        associated_games = {}
        for game, count in game_counts.items():
            total_reviews = len(df[df["game"] == game])
            percentage = (count / total_reviews) * 100

            if (
                percentage >= 2.0
            ):  # Only consider games that have at least 2% of their reviews in this cluster
                associated_games[game] = {
                    "count": int(count),
                    "percentage": round(percentage, 2),
                }
        clusters[c_id] = {
            "keywords": keywords,
            "total_reviews": int(group.shape[0]),
            "games": associated_games,
        }

    # Save the results to a JSON file
    with open(
        os.path.join(currentDirectory, "videogameClusters.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    basicTopicModeling()
