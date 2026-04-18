from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import dataHandling
import pandas as pd
import bertopic
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

    df = pd.DataFrame(columns=["game", "reviews"])

    for game, reviewTypes in reviewPaths.items():

        for reviewType, path in reviewTypes.items():
            reviews = []

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    review = eval(line)["review"]
                    reviews.append(review)

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [{"game": f"{game} {reviewType}", "reviews": " ".join(reviews)}]
                    ),
                ]
            )

    vectorizerModel = CountVectorizer(stop_words=stopwords.words("english"))
    model = bertopic.BERTopic(
        language="english",
        vectorizer_model=vectorizerModel,
    )
    pred, prob = model.fit_transform(df["reviews"].tolist())

    keyWords = model.get_topics()
    clusters = {}

    for game, cluster in zip(df["game"], pred):
        if cluster not in clusters:
            clusters[cluster] = {"games": [], "keywords": keyWords[cluster]}

        clusters[cluster]["games"].append(game)

    # Save the results to a JSON file
    with open(
        os.path.join(currentDirectory, "videogameClusters.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    basicTopicModeling()
