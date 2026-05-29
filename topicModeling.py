from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import torch
import dataHandling
import pandas as pd
import bertopic
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.representation import MaximalMarginalRelevance
import nltk
import json
import os
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

import decoders
import models
import LLMManager

nltk.download("stopwords", quiet=True)


def get_docs():
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

    return df


def evaluate_topic_models(model: bertopic.BERTopic, docs: list, pred: list) -> None:
    """
    Evaluate the result of a topic model
    """
    analyzer = model.vectorizer_model.build_analyzer()
    tokens = [analyzer(str(doc)) for doc in docs]
    dictionary = corpora.Dictionary(tokens)
    # Extract topic words for coherence evaluation
    topics = model.get_topic_info()
    topic_words = []
    for topic in topics["Topic"]:
        if topic != -1:
            words = [word for word, _ in model.get_topic(topic)]
            topic_words.append(words)

    # Calculate Coherencia C_v
    if len(topic_words) > 0:
        coherence_model = CoherenceModel(
            topics=topic_words, texts=tokens, dictionary=dictionary, coherence="c_v"
        )
        coherence_score = coherence_model.get_coherence()
    else:
        coherence_score = 0.0

    total_docs = len(pred)
    outliers = list(pred).count(-1)
    outlier_percentage = (outliers / total_docs) * 100
    num_topics = len(set(pred)) - (1 if -1 in pred else 0)

    print(
        f"Tópicos: {num_topics} | Outliers: {outlier_percentage:.2f}% | Coherencia C_v: {coherence_score:.4f}"
    )


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
    df = get_docs()

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
        "time",
        "also",
        "feel",
        "ive",
        "even",
        "much",
        "really",
        "make",
        "one",
        "people",
        "way",
        "things",
        "thing",
        "still",
        "got",
        "know",
        "think",
        "gameplay",
        "story",
        "hours",
    ]
    # min_df=5 to ignore words that appear in less than 5 reviews
    vectorizerModel = CountVectorizer(
        stop_words=gaming_stopwords, min_df=5, max_df=0.8, ngram_range=(1, 2)
    )
    representation_model = MaximalMarginalRelevance(diversity=0.3)

    umap_model = UMAP(
        n_neighbors=30,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=80,
        min_samples=5,
        cluster_selection_epsilon=0.1,
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
        nr_topics=25,  # number of topics to reduce to
    )

    docs = df["review"].tolist()
    pred, _ = model.fit_transform(docs)

    evaluate_topic_models(model, docs, pred)

    new_topics = model.reduce_outliers(docs, pred, strategy="c-tf-idf", threshold=0.05)

    model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizerModel)
    pred = new_topics

    evaluate_topic_models(model, docs, pred)

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
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    topicsPath = os.path.join(currentDirectory, "topicsData")
    os.makedirs(topicsPath, exist_ok=True)
    with open(
        os.path.join(currentDirectory, "topicsData", "videogameClusters.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
        f.write("\n")


def get_clusters():
    """
    Load the clusters from the JSON file and return them as a dictionary.

    Args:
        - None
    Returns:
        - A dictionary containing the clusters.
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    clusters_path = os.path.join(
        currentDirectory, "topicsData", "videogameClusters.json"
    )

    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    return clusters, clusters_path


def add_descriptions_to_clusters():
    """
    Add descriptions to the clusters based on the keywords and associated games.

    This function reads the clusters from the JSON file, generates a description for each cluster
    based on its keywords and associated games, and then saves the updated clusters back to the JSON file.

    Args:
        - None
    """
    clusters, clusters_path = get_clusters()

    for cluster_id, cluster_info in clusters.items():
        keywords = cluster_info["keywords"]

        if cluster_id == "-1":
            title = "Outliers"
            description = "This cluster contains outlier reviews that do not fit well into any of the main topics. These reviews may be very unique or may contain noise."
        else:
            sequence = decoders.TokenSequenceConstraint(
                wantedText=[
                    '{"name_topic": "',
                    decoders.EndingText('"', models.GEN_TOKENIZER),
                    ', "description": "',
                    decoders.EndingText('"', models.GEN_TOKENIZER),
                    "}",
                ],
                tokenizer=models.GEN_TOKENIZER,
            )

            d = decoders.FormattedDecoder(
                models.GEN_TOKENIZER, sequence, jsonFormat=True
            )
            m = LLMManager.LLMManager(d)
            d.setManager(m)

            prompt = f"""Analyze the following keywords that belong to a topic extracted from Steam reviews:
{", ".join(keywords)}. 

Generate a JSON with the following format:
{{
    "name_topic": "[A short title (3-5 words) that summarizes the issue or theme]",
    "description": "[A sentence that explains what this topic means for a player or game developer]"
}}
            """

            message = [
                {"role": "user", "content": prompt},
            ]
            # Generate the response
            tokens = m.processPrompt(message)

            while not d.finished:
                tokens = m.decode(torch.tensor([[tokens[-1]]]))

            finalText = m.decodeTokens(tokens)
            try:
                output = json.loads(finalText)
            except Exception as e:
                print(f"Error parsing JSON for cluster {cluster_id}: {e}")
                continue
            title = output["name_topic"]
            description = output["description"]

        clusters[cluster_id]["title"] = title
        clusters[cluster_id]["description"] = description

    with open(clusters_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)


def create_game_centric_json() -> None:
    clusters, path = get_clusters()
    res = {
        "games": {},
        "topics": {
            t: {
                k: inf[k] for k in ["keywords", "title", "description", "total_reviews"]
            }
            for t, inf in clusters.items()
            if t != "-1"
        },
    }
    for t, inf in clusters.items():
        if t != "-1":
            for g, data in inf.get("games", {}).items():
                res["games"].setdefault(g, {})[t] = data
    with open(
        os.path.join(os.path.dirname(path), "gameTopics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def completeTopicModelingPipeline(forceRefresh=False):
    """
    Complete pipeline for topic modeling and adding descriptions to clusters.

    This function runs the entire process of topic modeling, adding descriptions to clusters, and creating a game-centric JSON file. It can be run with a force refresh option to reprocess all data.

    Args:
        - forceRefresh (bool): If True, forces the reprocessing of all data. Default is False.
    """
    if forceRefresh:
        basicTopicModeling()
    add_descriptions_to_clusters()
    create_game_centric_json()


if __name__ == "__main__":
    completeTopicModelingPipeline(forceRefresh=True)
