from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import math
import networkx as nx
import dataHandling
import models
import json
import tqdm
import os


def extractiveSummaryFile(
    inputPath: str, outputPath: str, videogame: str, reviewType: str
) -> None:
    """
    Perform extractive summarization on the reviews in the input file
    and save the summary to the output file.

    For the personalization we use the scores that Steam gives to each review.

    This was mean to be parallelized, but it becomes slower.

    Args:
        - inputPath (str): The path to the input JSONL file containing the reviews.
        - outputPath (str): The path to the output JSON file where the summary will be written.
        - videogame (str): The name of the videogame associated with the reviews.
        - reviewType (str): The type of the reviews (e.g., "positive", "negative").

    Returns:
        - None
    """
    sentences = []
    personalization = {}

    with open(inputPath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            review = data["review"]
            reviewScore = data["score"]
            review = eval(line)["review"]

            newSentences = [
                sentence.strip()
                for sentence in sent_tokenize(review, language="english")
                if sentence.strip()
            ]
            num_sentences = len(newSentences)
            distributed_scores = reviewScore / (1 + math.log(num_sentences))
            start_idx = len(sentences)
            sentences.extend(newSentences)
            personalization.update(
                {start_idx + i: distributed_scores for i in range(num_sentences)}
            )

    # Build sentence embeddings and pairwise cosine similarity matrix
    embeddings = models.EMBEDDING_MODEL.encode(sentences)
    similarity = cosine_similarity(embeddings)

    similarityThreshold = 0.3
    graph = nx.Graph()
    graph.add_nodes_from(range(len(sentences)))

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            weight = float(similarity[i][j])

            if weight >= similarityThreshold:
                graph.add_edge(i, j, weight=weight)

    # Use PageRank to rank sentence centrality in the semantic graph
    scores = nx.pagerank(
        graph, alpha=0.85, weight="weight", personalization=personalization
    )
    topSentences = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]

    # Save output to JSON file
    results = [sentences[index] for index, score in topSentences]

    with open(
        outputPath,
        "w",
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")


def extractiveSummary(forceRefresh: bool = False) -> None:
    """
    Perform extractive summarization on the cleaned reviews using a graph-based approach.

    Args:
        - forceRefresh (bool): Whether to refresh the summary data even if it already exists.


    Returns:
        - None
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    summaryFolder = os.path.join(currentDirectory, "summaryData")
    os.makedirs(summaryFolder, exist_ok=True)

    reviewPaths = dataHandling.obtainReviewStructure(
        os.path.join(currentDirectory, "cleanData")
    )

    progressBar = tqdm.tqdm(
        total=sum(len(paths) for paths in reviewPaths.values()),
        desc="Summarizing reviews",
    )

    for game, reviewTypes in reviewPaths.items():

        for reviewType, path in reviewTypes.items():

            outputPath = os.path.join(
                summaryFolder, f"{game}{reviewType.capitalize()}.json"
            )

            if forceRefresh or not os.path.exists(outputPath):
                extractiveSummaryFile(
                    inputPath=path,
                    outputPath=outputPath,
                    videogame=game,
                    reviewType=reviewType,
                )

            progressBar.update(1)

    progressBar.close()


if __name__ == "__main__":
    extractiveSummary()
