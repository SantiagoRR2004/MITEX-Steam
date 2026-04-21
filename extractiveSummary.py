from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import networkx as nx
import dataHandling
import numpy as np
import json
import tqdm
import os


def extractiveSummary() -> None:
    """
    Perform extractive summarization on the cleaned reviews using a graph-based approach.

    Args:
        - None

    Returns:
        - None
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    summaryFolder = os.path.join(currentDirectory, "summaryData")
    os.makedirs(summaryFolder, exist_ok=True)

    reviewPaths = dataHandling.obtainReviewStructure(
        os.path.join(currentDirectory, "cleanData")
    )

    multilingualModel = SentenceTransformer("all-MiniLM-L6-v2")

    for game, reviewTypes in tqdm.tqdm(reviewPaths.items(), desc="Summarizing reviews"):

        for reviewType, path in reviewTypes.items():
            sentences = []

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    review = eval(line)["review"]
                    sentences.extend(
                        [
                            sentence.strip()
                            for sentence in sent_tokenize(review, language="english")
                            if sentence.strip()
                        ]
                    )

            # Build sentence embeddings and pairwise cosine similarity matrix
            embeddings = multilingualModel.encode(sentences)
            similarity = cosine_similarity(embeddings)

            similarityThreshold = 0.35
            graph = nx.Graph()
            graph.add_nodes_from(range(len(sentences)))

            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    weight = float(similarity[i][j])

                    if weight >= similarityThreshold:
                        graph.add_edge(i, j, weight=weight)

            # Use PageRank to rank sentence centrality in the semantic graph
            scores = nx.pagerank(graph, alpha=0.85, weight="weight")
            topSentences = sorted(
                scores.items(), key=lambda item: item[1], reverse=True
            )[:5]

            # Save output to JSON file
            results = [sentences[index] for index, score in topSentences]

            with open(
                os.path.join(summaryFolder, f"{game}{reviewType.capitalize()}.json"),
                "w",
            ) as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                f.write("\n")


if __name__ == "__main__":
    extractiveSummary()
