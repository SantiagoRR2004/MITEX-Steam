import requests
import random
import json
import time
import tqdm
import os


def fetchReviews(id: int, videogame: str) -> None:
    """
    Fetches reviews for a given video game from the Steam store
    and saves them as JSONL files.

    https://partner.steamgames.com/doc/store/getreviews

    Args:
        - id (int): The Steam app ID of the video game.
        - videogame (str): The name of the video game, used for naming the output files.

    Returns:
        - None
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataDirectory = os.path.join(currentDirectory, "rawData")
    os.makedirs(dataDirectory, exist_ok=True)

    url = f"https://store.steampowered.com/appreviews/{id}?json=1"
    parameters = {"filter": "all", "language": "english", "num_per_page": 100}

    for reviewType in ["positive", "negative"]:

        outputPath = os.path.join(
            dataDirectory, f"{videogame}{reviewType.capitalize()}.jsonl"
        )

        # Only fetch reviews if the output file doesn't already exist
        if not os.path.exists(outputPath):

            parameters["review_type"] = reviewType
            response = requests.get(url, params=parameters)
            time.sleep(random.random())

            if response.status_code == 200:
                data = response.json()

                if data["success"] == 1:

                    with open(outputPath, "w", encoding="utf-8") as file:

                        for review in data["reviews"]:
                            important = {
                                "score": float(review["weighted_vote_score"]),
                                "review": review["review"],
                            }
                            json.dump(important, file, ensure_ascii=False)
                            file.write("\n")


def getAllGames() -> None:
    """
    Fetches reviews for all video games listed in the "videogames.json" file.

    Args:
        - None

    Returns:
        - None
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    videogamesPath = os.path.join(currentDirectory, "videogames.json")

    with open(videogamesPath, "r", encoding="utf-8") as file:
        videogames = json.load(file)

    for videogame, id in tqdm.tqdm(videogames.items(), desc="Fetching reviews"):
        fetchReviews(id, videogame)


getAllGames()
