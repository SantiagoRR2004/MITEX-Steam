import requests
import json
import os


def fetch_reviews(id: int, videogame: str) -> None:
    """
    Fetches reviews for a given video game from the Steam store
    and saves them as JSONL files.

    Args:
        - id (int): The Steam app ID of the video game.
        - videogame (str): The name of the video game, used for naming the output files.

    Returns:
        - None
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataDirectory = os.path.join(currentDirectory, "data")
    os.makedirs(dataDirectory, exist_ok=True)

    url = f"https://store.steampowered.com/appreviews/{id}?json=1"
    parameters = {"filter": "all", "language": "english", "num_per_page": 100}

    for reviewType in ["positive", "negative"]:
        parameters["review_type"] = reviewType

        response = requests.get(url, params=parameters)

        if response.status_code == 200:
            data = response.json()

            outputPath = os.path.join(
                dataDirectory, f"{videogame}{reviewType.capitalize()}.jsonl"
            )
            with open(outputPath, "w", encoding="utf-8") as file:

                for review in data["reviews"]:
                    important = {
                        "positive": review["voted_up"],
                        "score": float(review["weighted_vote_score"]),
                        "review": review["review"],
                    }
                    json.dump(important, file, ensure_ascii=False)
                    file.write("\n")


terrariaID = 105600
fetch_reviews(terrariaID, "Terraria")
