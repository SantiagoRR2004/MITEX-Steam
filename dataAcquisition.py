import requests
import random
import json
import time
import tqdm
import os
import re


def steamSearchResults(params: dict) -> dict:
    """
    Fetches search results from the Steam store based on the provided parameters.

    Args:
        - params (dict): A dictionary of parameters to be sent with the search request.

    Returns:
        - dict: A dictionary containing the search results from the Steam store.
    """
    response = requests.get(
        "https://store.steampowered.com/search/results/", params=params
    )

    if response.status_code != 200:
        print(f"Failed to get search results: {response.status_code}")
        return {}

    try:
        results = response.json()

    except Exception as e:
        print(f"Failed to parse search results: {e}")
        return {}

    return results


def updateVideogames() -> None:
    """
    Fetches the top 100 video games from the Steam store and updates
    the "videogames.json" file with their names and app IDs.

    Args:
        - None

    Returns:
        - None
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    videogamesFile = os.path.join(currentDirectory, "videogames.json")

    with open(videogamesFile, "r", encoding="utf-8") as f:
        videogames = json.load(f)

    defaultParameters = {
        "filter": "globaltopsellers",
        "hidef2p": 1,
        "page": None,  # Page is used to go through different parts of the ranking. Each page contains 25 results
        "json": 1,
    }

    for pageNum in range(1, 5):

        defaultParameters["page"] = pageNum
        searchResults = steamSearchResults(defaultParameters)

        if not searchResults.get("items"):
            continue

        # Processing search results to retrieve the appid of the game
        for item in searchResults["items"]:

            try:
                name = item["name"]

                # The URL can be steam/bundles/{appid} or steam/apps/{appid}
                appid = re.search(r"steam/\w+/(\d+)", item["logo"]).group(1)

                # Save the appid in the videogame json for later use
                videogames[name] = int(appid)

            except Exception as e:
                print(f"Failed to extract appid: {e}")
                item["appid"] = None

    # Sort by name
    videogames = dict(sorted(videogames.items(), key=lambda x: x[0]))

    # Save the search results
    with open(videogamesFile, "w", encoding="utf-8") as f:
        json.dump(videogames, f, indent=2, ensure_ascii=False)
        f.write("\n")


def fetchReviews(id: int, videogame: str, forceRefresh: bool = False) -> None:
    """
    Fetches reviews for a given video game from the Steam store
    and saves them as JSONL files.

    https://partner.steamgames.com/doc/store/getreviews

    Args:
        - id (int): The Steam app ID of the video game.
        - videogame (str): The name of the video game, used for naming the output files.
        - forceRefresh (bool): Whether to fetch reviews even if the output file already exists.

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

        if not os.path.exists(outputPath) or forceRefresh:

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


def fetchGameInfo(id: int, videogame: str, forceRefresh: bool = False) -> None:
    """
    Fetches detailed information for a given video game from the Steam store
    and saves it in a JSON file.

    Args:
        - id (int): The Steam app ID of the video game.
        - videogame (str): The name of the video game, used for naming the output file.
        - forceRefresh (bool): Whether to fetch game info even if the output file already exists.

    Returns:
        - None
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    dataDirectory = os.path.join(currentDirectory, "rawData")
    gameInfoPath = os.path.join(dataDirectory, f"{videogame}Info.json")

    if not os.path.exists(gameInfoPath) or forceRefresh:
        url = f"https://store.steampowered.com/api/appdetails?appids={id}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data[str(id)]["success"]:
                with open(gameInfoPath, "w", encoding="utf-8") as file:
                    json.dump(data[str(id)]["data"], file, indent=2, ensure_ascii=False)
                    file.write("\n")


def getAllGames(forceRefresh: bool = False) -> None:
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

    for videogame, id in tqdm.tqdm(
        videogames.items(), desc="Fetching reviews and info"
    ):
        fetchReviews(id, videogame, forceRefresh=forceRefresh)
        fetchGameInfo(id, videogame, forceRefresh=forceRefresh)


if __name__ == "__main__":
    updateVideogames()
