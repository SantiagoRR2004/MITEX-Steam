from datetime import datetime
import time
import requests
import json
import os
import re


def get_search_results(params):
    req_sr = requests.get(
        "https://store.steampowered.com/search/results/", params=params
    )

    if req_sr.status_code != 200:
        print(f"Failed to get search results: {req_sr.status_code}")
        return {"items": []}

    try:

        search_results = req_sr.json()
    except Exception as e:
        print(f"Failed to parse search results: {e}")
        return {"items": []}

    return search_results


page_list = list(range(1, 5))

params_sr_default = {
    "filter": "globaltopsellers",
    "hidef2p": 1,
    "page": 1,  # page is used to go through different parts of the ranking. Each page contains 25 results
    "json": 1,
}
currentDirectory = os.path.dirname(os.path.abspath(__file__))
videogame_json_path = os.path.join(currentDirectory, "videogames.json")
videogame_json = json.load(open(videogame_json_path, "r"))

for page_no in page_list:
    param = params_sr_default.copy()
    param["page"] = page_no

    search_results = get_search_results(param)

    if not search_results:
        continue

    items = search_results.get("items", [])

    # proprocessing search results to retrieve the appid of the game
    for item in items:
        try:
            name = item["name"]
            appid = re.search(r"steam/\w+/(\d+)", item["logo"]).group(
                1
            )  # the URL can be steam/bundles/{appid} or steam/apps/{appid}
            # Save the appid in the videogame_json for later use
            videogame_json[name] = int(appid)
        except Exception as e:
            print(f"Failed to extract appid: {e}")
            item["appid"] = None

videogame_json = dict(
    sorted(videogame_json.items(), key=lambda x: x[0])
)  # sort by name
# save the search results
with open(videogame_json_path, "w", encoding="utf-8") as f:
    json.dump(videogame_json, f, indent=2, ensure_ascii=False)
    f.write("\n")
