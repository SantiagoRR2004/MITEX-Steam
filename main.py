import requests
import json


terrariaID = 105600
url = f"https://store.steampowered.com/appreviews/{terrariaID}?json=1"


parameters = {
    "filter": "all",
    "language": "english",
    "num_per_page": 100
}

for reviewType in ["positive", "negative"]:
    parameters["review_type"] = reviewType

    response = requests.get(url, params=parameters)

    if response.status_code == 200:
        data = response.json()

        with open(f"{reviewType}.json", "w") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
