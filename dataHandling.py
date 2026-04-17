from langdetect import detect_langs, LangDetectException
import json
import tqdm
import re
import os


def obtainReviewStructure() -> dict:
    """
    Scans the "rawData" folder for JSONL files containing positive and negative reviews of video games.

    Args:
        - None

    Returns:
        - dict: A dictionary mapping each video game name to the file paths of its positive and negative reviews.
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    rawDataFolder = os.path.join(currentDirectory, "rawData")

    reviewFiles = {}

    for filename in os.listdir(rawDataFolder):
        if filename.endswith(".jsonl"):

            if filename.endswith("Positive.jsonl"):
                reviewType = "positive"
                name = filename[: -len("Positive.jsonl")]

            elif filename.endswith("Negative.jsonl"):
                reviewType = "negative"
                name = filename[: -len("Negative.jsonl")]

            else:
                print(f"Unexpected file format: {filename}")

            if name not in reviewFiles:
                reviewFiles[name] = {}

            reviewFiles[name][reviewType] = os.path.join(rawDataFolder, filename)

    # Check that each game has both positive and negative reviews
    for name, reviewTypes in reviewFiles.items():
        assert (
            len(reviewTypes) == 2
        ), f"{name} does not have both positive and negative reviews."

    return reviewFiles


def onlyEnglish():
    """
    Remove the reviews that are not in English

    Args:
        - None

    Returns:
        - None
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    cleanDataFolder = os.path.join(currentDirectory, "cleanData")
    os.makedirs(cleanDataFolder, exist_ok=True)

    languageLog = {}
    reviewPaths = obtainReviewStructure()

    for game in tqdm.tqdm(reviewPaths.keys(), desc="Filtering reviews"):
        languageLog[game] = {}

        for t in ["positive", "negative"]:
            inputPath = reviewPaths[game][t]
            outputPath = os.path.join(cleanDataFolder, f"{game}{t.capitalize()}.jsonl")

            with open(inputPath, "r", encoding="utf-8") as infile, open(
                outputPath, "w", encoding="utf-8"
            ) as outfile:

                for line in infile:
                    lineData = json.loads(line)

                    try:
                        results = detect_langs(lineData["review"])
                        output = {r.lang: r.prob for r in results}

                        if "en" in output and output["en"] > 0.9:
                            json.dump(lineData, outfile, ensure_ascii=False)
                            outfile.write("\n")
                        else:
                            for lang, prob in output.items():
                                if lang != "en":
                                    languageLog[game][lang] = (
                                        languageLog[game].get(lang, 0) + 1
                                    )

                    except LangDetectException as e:
                        languageLog[game]["gibberish"] = (
                            languageLog[game].get("gibberish", 0) + 1
                        )

    # Sort language log by game name and then by language
    languageLog = dict(
        sorted(
            {
                game: dict(sorted(langs.items(), key=lambda x: x[0]))
                for game, langs in languageLog.items()
            }.items(),
            key=lambda x: x[0].lower(),
        )
    )

    with open(
        os.path.join(currentDirectory, "languageLog.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(languageLog, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    onlyEnglish()
