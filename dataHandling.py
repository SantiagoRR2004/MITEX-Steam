from concurrent.futures import ProcessPoolExecutor, as_completed
from langdetect import detect_langs, LangDetectException
import json
import tqdm
import re
import os


def cleanFile(inputPath: str, outputPath: str, videogame: str, reviewType: str) -> dict:
    """
    Cleans the reviews in the input file and writes the cleaned reviews to the output file.

    Args:
        - inputPath (str): The path to the input JSONL file containing the reviews.
        - outputPath (str): The path to the output JSONL file where the cleaned reviews will be written.
        - videogame (str): The name of the video game for which to clean reviews.
        - reviewType (str): The type of reviews to clean ("positive" or "negative").

    Returns:
        - dict: A dictionary containing the non-English languages found in the cleaned reviews.
    """
    differentLanguages = {}

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
                            differentLanguages[lang] = (
                                differentLanguages.get(lang, 0) + 1
                            )

            except LangDetectException as e:
                differentLanguages["gibberish"] = (
                    differentLanguages.get("gibberish", 0) + 1
                )

    return {f"{videogame} {reviewType.capitalize()}": differentLanguages}


def obtainReviewStructure(folderPath: str) -> dict:
    """
    Scans the folder for JSONL files containing positive and negative reviews of video games.

    Args:
        - folderPath (str): The path to the folder containing the review files.

    Returns:
        - dict: A dictionary mapping each video game name to the file paths of its positive and negative reviews.
    """
    reviewFiles = {}

    for filename in os.listdir(folderPath):
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

            reviewFiles[name][reviewType] = os.path.join(folderPath, filename)

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

    reviewPaths = obtainReviewStructure(os.path.join(currentDirectory, "rawData"))

    futures = []
    languageLog = {}
    progressBar = tqdm.tqdm(total=2 * len(reviewPaths), desc="Processing files")

    with ProcessPoolExecutor() as executor:

        for game in reviewPaths.keys():

            for t in ["positive", "negative"]:
                inputPath = reviewPaths[game][t]
                outputPath = os.path.join(
                    cleanDataFolder, f"{game}{t.capitalize()}.jsonl"
                )

                futures.append(
                    executor.submit(
                        cleanFile, inputPath, outputPath, game, reviewType=t
                    )
                )

        for future in as_completed(futures):
            result = future.result()
            languageLog.update(result)
            progressBar.update(1)

    progressBar.close()

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
