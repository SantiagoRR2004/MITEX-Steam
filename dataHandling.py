from concurrent.futures import ProcessPoolExecutor, as_completed
from langdetect import detect_langs, LangDetectException
import json
import tqdm
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

    # Check that the input file is not empty
    if os.path.getsize(inputPath) == 0:
        return {f"{videogame} {reviewType.capitalize()}": differentLanguages}

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

    # If the output file is empty, remove it
    if os.path.getsize(outputPath) == 0:
        os.remove(outputPath)

    return {f"{videogame} {reviewType.capitalize()}": differentLanguages}


def keepRelevantInfo(inputPath: str, outputPath: str) -> None:
    """
    Extracts and keeps only the relevant information from the input JSON
    file and writes it to the output JSON file.

    Args:
        - inputPath (str): The path to the input JSON file containing the video game information.
        - outputPath (str): The path to the output JSON file where the relevant information will
            be written.

    Returns:
        - None
    """
    with open(inputPath, "r", encoding="utf-8") as infile:
        info = json.load(infile)

    relevantInfo = {
        "name": info["name"],
        "description": info["short_description"],
        "platforms": ", ".join(
            [k.capitalize() for k, v in info.get("platforms", {}).items() if v]
        ),
        "pegi": ((info.get("ratings") or {}).get("pegi") or {}).get("rating", "RP"),
        "genres": ", ".join([g["description"] for g in info.get("genres", [])]),
        "price": float(info.get("price_overview", {}).get("final", 0)) / 100.0,
    }

    with open(outputPath, "w", encoding="utf-8") as outfile:
        json.dump(relevantInfo, outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")


def obtainPreviousLanguageLog(videogame: str, reviewType: str) -> dict:
    """
    Obtains the language log for a given video game and review type from the existing language log file.

    This should only be used if the cleaned data already exists.

    Args:
        - videogame (str): The name of the video game.
        - reviewType (str): The type of reviews ("positive" or "negative").

    Returns:
        - dict: A dictionary containing the non-English languages found
            in the reviews for the given video game and review type.
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    languageLogPath = os.path.join(currentDirectory, "languageLog.json")

    if not os.path.exists(languageLogPath):
        return {}

    with open(languageLogPath, "r", encoding="utf-8") as f:
        languageLog = json.load(f)

    fullName = f"{videogame} {reviewType.capitalize()}"

    return {fullName: languageLog.get(fullName, {})}


def obtainReviewStructure(folderPath: str, fileEnding: str = ".jsonl") -> dict:
    """
    Scans the folder for files containing positive and negative reviews of video games.

    Args:
        - folderPath (str): The path to the folder containing the review files.
        - fileEnding (str): The ending of the review files to scan for.

    Returns:
        - dict: A dictionary mapping each video game name to the file paths of its positive and negative reviews.
    """
    reviewFiles = {}

    for filename in os.listdir(folderPath):
        if filename.endswith(fileEnding):

            if filename.endswith(f"Positive{fileEnding}"):
                reviewType = "positive"
                name = filename[: -len(f"Positive{fileEnding}")]

            elif filename.endswith(f"Negative{fileEnding}"):
                reviewType = "negative"
                name = filename[: -len(f"Negative{fileEnding}")]

            else:
                print(f"Unexpected file format: {filename}")

            if name not in reviewFiles:
                reviewFiles[name] = {}

            reviewFiles[name][reviewType] = os.path.join(folderPath, filename)

    return reviewFiles


def cleanData(forceRefresh: bool = False) -> None:
    """
    Remove the reviews that are not in English and
    only keep the information that we need from each game.

    Args:
        - forceRefresh (bool): Whether to refresh the cleaned data even if it already exists.

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

            keepRelevantInfo(
                inputPath=os.path.join(currentDirectory, "rawData", f"{game}Info.json"),
                outputPath=os.path.join(cleanDataFolder, f"{game}Info.json"),
            )

            for t in reviewPaths[game].keys():
                inputPath = reviewPaths[game][t]
                outputPath = os.path.join(
                    cleanDataFolder, f"{game}{t.capitalize()}.jsonl"
                )

                if forceRefresh or not os.path.exists(outputPath):
                    futures.append(
                        executor.submit(
                            cleanFile, inputPath, outputPath, game, reviewType=t
                        )
                    )

                else:
                    futures.append(
                        executor.submit(obtainPreviousLanguageLog, game, reviewType=t)
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
    cleanData()
