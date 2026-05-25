from rank_bm25 import BM25Okapi
import dataHandling
import chromadb
import hashlib
import models
import os
import json

databaseName = "videogames"


def createCollection() -> chromadb.api.models.Collection.Collection:
    """
    Create a ChromaDB collection.

    Args:
        - None

    Returns:
        - chromadb.api.models.Collection.Collection: The created or accessed ChromaDB collection.
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))

    # https://docs.trychroma.com/reference/python/client#persistentclient
    client = chromadb.PersistentClient(path=os.path.join(currentDirectory, "db"))

    try:
        collection = client.get_collection(name=databaseName)

    except chromadb.errors.NotFoundError:
        print(f"Collection {databaseName} not found. Creating a new one.")
        # hnsw:space": "cosine" define the distance function of HNSW (Hierarchical Navigable Small World)
        collection = client.create_collection(
            name=databaseName, metadata={"hnsw:space": "cosine"}
        )

    return collection


COLLECTION = createCollection()


def addReviewToCollection(reviews: list[str], videogame: str, reviewType: str) -> None:
    """
    Add multiple reviews to the ChromaDB collection.

    Args:
        - reviews: A list of review texts to be added.
        - videogame: The name of the videogame associated with the reviews.
        - reviewType: The type of the reviews.

    Returns:
        - None

    Título: [Nombre] | Género: [Géneros]
    """
    newReviews = []
    reviewIds = []
    newMetadata = []
    seenIds = set()

    # Check which reviews are new
    for review in reviews:
        text = f"{videogame}:{review}"
        reviewID = hashlib.sha256(text.encode()).hexdigest()

        if reviewID in seenIds:
            continue

        seenIds.add(reviewID)

        existing = COLLECTION.get(ids=[reviewID])

        if not existing["ids"]:
            newReviews.append(review)
            reviewIds.append(reviewID)
            newMetadata.append({"videogame": videogame, "type": reviewType})

    # Calculate embeddings for all new reviews at once
    if newReviews:
        embeddings = models.EMBEDDING_MODEL.encode(newReviews).tolist()

        COLLECTION.add(
            embeddings=embeddings,
            documents=newReviews,
            metadatas=newMetadata,
            ids=reviewIds,
        )


def addDocsToCollection(forceRefresh: bool = False) -> None:
    """
    Add documents to the ChromaDB collection.

    Args:
        - forceRefresh (bool): Whether to force refresh the data. Default is False.
            Takes a long time if set to True.

    Returns:
        - None
    Descripción: [La describción corta (podemos coger la versión larga pero habría que limpiar el html]
    Resumen Reseñas Positivas: []
    Resumen Reseñas Negativas: []
    Tópicos asociados: [Tópico X (peso %), Tópico Y (peso %)]
    Y viendo el json podemos añadirle el precio, pegi y plataformas jugables
    """
    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    reviewsFiles = dataHandling.obtainReviewStructure(
        os.path.join(currentDirectory, "summaryData")
    )
    videogames_info = os.path.join(currentDirectory, "rawData")

    # Load topic structures
    gt = json.load(
        open(
            os.path.join(currentDirectory, "topicsData", "gameTopics.json"),
            "r",
            encoding="utf-8",
        )
    )
    topics_meta, games_to_topics = gt["topics"], gt["games"]

    newDocuments = []
    newIds = []
    newMetadatas = []

    for file in filter(lambda f: f.endswith("Info.json"), os.listdir(videogames_info)):
        info = json.load(
            open(os.path.join(videogames_info, file), "r", encoding="utf-8")
        )
        v_game = info["name"]
        if v_game not in reviewsFiles:
            continue
        reviewsPath = reviewsFiles[v_game]
        reviews = {}
        for reviewType, path in reviewsPath.items():
            summary = json.load(open(path, "r", encoding="utf-8"))
            reviews[reviewType] = "\n".join(f"- {r.strip()}" for r in summary)

        pos = reviews["positive"]
        neg = reviews["negative"]
        desc = info["short_description"]
        plat_data = info.get("platforms", {})
        plats = ", ".join([k.capitalize() for k, v in plat_data.items() if v])
        pegi = info.get("ratings", {}).get("pegi", {}).get("rating", "RP")
        genres_list = info.get("genres", [])
        genres = ", ".join([g["description"] for g in genres_list])

        # get price and transform it to float if possible
        price_overview = info.get("price_overview")
        if price_overview:
            price_val = float(price_overview.get("final", 0)) / 100.0
        else:
            price_val = 0.0

        docID = hashlib.sha256(v_game.encode()).hexdigest()

        if not forceRefresh:
            existing = COLLECTION.get(ids=[docID])

            if existing and existing["ids"]:
                continue

        sorted_t = sorted(
            games_to_topics.get(v_game, {}).items(),
            key=lambda x: x[1].get("percentage", 0),
            reverse=True,
        )
        top_str = "\n".join(
            f"- Theme: {topics_meta.get(str(tid), {})['title']} Description: {topics_meta.get(str(tid), {})['description']} ({d.get('percentage', 0)}%)"
            for tid, d in games_to_topics.get(v_game, {}).items()
        )

        document = (
            f"Title: {v_game} | Genres: {genres}\n"
            f"Description: {desc}\n\n"
            f"Positive Reviews Summary:\n{pos}\n\n"
            f"Negative Reviews Summary:\n{neg}\n\n"
            f"Associated Topics:\n{top_str or '- None.'}"
        )

        metadata = {
            "videogame": v_game,
            "type": "game_summary",
            "platforms": plats,
            "pegi": str(pegi),
            "price": price_val,
            "genres": genres,
        }

        newDocuments.append(document)
        newIds.append(docID)
        newMetadatas.append(metadata)
    if newDocuments:
        embeddings = models.EMBEDDING_MODEL.encode(newDocuments).tolist()

        if forceRefresh:
            COLLECTION.upsert(
                embeddings=embeddings,
                documents=newDocuments,
                metadatas=newMetadatas,
                ids=newIds,
            )
        else:
            COLLECTION.add(
                embeddings=embeddings,
                documents=newDocuments,
                metadatas=newMetadatas,
                ids=newIds,
            )


def rrf(
    query: str,
    k: int = 60,
    nResults: int = 15,
) -> set:
    """
    Reciprocal Rank Fusion (RRF) implementation that combines BM25 and semantic search results.

    Args:
        - query: The search query string.
        - k: The RRF parameter that controls the influence of the rank (default is 60).
        - nResults: The number of top results to return (default is 15).

    Returns:
        - set: A set of videogame names that are relevant to the query based on the RRF ranking.
    """
    collectionData = COLLECTION.get()
    chroma_docs = collectionData["documents"]
    chroma_ids = collectionData["ids"]
    id_to_index = {doc_id: index for index, doc_id in enumerate(chroma_ids)}

    tokenized = [d.lower().split() for d in chroma_docs]
    bm25 = BM25Okapi(tokenized)

    # BM25 top 50
    scores = bm25.get_scores(query.lower().split())
    bm25_rank_indices = scores.argsort()[::-1][:50]

    # Semantic top 50
    q_emb = models.EMBEDDING_MODEL.encode([query]).tolist()
    sem = COLLECTION.query(query_embeddings=q_emb, n_results=50)
    sem_ids = sem["ids"][0]

    # RRF
    k = 60
    # Create a list of zeros with the same lenght as chroma_docs
    rrf = [0.0] * len(chroma_docs)

    for r, doc_id in enumerate(bm25_rank_indices):
        # As Wsem=Wlex=0.5, we can omit them in the calculation.
        # Add 1 to the denominator to avoid division by zero when r=0.
        rrf[doc_id] += 1 / (k + r + 1)

    for r, doc_id in enumerate(sem_ids):
        rrf[id_to_index[doc_id]] += 1 / (k + r + 1)

    # List of tuples (doc_id, score) sorted by score in descending order
    top = sorted(enumerate(rrf), key=lambda x: x[1], reverse=True)

    foundVideogames = set()

    while len(foundVideogames) < nResults and top:
        docID, _ = top.pop(0)
        m = COLLECTION.get(ids=[chroma_ids[docID]])
        foundVideogames.add(m["metadatas"][0]["videogame"])

    return foundVideogames


if __name__ == "__main__":
    addDocsToCollection(forceRefresh=True)
