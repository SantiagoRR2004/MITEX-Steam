from rank_bm25 import BM25Okapi
import chromadb
import hashlib
import models
import os

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
    chroma_docs = COLLECTION.get()[databaseName]
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
        rrf[int(doc_id)] += 1 / (k + r + 1)

    # List of tuples (doc_id, score) sorted by score in descending order
    top = sorted(enumerate(rrf), key=lambda x: x[1], reverse=True)

    foundVideogames = set()

    while len(foundVideogames) < nResults and top:
        docID, _ = top.pop(0)
        m = COLLECTION.get(ids=[str(docID)])
        foundVideogames.add(m["metadatas"][0]["videogame"])

    return top
