import json

from models import Document, PositionalInvertedIndexOnMemory, IRData
import optimizer


def load_documents(file) -> list[Document]:
    """
    Reads a JSON file containing document data and returns a list of Document objects.

    Args:
        file (str): The path to the JSON file containing document data.

    Returns:
        list[Document]: A list of Document objects representing the documents in the file.


    JSON File Structure:
        The JSON file is expected to contain an object where each key represents a document identifier (doc_id)
        and the corresponding value is another object containing the document's metadata. The metadata object
        should have the following structure:
            * `title` (str): The title of the document.
            * `content` (str): The content of the document.
            * `url` (str, optional): The URL of the document.
            * `date` (str, optional): The date of the document.
            * `tags` (list of str, optional): A list of tags associated with the document.

    Example JSON File Structure:
    {
        "doc_id": {
            "title": "Title of the document",
            "content": "Content of the document",
            "url": "https://github.com/Aghajari",
            "date": "6/28/2024 5:35:28 PM",
            "tags": ["tag1", "tag2", ...]
        },
        ...
    }
    """
    with open(file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    docs = []
    for docId, data in json_data.items():
        docs.append(
            Document(
                doc_id=docId,
                title=data["title"],
                url=data["url"],
                content=data["content"],
                date=data["date"],
                tags=data["tags"],
            )
        )

    return docs


def create_pii(docs: list[Document]) -> PositionalInvertedIndexOnMemory:
    """
    Creates a positional inverted index (PII) from a list of Document objects.

    Args:
        docs (list[Document]): A list of Document objects representing the documents to be indexed.

    Returns:
        PositionalInvertedIndexOnMemory: The created positional inverted index object.
    """
    pii = PositionalInvertedIndexOnMemory()
    for doc in docs:
        for token, weight in doc.get_weighted_tokens():
            pii.__add_token__(
                token_str=token,
                doc_id=doc.id,
                position=-1,
                weight=weight,
            )

        tokens = doc.get_tokens()
        for i in range(len(tokens)):
            pii.__add_token__(
                token_str=tokens[i],
                doc_id=doc.id,
                position=i,
            )

    pii.__finalize__(len(docs))
    return pii


def create_query_pii(query: str) -> PositionalInvertedIndexOnMemory:
    """
    Creates a positional inverted index for a given query string.

    Args:
        query (str): The query string to be indexed.

    Returns:
        PositionalInvertedIndexOnMemory: The created positional inverted index object for the query.
    """
    return create_pii([
        Document(
            doc_id="0",
            title="",
            url="",
            content=query,
            date=None
        )
    ])


def create(file, cache=True, champions_list_r=None) -> IRData:
    """
    Creates an IRData object containing the positional inverted index (PII) and document data.

    This function first attempts to read the PII from the cache. If successful, it uses the cached PII.
    Otherwise, it creates the PII from the provided document data.

    Args:
        file (str): The path to the JSON file containing document data.
        cache (bool, optional): Whether to use the cache for the PII (defaults to True).
        champions_list_r (int, optional): The number of top documents to include in champions lists
            for each term (used for ranking, defaults to None).

    Returns:
        IRData: The created IRData object containing the PII and document data.
    """
    docs = load_documents(file)
    if cache:
        cache_pii = optimizer.read_from_cache(file)
        if cache_pii is not None:
            ir = IRData(cache_pii, docs)
            optimizer.generate_champions_list(ir, champions_list_r)
            return ir

    pii = create_pii(docs)

    if cache:
        optimizer.write_cache(pii, file)

    ir = IRData(pii, docs)
    optimizer.generate_champions_list(ir, champions_list_r)
    return ir
