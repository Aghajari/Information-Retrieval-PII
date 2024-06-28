from typing import Iterable

import score_weight
from models import IRData, PositionalInvertedIndexOnMemory, Document
from loader import create_query_pii
import heapq


class __SearchedDocument:
    """
    Representing a document with an associated score for comparison in heap
    """

    def __init__(self, doc, score):
        self.doc = doc
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

    def __le__(self, other):
        return self.score <= other.score

    def __gt__(self, other):
        return self.score > other.score

    def __ge__(self, other):
        return self.score >= other.score


def cosine_score(
        pii: PositionalInvertedIndexOnMemory,
        query_pii: PositionalInvertedIndexOnMemory,
):
    """
    Calculate cosine similarity scores between documents and a query based on their
    positional inverted index representations.

    Args:
        pii (PositionalInvertedIndexOnMemory): Positional inverted index of documents.
        query_pii (PositionalInvertedIndexOnMemory): Positional inverted index of the query.

    Returns:
        dict: A dictionary where keys are document IDs and values are their respective scores.
    """
    doc_scores = {}

    for (token, query_token_data) in query_pii:
        wtq = query_token_data.list[0][1].tf
        try:
            token_data = pii[token]
        except KeyError:
            continue

        for (doc_id, doc_data) in token_data:
            wtd = doc_data.tf * token_data.idf
            if doc_id in doc_scores:
                doc_scores[doc_id] += wtd * wtq
            else:
                doc_scores[doc_id] = wtd * wtq

            if score_weight.INDEX_ELIMINATION is not None and \
                    wtd < score_weight.INDEX_ELIMINATION:
                break

    return doc_scores


def phrase_query(
        pii: PositionalInvertedIndexOnMemory,
        query_pii: PositionalInvertedIndexOnMemory,
        doc_ids: Iterable[str],
):
    """
    Perform phrase queries to find consecutive matches in documents.

    Args:
        pii (PositionalInvertedIndexOnMemory): Positional inverted index of documents.
        query_pii (PositionalInvertedIndexOnMemory): Positional inverted index of the query.
        doc_ids (Iterable[str]): Iterable of document IDs to search within.

    Returns:
        dict: A dictionary where keys are document IDs and values are their respective phrase scores.
    """
    len_of_query = len(query_pii.tokens)
    doc_scores = {}
    for doc_id in doc_ids:
        max_consecutive = 0
        token_pos = 0
        positions = None
        while token_pos + 1 < len_of_query:
            try:
                positions = pii[query_pii.tokens[token_pos][0]][doc_id].positions
                break
            except KeyError:
                token_pos += 1
                continue

        if positions is None or token_pos + 1 >= len_of_query:
            continue

        for start_pos in positions:
            if start_pos == -1:
                continue

            pos = start_pos
            match_length = 1

            for token, _ in query_pii.tokens[token_pos + 1:]:
                try:
                    sp = pii[token][doc_id].positions
                except KeyError:
                    break

                pos += 1
                if pos in sp:
                    match_length += 1
                else:
                    break

            max_consecutive = max(max_consecutive, match_length)

        if max_consecutive > 1:
            doc_scores[doc_id] = 1 + score_weight.PHRASE_QUERY_WEIGHT * \
                                 (max_consecutive / len_of_query)

    return doc_scores


def search(
        ir: IRData,
        query: str,
        k: int = 10,
        date_score: bool = True,
        phrase_query_score: bool = True,
        score_function=cosine_score,
) -> list[tuple[Document, float]]:
    """
    Perform a document search using a given Information Retrieval (IR) data structure and query.

    Args:
        ir (IRData): Information Retrieval data containing documents and indices.
        query (str): The query string to search for.
        k (int, optional): The number of top results to return. Defaults to 10.
        date_score (bool, optional): Whether to consider date-based scoring. Defaults to True.
        phrase_query_score (bool, optional): Whether to include phrase query scoring. Defaults to True.
        score_function (function, optional): The scoring function to use. Defaults to cosine_score.

    Returns:
        list[tuple[Document, float]]: A list of tuples where each tuple contains a Document object
        and its corresponding relevance score.
    """
    query_pii = create_query_pii(query)

    doc_scores = score_function(ir.pii, query_pii)
    phrase_scores = None
    if phrase_query_score:
        phrase_scores = phrase_query(ir.pii, query_pii, doc_scores.keys())
    doc_heap = []

    for doc_id, score_value in doc_scores.items():
        doc = ir.docs[doc_id]
        normalized_score = score_value / ir.doc_lengths[doc_id]
        if phrase_scores is not None:
            normalized_score *= phrase_scores.get(doc_id, 1)

        if date_score:
            normalized_score += score_weight.date_score(doc.date, ir.max_doc_date, ir.min_doc_date)
        heapq.heappush(doc_heap, __SearchedDocument(doc, normalized_score))

    return list(map(
        lambda sd: (sd.doc, sd.score),
        heapq.nlargest(k, doc_heap, key=lambda sd: sd.score)
    ))
