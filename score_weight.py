import math
from utils import parse_date
from datetime import datetime

# Constants defining weights for ranking factors in search results
PHRASE_QUERY_WEIGHT = 2  # Maximum weight for exact phrase matches
TITLE_TOKEN_WEIGHT = 2  # Weight for tokens found in document title
TAG_TOKEN_WEIGHT = 1.5  # Weight for tokens found in document tags
DATE_WEIGHT = 0.4  # Weight for document recency (newer gets a boost)

# Optional threshold for eliminating terms from the search index if they appear too frequently
INDEX_ELIMINATION = None


def calculate_tf(linear_tf):
    """
    Calculates Term Frequency (TF) for a term within a document.

    TF reflects how often a term appears in the document. This function uses a logarithmic scale
    (base 2) to reward frequent terms but avoid giving excessive weight to very frequent ones.

    Args:
        linear_tf: The number of times the term appears in the document (linear count).

    Returns:
        The calculated TF value.
    """
    return 0 if linear_tf == 0 else \
        1 + math.log2(linear_tf)


def calculate_idf(number_of_docs_contain_token, number_of_docs):
    """
    Calculates Inverse Document Frequency (IDF) for a term.

    IDF reflects how common a term is across all documents. Terms appearing in very few documents
    get a higher IDF score, indicating they are more specific and potentially more relevant to the search.

    Args:
        number_of_docs_contain_token: Number of documents containing the term.
        number_of_docs: Total number of documents in the collection.

    Returns:
        The calculated IDF value.
    """
    return 0 if number_of_docs_contain_token == 0 else \
        math.log2(number_of_docs / number_of_docs_contain_token)


def date_score(date: str, max_date: datetime, min_date: datetime):
    """
    Calculates a score based on the document's date relative to the collection's date range.

    Args:
        date: String representing the document's date.
        max_date: The latest date in the document collection (datetime object).
        min_date: The earliest date in the document collection (datetime object).

    Returns:
        A weighted score based on the document's date (newer documents score higher).
    """
    date_obj = parse_date(date, min_date)
    min_timestamp = min_date.timestamp()
    total_delta = max(1.0, max_date.timestamp() - min_timestamp)
    time_delta = date_obj.timestamp() - min_timestamp
    weighted_score = abs(time_delta / total_delta) * DATE_WEIGHT
    return weighted_score
