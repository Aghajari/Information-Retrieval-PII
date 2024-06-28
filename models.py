from typing import Iterator
from utils import *
import score_weight
import tokenizer
import math


class Document:
    """
    Represents a document within a search system.

    This class stores information about a document and provides methods to access its content
    and generate a list of tokens for searching.
    """

    def __init__(self, doc_id, title, url, content, date, tags: list[str] = None):
        """
        Initializes a new Document object.

        Args:
            doc_id: Unique identifier for the document.
            title: The document's title.
            url: The document's URL.
            content: The document's main content text.
            date: The document's date (format depends on parsing logic).
            tags: A list of optional tags associated with the document (defaults to None).
        """
        self.id = doc_id
        self.title = title
        self.url = url
        self.content = content
        self.date = date
        self.tags = tags

    def get_tokens(self) -> list[str]:
        """
        Returns a list of tokens extracted from the document's content.

        This method relies on tokenizer function to break down
        the content text into individual words or terms.

        Returns:
            A list of tokens (strings) representing the document's content.
        """
        return list(tokenizer.tokenize(self.content))

    def get_weighted_tokens(self):
        """
        Generates a list of tokens with weights based on their location in the document.

        This method extracts tokens from the title and optionally tags (if provided).
        Each token is assigned a weight based on its location:
            * Title tokens receive a weight defined by `score_weight.TITLE_TOKEN_WEIGHT`.
            * Tag tokens receive a weight defined by `score_weight.TAG_TOKEN_WEIGHT`.
            * Content tokens receive a weight of 1 (no specific weight applied).

        Returns:
            A list of tuples where the first element is the token (string) and the second element
            is its weight (float).
        """
        out = list(map(
            lambda x: (x, score_weight.TITLE_TOKEN_WEIGHT),
            tokenizer.tokenize(self.title)
        ))

        if self.tags is not None:
            for tag in self.tags:
                out += list(map(
                    lambda x: (x, score_weight.TAG_TOKEN_WEIGHT),
                    tokenizer.tokenize(tag)
                ))

        return out


class DocumentTokenData:
    """
    Represents information about a specific token within a document for a search system.

    This class stores data related to a token's occurrence in a single document, including:
        * Term Frequency (TF): The weighted frequency of the term within the document.
        * Linear Term Frequency (linear_tf): The raw count of the term's occurrences in the document.
        * Positions: A list of positions where the term appears within the document.
        * Weight: An internal weight used during TF calculation (defaults to 1).
    """

    def __init__(self, tf=-1, linear_tf=0, positions: list[int] = None):
        self.tf = tf
        self.linear_tf = linear_tf
        self.positions = list[int]() if positions is None else positions
        self._weight = 1

    def __add_position__(self, position: int, weight=1):
        """
        Adds a new position where the term appears in the document and updates internal data.

        Args:
            position: The position (integer index) of the term occurrence.
            weight: An optional weight to apply to this term in the document (defaults to 1).
        """
        self.positions.append(position)
        self.linear_tf += 1
        self._weight *= weight

    def __update_tf__(self):
        """
        Calculates the weighted Term Frequency (TF) for this token in the document.

        This method uses the `score_weight.calculate_tf` function and
        the internal weight to compute the final TF value.
        """
        self.tf = score_weight.calculate_tf(self.linear_tf) * self._weight

    def __iter__(self, *args, **kwargs):
        """
        Allows iterating over the positions where the term appears in the document.

        This method delegates iteration to the positions list.
        """
        return self.positions.__iter__()


class Token:
    """
    Represents a term (token) within the search system's vocabulary.

    This class stores information about a term across all documents, including:
        * Inverse Document Frequency (IDF): A score reflecting the term's rarity in the collection.
        * Linear Document Frequency (linear_df): The raw count of documents containing the term.
        * Positional List: A dictionary mapping document IDs to `DocumentTokenData` objects
          containing details about the term's occurrences in each document. (This can be
          replaced with a champions list for efficiency)
        * Champions List: An optional list containing document IDs and corresponding
          `DocumentTokenData` objects for the top documents (champions) where the term appears.
    """

    def __init__(
            self,
            idf=-1,
            linear_df=0,
            positional_list: list[tuple[str, DocumentTokenData]] = None,
            champions_list: list[tuple[str, DocumentTokenData]] = None,
    ):
        self.idf = idf
        self.linear_df = linear_df
        self.__list__ = dict[str, DocumentTokenData]() if positional_list is None else None
        self.list = positional_list
        self.champions_list = champions_list

    def __add_position__(self, doc_id, position: int, weight=1):
        """
        Adds a new position where the term appears in a specific document.
        """
        source = self.__list__.get(doc_id)
        if source is None:
            source = DocumentTokenData()
            self.__list__[doc_id] = source
            self.linear_df += 1

        source.__add_position__(position, weight)

    def __finalize__(self, number_of_documents):
        self.idf = score_weight.calculate_idf(self.linear_df, number_of_documents)
        for token_data in self.__list__.values():
            token_data.__update_tf__()

        self.list = sorted(
            self.__list__.items(),
            key=lambda x: x[0]
        )

    def __get_search_scope_list__(self):
        """
        Returns the appropriate list to use for search operations.

        This method prioritizes the champions list if it exists, as it might be more efficient
        for specific search algorithms. Otherwise, it falls back to the positional list.
        """
        return self.champions_list if self.champions_list is not None else self.list

    def get_term_frequency(self, doc_id):
        """
        Retrieves the linear term frequency (TF) for a specific document.

        This method attempts to get the `DocumentTokenData` object for the given document ID
        and returns its linear TF value, or 0 if the document doesn't contain the term.
        """
        try:
            return self[doc_id].linear_tf
        except KeyError:
            return 0

    def __getitem__(self, doc_id) -> DocumentTokenData:
        """
        Provides access to the `DocumentTokenData` object for a specific document ID.

        This method utilizes a binary search implemented in `binary_search_tuple` to
        efficiently find the document's data within the sorted list of document IDs.
        """
        return binary_search_tuple(self.__get_search_scope_list__(), doc_id)

    def __iter__(self, *args, **kwargs) -> Iterator[tuple[str, DocumentTokenData]]:
        """
         Allows iterating over the documents containing the term.

         This method delegates iteration to the appropriate list (positional or champions) based
         on the chosen search strategy.
         """
        return self.__get_search_scope_list__().__iter__()


class PositionalInvertedIndexOnMemory:
    """
    Represents an in-memory positional inverted index for efficient full-text search.

    This class stores terms (tokens) and their occurrences across documents. It maintains
    either a positional list or a champions list for each term, depending on the chosen
    implementation.
    """

    def __init__(self, tokens: list[tuple[str, Token]] = None):
        self.__tokens__ = dict[str, Token]() if tokens is None else None
        self.tokens = tokens

    def __add_token__(self, token_str, doc_id, position, weight=1):
        """
        Adds a new term occurrence to the index.
        """
        source = self.__tokens__.get(token_str)
        if source is None:
            source = Token()
            self.__tokens__[token_str] = source

        source.__add_position__(doc_id, position, weight)

    def __finalize__(self, number_of_documents):
        """
        Finalizes the in-memory index by calculating IDF and TF values for each term.
        """
        for token in self.__tokens__.values():
            token.__finalize__(number_of_documents)

        self.tokens = sorted(
            self.__tokens__.items(),
            key=lambda x: x[0],
        )

    def __iter__(self, *args, **kwargs) -> Iterator[tuple[str, Token]]:
        """
        Allows iterating over the terms (tokens) in the vocabulary.

        This method delegates iteration to the internal dictionary containing all token objects.
        """
        return self.tokens.__iter__()

    def __getitem__(self, token) -> Token:
        """
        Provides access to a specific term (token) by string.

        This method utilizes a binary search implemented in `binary_search_tuple`
        to efficiently find the token object within the sorted list of tokens.
        """
        return binary_search_tuple(self.tokens, token)


class IRData:
    """
    Holds all data structures required for full-text search operations.

    This class combines a positional inverted index, a list of documents, and additional metadata
    like document lengths and date range.
    """

    def __init__(
            self,
            pii: PositionalInvertedIndexOnMemory,
            docs: list[Document]
    ):
        self.pii = pii

        self.docs = dict[str, Document]()
        self.max_doc_date = datetime.min
        self.min_doc_date = datetime.max

        for d in docs:
            # Update document metadata (max/min date)
            self.max_doc_date = max(parse_date(d.date, self.max_doc_date), self.max_doc_date)
            self.min_doc_date = min(parse_date(d.date, self.min_doc_date), self.min_doc_date)

            # Store document object with minimal content (for space efficiency)
            self.docs[d.id] = Document(
                doc_id=d.id,
                title=d.title,
                url=d.url,
                content="",
                date=d.date,
                tags=None
            )

        # Precompute document lengths for scoring
        doc_lengths = {}
        for (_, token) in pii.tokens:
            for (doc_id, doc_data) in token.list:
                tf_idf = doc_data.tf * token.idf
                if doc_id in doc_lengths:
                    doc_lengths[doc_id] += tf_idf ** 2
                else:
                    doc_lengths[doc_id] = tf_idf ** 2

        for doc in doc_lengths.keys():
            doc_lengths[doc] = max(1.0, math.sqrt(doc_lengths[doc]))
        self.doc_lengths = doc_lengths

    def get_document_frequency(self, token_str):
        """
         Retrieves the document frequency (DF) for a term.

         This method attempts to get the `Token` object for the given term string from the index
         and returns its linear document frequency (number of documents containing the term),
         or 0 if the term is not in the vocabulary.
         """
        try:
            return self.pii[token_str].linear_df
        except KeyError:
            return 0

    def get_term_frequency(self, token_str, doc_id):
        """
        Retrieves the linear term frequency (TF) for a specific term in a document.

        This method attempts to get the `Token` object for the given term string from the index
        and then uses it to retrieve the `DocumentTokenData` object for the specified document ID.
        If successful, it returns the TF value from `DocumentTokenData`, or 0 if the document
        doesn't contain the term.
        """
        try:
            return self.pii[token_str].get_term_frequency(doc_id)
        except KeyError:
            return 0
