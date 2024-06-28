import json
from models import *


def write_cache(
        pii: PositionalInvertedIndexOnMemory,
        file: str,
):
    """
    Writes the contents of a Positional Inverted Index (PII) stored in memory to a cache file in JSON format.

    Args:
        pii (PositionalInvertedIndexOnMemory): An object representing the positional inverted index stored in memory.
        file (str): The base name of the file to which the cache will be written. The function appends ".cache" to this name.

    Cache Structure:
        The cache file is a JSON array of tokens, where each token is represented by a dictionary with the following keys:
            - "term": The string representation of the token.
            - "idf": The inverse document frequency of the token.
            - "df": The linear document frequency of the token.
            - "list": A list of dictionaries, each representing a document in which the token appears.
            Each dictionary contains:
                - "doc_id": The document ID where the token is found.
                - "tf": The term frequency of the token in the document.
                - "lf": The linear term frequency of the token in the document.
                - "list": A list of positions (integers) where the token appears in the document.

    Example Cache File Structure:
    [
        {
            "term": "example",
            "idf": 1.5,
            "df": 10,
            "list": [
                {
                    "doc_id": 1,
                    "tf": 3,
                    "lf": 1.2,
                    "list": [5, 15, 20]
                },
                ...
            ]
        },
        ...
    ]
    """

    postings_list = []
    for (token_str, token) in pii.tokens:
        positional_list = []
        for (doc_id, doc_data) in token.list:
            positional_list.append(
                {
                    "doc_id": doc_id,
                    "tf": doc_data.tf,
                    "lf": doc_data.linear_tf,
                    "list": doc_data.positions
                }
            )

        token_data = {
            "term": token_str,
            "idf": token.idf,
            "df": token.linear_df,
            "list": positional_list,
        }
        postings_list.append(token_data)

    with open(file + ".cache", 'w', encoding='utf-8') as f:
        json.dump(postings_list, f, ensure_ascii=False)


def read_from_cache(file: str):
    """
    Reads a positional inverted index from a JSON cache file.

    Args:
        file (str): The base filename for the cache file. The ".cache" extension will be appended.

    Returns:
        PositionalInvertedIndexOnMemory: The loaded positional inverted index.
            Or None if the cache file is not found.
    """
    try:
        with open(file + ".cache", 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        return None

    tokens = list[tuple[str, Token]]()
    for json_token in json_data:
        positional_list = list[tuple[str, DocumentTokenData]]()
        for json_list in json_token["list"]:
            doc_data = DocumentTokenData(
                tf=json_list["tf"],
                linear_tf=json_list["lf"],
                positions=json_list["list"],
            )
            positional_list.append(tuple((json_list["doc_id"], doc_data)))

        token = Token(
            idf=json_token["idf"],
            linear_df=json_token["df"],
            positional_list=positional_list,
        )
        tokens.append(tuple((json_token["term"], token)))

    return PositionalInvertedIndexOnMemory(tokens)


def generate_champions_list(ir: IRData, r):
    """
    Generates champions lists for tokens in an IRData object.

    This function iterates through each token in the IRData object's positional inverted index
    and creates a champions list containing the top 'r' documents for each term based on a TF-IDF score.

    Args:
        ir (IRData): The IRData object containing the inverted index and document data.
        r (int): The number of top documents to include in the champions list.
            If None, no champions lists are generated.
    """
    if r is None:
        return

    for (_, token) in ir.pii:
        def tf_idf_func(td):
            return token.idf * td[1].tf / ir.doc_lengths[td[0]]

        tf_sorted = sorted(token.list, key=tf_idf_func, reverse=True)
        tf_sorted = tf_sorted[:r]
        token.champions_list = sorted(tf_sorted, key=lambda x: x[0])
