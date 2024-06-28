# Information Retrieval PII

This repository contains an Information Retrieval (IR) system developed by me as part of an academic project at Amirkabir University of Technology (AUT). The system utilizes a Positional Inverted Index (PII) to efficiently index and retrieve information from documents. In addition to storing the frequency of words, the PII also retains their positional information within the text.

This IR system demonstrates efficient document retrieval using advanced indexing techniques and scoring models, suitable for applications requiring fast and accurate information retrieval in Persian texts.

For more detailed explanations and benchmarks, please refer to [this PDF file](./IR_project.pdf).
<br>Note that the PDF document is written in Persian (Farsi).

## Key Features
- **Loading Documents:** Documents are loaded from a JSON file containing metadata such as ID, title, content, date, and tags. The `load_documents` function handles this initial loading process.
- **Tokenization:** Textual content of each document is tokenized into separate words using the tokenizer function. (Using [Hzam](https://github.com/roshan-research/hazm) library)
- **Inverted Index Creation:** Once documents are loaded, a positional inverted index is created. This index includes each document's words along with their positional occurrences in the content, title, and tags.
- **Finalizing the Inverted Index:** After adding all tokens to the index, finalization includes computing Term Frequency (TF) and Inverse Document Frequency (IDF) weights for optimal search efficiency.
- **Query-based Inverted Index Creation:** A similar inverted index is created for user queries, enabling precise and efficient search operations.
- **Utilization in Search System:** The finalized inverted index, along with the loaded documents, is used to quickly find and rank documents containing query words.

## Caching Positional Inverted Index
Post creation, the inverted index data is temporarily stored to improve program execution speed in subsequent runs. This caching reduces runtime by approximately 89.5% (hardware-dependent) and utilizes about 76.3% less storage space compared to initial data.

The structure for storing index data is as follows:

```json
[
 {
   "term": "AFC",
   "idf": 5.8469095607744315,
   "df": 212,
   "list": [
     {
       "doc_id": "1308",
       "tf": 3.0,
       "lf": 4,
       "list": [70, 86, 122, 163]
     },
     ...
   ]
 },
 ...
]
```

## Document Preprocessing Details
Document preprocessing involves several steps:

- **Normalization:** Using the Hazm normalizer, text undergoes normalization to standardize characters, remove duplicates, and correct words.
- **Tokenization:** The normalized text is tokenized into meaningful units using the Hazm tokenizer.
- **Punctuation Removal:** All punctuation marks from tokens are removed.
- **Number Removal:** Optional removal of numeric digits.
- **Emoji Removal:** Optionally, emojis and emoticons are removed.
- **Stopwords Filtering:** Commonly used stopwords in Persian are filtered out to focus on meaningful text.
- **Lemmatization:** Tokens are lemmatized to their root forms using the Hazm lemmatizer.

## Implementation Methods for Scoring
The IR system employs multiple scoring methods:

- **Cosine Similarity Model:** <br>This model scores based on cosine similarity between document vector representations and query vector representations.
  - `cos(a, b) = (a . b) / (||a|| * ||b||)`
  - Output range: [0, 1].

- **Phrase Query:** <br>This method searches for consecutive phrases in documents. For each phrase in the query, the system checks different positions of term combinations in documents and assigns an appropriate score based on the number of occurrences.
  - `phrase_query(d, q) = 1 + max(start(i=1 to n) δ(positions(ti, d), start+i−1)) * pqw`
  - Output range: [1, 1 + pqw].

- **Date-based Scoring:** <br>For each document, a score is calculated based on the time difference from its publication date. Less time difference gives higher scores, so fresher documents are ranked higher in chronological order.
  - `date_score(d) = [timestamp(date(d)) - timestamp(min)] / [timestamp(max) - timestamp(min)] * dsw`
  - Output range: [0, dsw].
  
- **Title and Tags Weighting:** <br>Words in titles and tags are weighted higher compared to the main content of documents. This weighting emphasizes words in sections of documents that often contain important and key information, enabling users to access desired documents more accurately.<br>These weights are computed during document preprocessing and directly affect the final TF. Initially, each document has an internal weight of 1 for each term. For each term in the title, this internal weight is multiplied by `TITLE_TOKEN_WEIGHT`, and for each term in the tags, this internal weight is multiplied by `TAG_TOKEN_WEIGHT`. Finally, TF is multiplied by this internal weight after computation.
  - `tf_weighted(t, d) = tf_initial(t, d) * title_weight(t, d) * tag_weight(t, d)`

## Usage

```python
ir = loader.create(
    file='IR_data_news_12k.json',
    cache=True,
    champions_list_r=40
)

query = input("Query: ")
search_result = search(ir, query)
if len(search_result) == 0:
  print("Nothing there :(")
else:
  for (doc, score) in search_result:
    print(f"ID: {doc.id} | Score: {score}")
    print("Title: " + doc.title)
```
