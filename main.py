import loader
from search import search
import urllib.parse
import matplotlib.pyplot as plt

file = 'IR_data_news_12k.json'
ir = loader.create(file, cache=True, champions_list_r=40)


def show_chart(x, y, xl, yl, title):
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.grid(True)
    plt.show()


def find_most_used_tokens():
    df_token = []
    for (token, _) in ir.pii:
        df_token.append((ir.get_document_frequency(token), token))

    df_token = sorted(df_token, key=lambda x: x[0], reverse=True)
    df_token = df_token[:50]
    print(list(map(lambda x: x[1], df_token)))
    for (df, token) in df_token:
        print(f"{token} : {df}")

    df_token = df_token[:10]
    df_values, tokens = zip(*df_token)
    tokens = list(map(lambda t: t[::-1], tokens))
    show_chart(tokens, df_values, 'Term', 'DF', 'Top 10 Tokens Distribution')


def find_term_freq(term):
    df = ir.get_document_frequency(term)
    print(f"Frequency of {term} in all documents={df}")

    # print("List of DocIDs:")
    # token = ir.pii[term]
    # for (doc_id, info) in token:
    #     print(f"{doc_id}: Frequency={info.linear_tf}")


def search_for(query):
    search_result = search(ir, query)
    if len(search_result) == 0:
        print("Nothing there :(")
        return

    for (res, score) in search_result:
        print(f"ID: {res.id} | Score: {score}")
        print("Title: " + res.title)
        url = res.url.rsplit('/', 1)
        print("URL: " + url[0] + '/' + urllib.parse.quote_plus(url[1][1:]))
        print("------")

    docs, scores = zip(*search_result)
    docs = list(map(lambda t: t.id, docs))
    show_chart(docs, scores, 'DocID', 'Score', f'Top 10 Pages for {query}')


# find_most_used_tokens()
# find_term_freq('جهانی')

while True:
    print("What are you looking for?")
    q = input()
    if q == 'exit':
        break
    search_for(q)
