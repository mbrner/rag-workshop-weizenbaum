from rich import print
from rich.panel import Panel
from dotenv import load_dotenv
load_dotenv()
from document_store.basic import OpenAIDocumentStore
from document_store.base import Document
from openai import OpenAI


PROMPT = """    Answer the question only from the customer query marked with delimiters <!> and context marked with delimiters <#>.

question:  <!>{question}<!>

Context:  <#>{context}<#>

"""

MULTI_QUERY_PROMPT = """Your task is to generate {n_questions} different versions of the
given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
You answer should constist only of the new questions. No further explanations!
Provide these alternative questions separated by newlines. Original question: {question}
"""

def do_rag_fusion(question, store, client, k_retrieve=10, k_keep=5, n_questions=5):
    assert k_keep <= k_retrieve
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "user", "content": MULTI_QUERY_PROMPT.format(n_questions=n_questions, question=question)}
        ]
    )
    docs = []
    questions = response.choices[0].message.content.split('\n')
    print(Panel.fit('\n'.join(questions), title='Alternative Questions'))
    for q in response.choices[0].message.content.split('\n'):
        docs.append(store.search(q, top_k=k_retrieve))

    fused_scores = {}
    for docs_q_i in docs:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs_q_i):
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc not in fused_scores:
                fused_scores[doc] = 0
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc] += 1 / (rank + 60)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_docs = [
        doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ][:k_keep]
    return reranked_docs

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <question>')
        sys.exit(1)
    question = ' '.join(sys.argv[1:])
    print(Panel.fit(question, title='Question'))
    client = OpenAI()
    store = OpenAIDocumentStore()
    store.load('store.pkl')
    docs = do_rag_fusion(question, store, client)
    context = '\n---\n'.join([doc.text for doc in docs])
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "user", "content": PROMPT.format(question=question, context=context)}
        ]
    )
    print(Panel.fit(response.choices[0].message.content, title='Answer'))
