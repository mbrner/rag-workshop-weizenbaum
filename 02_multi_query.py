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

def do_multi_query(question, store, client, k=3, n_questions=3):
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
        docs.extend(store.search(q, top_k=k))
    # depuplicate
    return [*set(docs)]
    

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
    docs = do_multi_query(question, store, client)
    context = '\n---\n'.join([doc.text for doc in docs])
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "user", "content": PROMPT.format(question=question, context=context)}
        ]
    )
    print(Panel.fit(response.choices[0].message.content, title='Answer'))
