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

HYDE_PROMPT = """Please write a scientific paper passage to answer the question
Question: {question}
"""

def do_hyde_search(question, store, client):
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "user", "content": HYDE_PROMPT.format(question=question)}
        ]
    )
    return store.search(response.choices[0].message.content, top_k=5)


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
    docs = do_hyde_search(question, store, client)
    context = '\n---\n'.join([doc.text for doc in docs])
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "user", "content": PROMPT.format(question=question, context=context)}
        ]
    )
    print(Panel.fit(response.choices[0].message.content, title='Answer'))
