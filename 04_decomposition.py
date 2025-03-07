from rich import print
from rich.panel import Panel
from dotenv import load_dotenv
load_dotenv()
from document_store.basic import OpenAIDocumentStore
from document_store.base import Document
from openai import OpenAI


PROMPT = """Answer the question delimited by <!>, use previously answered questions marked with <$> and context marked with delimiters <#>.

question:  <!>{question}<!>

Previously answered questions: <$> {previous_questions} <$>

context: <#>{context}<#>
"""

DECOMPOSITION_PROMPT = """You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of {n_questions} sub-problems / sub-questions that can be answers in isolation.
These sub-questions should be easier to answer than the original question and decompose the problem into smaller parts and
clarify the problem.
Generate multiple search queries related to: "{question}"
Provide these sub-questions separated by newlines. 
"""


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <question>')
        sys.exit(1)
    question = ' '.join(sys.argv[1:])
    client = OpenAI()
    store = OpenAIDocumentStore()
    store.load('store.pkl')

    n_subquestions = 3

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "user", "content": DECOMPOSITION_PROMPT.format(n_questions=3, question=question)}
        ]
    )
    questions = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()] + [question]
    previous_questions = []
    for i, sub_question in enumerate(questions):
        print(Panel.fit(sub_question, title=f'Sub-question {i}' if i < n_subquestions else 'Main Question'))
        context = store.search(sub_question, top_k=5)
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(
                        question=sub_question,
                        context='\n---\n'.join([d.text for d in context]),
                        previous_questions='\n\n'.join(previous_questions)
                    )
                }
            ]
        )
        sub_answer = response.choices[0].message.content
        print(Panel.fit(sub_answer, title='Answer'))
        previous_questions.append(f"Question: {sub_question}\n\nAnswer: {sub_answer}")
