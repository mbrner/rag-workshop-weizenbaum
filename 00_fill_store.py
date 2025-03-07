
from dotenv import load_dotenv
load_dotenv()

import re

from doc_prep.convert import convert_all_docling
from doc_prep.chunking import recursive_text_splitter
from document_store.basic import OpenAIDocumentStore
from document_store.base import Document

def is_arxiv_id(text):
    pattern = r"^\d{4}\.\d{4,5}(v\d+)?$"
    return bool(re.match(pattern, text))


if __name__ == '__main__':
    documents = convert_all_docling('data', 'converted')
    document_chunks = []
    for doc in documents:
        tags = ['paper'] if is_arxiv_id(doc.name) else ['ai-act']
        print(f'{tags=}')
        chunks = recursive_text_splitter(doc.export_to_markdown(), )
        for chunk in chunks:
            document_chunks.append(
                Document(
                    tags=set(tags),
                    text=chunk
                )
            )
        print(len(document_chunks))

    store = OpenAIDocumentStore()
    store.add_documents(document_chunks)
    print(store.search('machine learning', top_k=5))
    store.store('store.pkl')