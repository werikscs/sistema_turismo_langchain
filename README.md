# Sistema de Turismo — RAG com LangChain

A retrieval-augmented generation assistant that answers questions about travel destinations from a
curated knowledge base, instead of relying on whatever the model happens to remember.

Built as a study project during my master's program.

---

## How it works

1. **Knowledge base** — plain-text notes on two destinations, Paris and Rio de Janeiro
   (`src/base_conhecimento/`).
2. **Indexing** — documents are split into chunks, embedded with the
   `sentence-transformers/all-MiniLM-L6-v2` model via HuggingFace, and stored in a Pinecone index.
3. **Retrieval and generation** — a question retrieves the most relevant chunks, which are injected
   into a prompt template and answered by **Llama 4 Maverick** running on Groq.

The point of the exercise is that answers stay grounded in the provided documents: ask about a
destination that isn't in the knowledge base and the assistant has nothing to draw on.

## Stack

`LangChain` · `Pinecone` (vector store) · `HuggingFace` (embeddings) · `Groq` (inference) ·
`Python` · `Jupyter`

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill in the three keys:

```
HF_API_KEY=
GROQ_API_KEY=
PINECONE_API_KEY=
```

## Running

Open `src/notebook.ipynb` and run the cells in order. They're numbered by stage:

| Cells | Stage |
|---|---|
| 001–002 | environment variables and imports |
| 003 | language and embedding models |
| 004 | Pinecone setup and index population |
| 005+ | retrieval chain and queries |

The indexing step only needs to run once per index.

---

## Notes

To add a destination, drop a `.txt` file into `src/base_conhecimento/` and re-run the indexing cells.
The knowledge base is deliberately small — the goal was to understand the RAG pipeline end to end,
not to build a travel product.
