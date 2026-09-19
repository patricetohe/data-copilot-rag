# Data Copilot (RAG + Text-to-SQL Agent)

A natural-language-to-SQL agent built with LangChain/LangGraph: ask a question in plain English, the agent retrieves the relevant schema context (RAG over table/column documentation), generates SQL, runs it safely against a sample warehouse, and self-corrects on errors.

## Architecture

```
User question --> Retriever (schema docs, vector store) --> Prompt w/ schema context
                                                                    |
                                                                    v
                                                        LLM generates SQL
                                                                    |
                                                                    v
                                        Sandbox execution (read-only, row-limited) --> result
                                                                    |
                                                        (on error) self-correct loop
```

- **src/data_copilot/** — retriever, SQL generation chain, execution sandbox, agent loop
- **eval/** — question/expected-SQL pairs + accuracy scoring harness
- **notebooks/** — exploration and prompt iteration
- **tests/** — unit tests

## Status

Built incrementally — see [ROADMAP.md](ROADMAP.md) for the current backlog.

## Stack

Python, LangChain, LangGraph, a local vector store (Chroma), DuckDB (sample warehouse), FastAPI, Streamlit.
