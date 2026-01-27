# RAG CLI Agent

A **CLI-based Retrieval-Augmented Generation (RAG) system** that allows you to ingest `.txt` documents into a vector database and interactively query them using an LLM-powered retrieval pipeline.

The project is designed with **clean separation of concerns**, supports **metadata-aware ingestion**, and provides an **interactive REPL-style interface** for querying without repeatedly typing commands.

---

## Features

- 📄 **TXT-only ingestion**
- 🧠 **Vector database–backed retrieval, Chroma_DB**
- 🏷️ **Optional metadata support during ingestion**
- 💬 **Interactive `ask` mode (REPL-style)**
- ⚡ **Async-safe query handling**
- 🔁 **Reusable RAG core (CLI is a thin layer)**
- 🧩 **Extensible architecture (agents, retrievers, memory)**

---

## Project Structure

