## MultiAgentic RAG

This repository showcases the implementation of a **Multi-Agent Research RAG (Retriever-Augmented Generation) Tool** built with **LangGraph**. This project leverages the capabilities of agent-based frameworks to handle complex queries by breaking them down into manageable steps, dynamically utilizing tools, and ensuring response accuracy through error correction and hallucination checks.

## Getting Started

To get started with this project, follow these steps:

First, clone the repository to your local machine:

```bash
git clone https://github.com/nicoladisabato/MultiAgenticRAG.git
cd MultiAgenticRAG
```

```bash
pip install -r requirements.txt
```

Then open the config.yml file located in the root directory of the project. Set the value of load_documents to **true** to ensure the necessary documents are loaded into the vector database:

Then run:

```bash
python3 -m retriever.retriever
```

Once the PDF has been processed and indexed, you can start the application by running the following command:

```bash
python3 app.py
```

Now ask your question based on the document: https://sustainability.google/reports/google-2024-environmental-report/

self notes :
1. source .venv/bin/activate
Every time you open a new terminal and want to work on this project, you'll need to activate the virtual environment.