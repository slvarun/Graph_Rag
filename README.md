# Graph-based Retrieval-Augmented Generation (RAG) System

This project is a **Graph-based Retrieval-Augmented Generation (RAG) system** that integrates Neo4j, LangChain, Hugging Face, and Streamlit. The system ingests textual documents, stores them in a graph database, and uses large language models (LLMs) and vector search to retrieve relevant information. Users can query the system in natural language, and the system provides structured and unstructured data-based answers.

## Table of Contents
- [Features](#features)
- [How the App Works](#how-the-app-works)
- [Benefits](#benefits)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Features

- **Text Upload**: Users can upload text files that are ingested into the system.
- **Graph Database**: Uploaded documents are split and stored in a Neo4j graph, which allows for structured queries and relationship-based data retrieval.
- **Natural Language Querying**: Users can input natural language questions, which are answered using both the structured graph and unstructured vector search.
- **Entity Extraction**: Automatically extracts entities (e.g., people, organizations) from queries and uses them in graph-based searches.
- **Hybrid Search**: Combines full-text search from Neo4j with vector similarity search for more accurate answers.

---

## How the App Works

1. **Uploading Documents**: Users upload a text file through the Streamlit UI. The text is split into manageable chunks, and the chunks are stored in a Neo4j graph database along with their embeddings.
   
2. **Text Processing**: The text is processed using a LangChain `LLMGraphTransformer` which converts the text into a graph-ready format. Documents are stored as nodes with relationships that enable entity-level retrieval.

3. **Querying**:
   - Users enter a natural language query.
   - The system uses a large language model to extract entities and generate a full-text search query.
   - Both **structured (graph-based)** and **unstructured (vector-based)** searches are performed to retrieve the most relevant information.

4. **Answer Generation**: The system combines the structured and unstructured results, returning a concise answer that includes both node relationships and document similarity matches.

---

## Benefits

- **Flexible Querying**: Users can ask questions in natural language, and the system intelligently processes the query to find relevant answers.
- **Structured + Unstructured Search**: Combines the power of graph relationships and document similarity for more accurate answers.
- **Entity Recognition**: Automatically extracts important entities to improve search precision.
- **Scalable**: Can ingest large amounts of data and run sophisticated searches efficiently using Neo4j and Hugging Face embeddings.
- **User-friendly Interface**: Streamlit provides an intuitive interface for uploading documents and querying the knowledge base.

---

## Installation

To install and run the app, follow these steps:

### Prerequisites

- Python 3.9 or later
- Neo4j Database (local or cloud-based)
- Hugging Face API Token
- GROQ API Token (for LLM)

### Step 1: Clone the Repository

```bash
git clone https://github.com/slvarun/Graph_Rag.git
cd graph-rag-system
```
### Step 2: Create a Virtual Environment

It's recommended to create a virtual environment for the project:

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### Step 3: Install Dependencies

Install the required Python packages:


```bash
pip install -r requirements.txt
```

###Step 4: Set Up Environment Variables

Create a .env file in the project directory and add your credentials:

URL=bolt://localhost:7687  # Replace with your Neo4j instance URL
USERNAME=neo4j
PASSWORD=your-password
HF_TOKEN=your-huggingface-api-token
GROQ_API_KEY=your-groq-api-key
LLM_MODEL_NAME=gpt-3.5-turbo  # Adjust this based on your LLM
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2  # Change to your preferred embedding model


###Step 5: Start the Neo4j Database

Make sure your Neo4j database is running. You can start Neo4j locally or connect to a cloud-based instance like Neo4j Aura.

###Step 6: Run the Application

Start the Streamlit app by running:
```bash
streamlit run app.py
```

This will launch the app on http://localhost:8501/. You can now upload documents and start querying the system.
