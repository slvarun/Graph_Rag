'''
This code sets up a Graph-based Retrieval-Augmented Generation (RAG) application using Neo4j,
LangChain, and Streamlit. The application allows users to upload a text file to add knowledge
into a Neo4j graph database, run similarity searches over the stored documents, and retrieve 
structured and unstructured data based on user queries. It integrates various components like 
full-text search, large language models (LLMs), and vector search, to provide structured responses.
'''


from datetime import datetime
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
import os
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from typing import List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
import logging
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Set up logging
logging.getLogger("neo4j").setLevel(logging.ERROR)

# Constants
CHUNK_SIZE = 500
CHUNK_OVERLAP = 20

# Graph and LLM initialization
url, username, password = os.getenv("URL"), os.getenv("USERNAME"), os.getenv("PASSWORD")
graph = Neo4jGraph(url=url, username=username, password=password)
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=os.getenv("LLM_MODEL_NAME"))
embeddings_model = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL_NAME"))

# Function to add knowledge to the graph

def add_knowledge(text):
    '''
    Explanation:
        Text Splitting: The input text is split into chunks using RecursiveCharacterTextSplitter,
        ensuring that each chunk is small enough to process efficiently.
        LLMGraphTransformer: The LLM transforms the chunks into a format that can be stored in a Neo4j graph.
        Adding to Graph: The transformed documents are added to the graph as nodes with relationships.
    Parameters:
        text: The content of the uploaded text file.
        Returns: None. Adds documents to the graph database.
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]
    graph_documents = LLMGraphTransformer(llm=llm).convert_to_graph_documents(documents)
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

# Vector Index
'''
Explanation:
Vector Indexing: Creates an index for the graph that allows for hybrid searches
(combining both full-text and vector-based searches). The embeddings generated
using Hugging Face are stored in the Neo4j database, and later used for similarity-based queries.
'''
vector_index = Neo4jVector.from_existing_graph(
    embeddings_model,
    url=url,
    username=username,
    password=password,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# Create fulltext index for querying
graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

# Model for extracting entities
class Entities(BaseModel):
    names: List[str]

# Prompt for extracting entities
entity_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are extracting organization and person entities from the text."),
        ("human", "Use the given format to extract information from the following input: {question}")
    ]
)

entity_chain = entity_prompt | llm.with_structured_output(Entities)

# Function to generate fulltext query
def generate_full_text_query(input: str) -> str:
    '''
    Explanation:
        This function generates a full-text query string that can be used with Neo4j's
        full-text search API. The query uses a fuzzy search (~2 allows for slight 
        variations in word spelling).
    Parameters:
        input: A raw query string.
        Returns: A formatted full-text query string.
    '''
    words = remove_lucene_chars(input).split()
    return " AND ".join([f"{word}~2" for word in words])

# Structured retriever
def structured_retriever(question: str) -> str:
    '''
    Explanation:
        Entity Extraction: The LLM extracts entities from the input question.
    Full-text Search: 
        A query is executed to find graph nodes related to the 
        extracted entities, and relationships (e.g., MENTIONS) are used to retrieve related nodes.
    Result Formatting: 
        Outputs structured information (nodes and relationships) in a readable format.
    '''
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

# Full retriever function
def retriever(question: str) -> str:
    '''
    Explanation:
        Structured Data: The structured_retriever function retrieves data from the graph based on entities found in the question.
        Unstructured Data: A similarity search is performed over the vector index for unstructured document matching.
        Final Result: The results of both the structured and unstructured searches are combined and returned.
    '''
    structured_data = structured_retriever(question)
    unstructured_data = "\n".join([el.page_content for el in vector_index.similarity_search(question)])
    return f"Structured data:\n{structured_data}\n\nUnstructured data:\n{unstructured_data}"

# Prompt template for rephrasing the question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# Function to format chat history
def _format_chat_history(chat_history: List) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

# Runnable for search query
_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    RunnableLambda(lambda x: x["question"]),
)

# Answer template
template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Final chain setup
chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit app setup
if 'history' not in st.session_state:
    st.session_state['history'] = []

st.title("Graph RAG")

st.sidebar.header("Upload Section")
uploaded_file = st.sidebar.file_uploader("Upload a text file (max 100KB)", type=["txt"])

if uploaded_file:
    file_content = uploaded_file.read().decode("utf-8")
    with st.spinner("Processing ..."):
        add_knowledge(file_content)
    st.session_state['history'] = []

query = st.text_input("Enter your query:")
if query:
    response = chain.invoke({"question": query})
    st.session_state['history'].append((query, response))
    st.write(f"**Response**: {response}")

st.subheader("Query History")
if len(st.session_state['history']) != 0 :
    for i in range(len(st.session_state['history'])-1,-1,-1):
        st.markdown(f"**Query**: {st.session_state['history'][i][0]}")
        st.markdown(f"**Response**: {st.session_state['history'][i][1]}")
        st.markdown("---")

