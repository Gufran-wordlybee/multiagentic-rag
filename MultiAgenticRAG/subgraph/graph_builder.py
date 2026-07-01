### Build Index

from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.retrievers import EnsembleRetriever, BM25Retriever
from dotenv import load_dotenv
from subgraph.graph_states import ResearcherState, QueryState
from utils.prompt import GENERATE_QUERIES_SYSTEM_PROMPT
from langchain_core.documents import Document
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.types import Send


from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
import logging
from utils.utils import config

load_dotenv()

logger = logging.getLogger(__name__)

# Vector store configuration
VECTORSTORE_COLLECTION = config["retriever"]["collection_name"]
VECTORSTORE_DIRECTORY = config["retriever"]["directory"]
TOP_K = config["retriever"]["top_k"]
TOP_K_COMPRESSION = config["retriever"]["top_k_compression"]
ENSEMBLE_WEIGHTS = config["retriever"]["ensemble_weights"]
COHERE_RERANK_MODEL = config["retriever"]["cohere_rerank_model"]

GROQ_MODEL = config["llm"]["groq_model"]
TEMPERATURE = config["llm"]["temperature"]

def _setup_vectorstore() -> Chroma:
    """
    Set up and return the Chroma vector store instance.
    """
    embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
    )
    return Chroma(
        collection_name=VECTORSTORE_COLLECTION,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIRECTORY
    )



def _load_documents(vectorstore: Chroma) -> list[Document]:
    """
    Load documents and metadata from the vector store and return them as Langchain Document objects.

    Args:
        vectorstore (Chroma): The vector store instance.

    Returns:
        list[Document]: A list of Document objects containing the content and metadata.
    """
    all_data = vectorstore.get(include=["documents", "metadatas"])
    documents: list[Document] = []

    for content, meta in zip(all_data["documents"], all_data["metadatas"]):
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise ValueError(f"Expected metadata to be a dict, but got {type(meta)}")

        documents.append(Document(page_content=content, metadata=meta))

    return documents




def _build_retrievers(documents: list[Document], vectorstore: Chroma) -> ContextualCompressionRetriever:
    """
    Build and return a compression retriever that includes
    an ensemble retriever and Cohere-based contextual compression.

    Args:
        documents (list[Document]): List of Document objects.
        vectorstore (Chroma): The vector store to use for building retrievers.

    Returns:
        ContextualCompressionRetriever: A compression retriever that can be used to fetch and re-rank documents.
    """
    # Create base retrievers
    retriever_bm25 = BM25Retriever.from_documents(documents, search_kwargs={"k": TOP_K})
    retriever_vanilla = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K})

    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vanilla, retriever_mmr, retriever_bm25],
        weights=ENSEMBLE_WEIGHTS,
    )

    # Set up Cohere re-ranking
    compressor = CohereRerank(top_n=TOP_K_COMPRESSION, model=COHERE_RERANK_MODEL)

    # Build compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    return compression_retriever


vectorstore = _setup_vectorstore()
documents = _load_documents(vectorstore)

# Build the compression retriever (with Cohere inside)
compression_retriever = _build_retrievers(documents, vectorstore)


async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate search queries based on the question (a step in the research plan).

    This function uses a language model to generate diverse search queries to help answer the question.

    Args:
        state (ResearcherState): The current state of the researcher, including the user's question.
        config (RunnableConfig): Configuration with the model used to generate queries.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing the list of generated search queries.
    """

    class Response(TypedDict):
        queries: list[str]

    logger.info("---GENERATE QUERIES---")
    # model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
    model = ChatGroq(model=GROQ_MODEL, temperature=TEMPERATURE, streaming=True)
    messages = [
        {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
        {"role": "human", "content": state.question},
    ]
    response = cast(Response, await model.with_structured_output(Response).ainvoke(messages))
    queries = response["queries"]
    queries.append(state.question)
    logger.info(f"Queries: {queries}")
    return {"queries": response["queries"]}


async def retrieve_and_rerank_documents(
    state: QueryState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on a given query.

    This function uses a retriever to fetch relevant documents for a given query.

    Args:
        state (QueryState): The current state containing the query string.
        config (RunnableConfig): Configuration with the retriever used to fetch documents.

    Returns:
        dict[str, list[Document]]: A dictionary with a 'documents' key containing the list of retrieved documents.
    """
    logger.info("---RETRIEVING DOCUMENTS---")
    logger.info(f"Query for the retrieval process: {state.query}")

    response = compression_retriever.invoke(state.query)

    return {"documents": response}


def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """Create parallel retrieval tasks for each generated query.

    This function prepares parallel document retrieval tasks for each query in the researcher's state.

    Args:
        state (ResearcherState): The current state of the researcher, including the generated queries.

    Returns:
        Literal["retrieve_documents"]: A list of Send objects, each representing a document retrieval task.

    Behavior:
        - Creates a Send object for each query in the state.
        - Each Send object targets the "retrieve_documents" node with the corresponding query.
    """
    return [
        Send("retrieve_and_rerank_documents", QueryState(query=query)) for query in state.queries
    ]



builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_node(retrieve_and_rerank_documents)
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,  # type: ignore
    path_map=["retrieve_and_rerank_documents"],
)
builder.add_edge("retrieve_and_rerank_documents", END)
researcher_graph = builder.compile()
