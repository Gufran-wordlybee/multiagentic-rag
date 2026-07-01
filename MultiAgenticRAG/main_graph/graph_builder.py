"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing & routing user queries, generating research plans to answer user questions,
conducting research, and formulating responses.
"""

from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command
from main_graph.graph_states import AgentState, Router, GradeHallucinations, InputState
from utils.prompt import ROUTER_SYSTEM_PROMPT, RESEARCH_PLAN_SYSTEM_PROMPT, MORE_INFO_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT, CHECK_HALLUCINATIONS, RESPONSE_SYSTEM_PROMPT
from subgraph.graph_builder import researcher_graph
from langchain_core.documents import Document
from typing import Any, Literal, Optional, Union
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import logging
from utils.utils import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("openai").setLevel(logging.WARNING)  
logging.getLogger("urllib3").setLevel(logging.WARNING) 

logging.getLogger("openai").propagate = False
logging.getLogger("urllib3").propagate = False
logging.getLogger("httpx").propagate = False


GPT_4o_MINI = config["llm"]["gpt_4o_mini"]
GPT_4o = config["llm"]["gpt_4o"]
TEMPERATURE = config["llm"]["temperature"]


async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    model = ChatOpenAI(model=GPT_4o, temperature=TEMPERATURE, streaming=True)
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
    ] + state.messages
    logging.info("---ANALYZE AND ROUTE QUERY---")
    logging.info(f"MESSAGES: {state.messages}")
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    return {"router": response}
    


def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state of the agent, including the router's classification.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]: The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    _type = state.router["type"]
    if _type == "environmental":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")
    


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for answering a environmental-related query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    class Plan(TypedDict):
        """Generate research plan."""

        steps: list[str]

    model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
    messages = [
        {"role": "system", "content": RESEARCH_PLAN_SYSTEM_PROMPT}
    ] + state.messages
    logging.info("---PLAN GENERATION---")
    response = cast(Plan, await model.with_structured_output(Plan).ainvoke(messages))
    return {"steps": response["steps"], "documents": "delete"}




async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information.

    This node is called when the router determines that more information is needed from the user.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
    system_prompt = MORE_INFO_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """
    result = await researcher_graph.ainvoke({"question": state.steps[0]}) #graph call directly
    docs = result["documents"]
    step = state.steps[0]
    logging.info(f"\n{len(docs)} documents retrieved in total for the step: {step}.")
    return {"documents": result["documents"], "steps": state.steps[1:]}


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node
        - Otherwise, route to the `respond` node

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["respond", "conduct_research"]: The next step to take based on whether research is complete.
    """
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to environmental.

    This node is called when the router classifies the query as a general question.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
    system_prompt = GENERAL_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )
    logging.info("---RESPONSE GENERATION---")
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}

def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"

def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


async def check_hallucinations(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Any]:
    """Analyze the user's query and checks if the response is supported by the set of facts based on the document retrieved,
    providing a binary score result.

    This function uses a language model to analyze the user's query and gives a binary score result.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
    system_prompt = CHECK_HALLUCINATIONS.format(
        documents=state.documents,
        generation=state.messages[-1]
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ] + state.messages
    logging.info("---CHECK HALLUCINATIONS---")
    response = cast(GradeHallucinations, await model.with_structured_output(GradeHallucinations).ainvoke(messages))
    
    return {"hallucination": response} 


def human_approval(
    state: AgentState,
):
    _binary_score = state.hallucination.binary_score
    if _binary_score == "1":
        return "END"
    else:
        retry_generation = interrupt(
        {
            "question": "Is this correct?",
            "llm_output": state.messages[-1]
        })

        if retry_generation == "y":
            return "respond"
        else:
            return "END"



async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a final response to the user's query based on the conducted research.

    This function formulates a comprehensive answer using the conversation history and the documents retrieved by the researcher.

    Args:
        state (AgentState): The current state of the agent, including retrieved documents and conversation history.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    logging.info("--- RESPONSE GENERATION STEP ---")
    model = ChatOpenAI(model=GPT_4o, temperature=TEMPERATURE, streaming=True)
    context = format_docs(state.documents)
    prompt = RESPONSE_SYSTEM_PROMPT.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = await model.ainvoke(messages)

    return {"messages": [response]}



checkpointer = MemorySaver()

builder = StateGraph(AgentState, input=InputState)
builder.add_node(analyze_and_route_query)
builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)
builder.add_node(create_research_plan)
builder.add_node(ask_for_more_info)
builder.add_node(respond_to_general_query)
builder.add_node(conduct_research)
builder.add_node("respond", respond)
builder.add_node(check_hallucinations)

builder.add_conditional_edges("check_hallucinations", human_approval, {"END": END, "respond": "respond"})

builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)

builder.add_edge("respond", "check_hallucinations")

graph = builder.compile(checkpointer=checkpointer)
