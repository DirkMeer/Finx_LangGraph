import asyncio
import functools
import operator
import uuid
from typing import Annotated, Sequence, TypedDict

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from setup_environment import set_environment_variables
from tools.pdf import OUTPUT_DIRECTORY
from tools.web import research
from web_research_prompts import RESEARCHER_SYSTEM_PROMPT, TAVILY_AGENT_SYSTEM_PROMPT


set_environment_variables("Web_Search_Graph")

TAVILY_TOOL = TavilySearchResults(max_results=6)
LLM = ChatOpenAI(model="gpt-3.5-turbo-0125")

TAVILY_AGENT_NAME = "tavily_agent"
RESEARCH_AGENT_NAME = "search_evaluator_agent"
SAVE_FILE_NODE_NAME = "save_file"


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore
    return executor


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def agent_node(state: AgentState, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


async def async_agent_node(state: AgentState, agent, name):
    result = await agent.ainvoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


tavily_agent = create_agent(LLM, [TAVILY_TOOL], TAVILY_AGENT_SYSTEM_PROMPT)
tavily_agent_node = functools.partial(
    agent_node, agent=tavily_agent, name=TAVILY_AGENT_NAME
)


research_agent = create_agent(LLM, [research], RESEARCHER_SYSTEM_PROMPT)
research_agent_node = functools.partial(
    async_agent_node, agent=research_agent, name=RESEARCH_AGENT_NAME
)


def save_file_node(state: AgentState):
    markdown_content = str(state["messages"][-1].content)
    filename = f"{OUTPUT_DIRECTORY}/{uuid.uuid4()}.md"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(markdown_content)
    return {
        "messages": [
            HumanMessage(
                content=f"Output written successfully to {filename}",
                name=SAVE_FILE_NODE_NAME,
            )
        ]
    }


workflow = StateGraph(AgentState)
workflow.add_node(TAVILY_AGENT_NAME, tavily_agent_node)
workflow.add_node(RESEARCH_AGENT_NAME, research_agent_node)
workflow.add_node(SAVE_FILE_NODE_NAME, save_file_node)

workflow.add_edge(TAVILY_AGENT_NAME, RESEARCH_AGENT_NAME)
workflow.add_edge(RESEARCH_AGENT_NAME, SAVE_FILE_NODE_NAME)
workflow.add_edge(SAVE_FILE_NODE_NAME, END)

workflow.set_entry_point(TAVILY_AGENT_NAME)
research_graph = workflow.compile()


async def run_research_graph(input):
    async for output in research_graph.astream(input):
        for node_name, output_value in output.items():
            print("---")
            print(f"Output from node '{node_name}':")
            print(output_value)
        print("\n---\n")


test_input = {"messages": [HumanMessage(content="Jaws")]}

asyncio.run(run_research_graph(test_input))
