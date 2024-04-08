import functools
import operator
from typing import Annotated, Sequence, TypedDict

from colorama import Fore, Style
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from multi_agent_prompts import (
    TEAM_SUPERVISOR_SYSTEM_PROMPT,
    TRAVEL_AGENT_SYSTEM_PROMPT,
    LANGUAGE_ASSISTANT_SYSTEM_PROMPT,
    VISUALIZER_SYSTEM_PROMPT,
    DESIGNER_SYSTEM_PROMPT,
)
from setup_environment import set_environment_variables
from tools import generate_image, markdown_to_pdf_file


set_environment_variables("Multi_Agent_Team")

TRAVEL_AGENT_NAME = "travel_agent"
LANGUAGE_ASSISTANT_NAME = "language_assistant"
VISUALIZER_NAME = "visualizer"
DESIGNER_NAME = "designer"

TEAM_SUPERVISOR_NAME = "team_supervisor"
MEMBERS = [TRAVEL_AGENT_NAME, LANGUAGE_ASSISTANT_NAME, VISUALIZER_NAME]
OPTIONS = ["FINISH"] + MEMBERS

TAVILY_TOOL = TavilySearchResults()
LLM = ChatOpenAI(model="gpt-3.5-turbo-0125")


def create_agent(llm: BaseChatModel, tools: list, system_prompt: str):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore
    return agent_executor


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


def agent_node(state: AgentState, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


router_function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "next",
                "anyOf": [
                    {"enum": OPTIONS},
                ],
            }
        },
        "required": ["next"],
    },
}


team_supervisor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", TEAM_SUPERVISOR_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=", ".join(OPTIONS), members=", ".join(MEMBERS))


team_supervisor_chain = (
    team_supervisor_prompt_template
    | LLM.bind_functions(functions=[router_function_def], function_call="route")
    | JsonOutputFunctionsParser()
)


travel_agent = create_agent(LLM, [TAVILY_TOOL], TRAVEL_AGENT_SYSTEM_PROMPT)
travel_agent_node = functools.partial(
    agent_node, agent=travel_agent, name=TRAVEL_AGENT_NAME
)

language_assistant = create_agent(LLM, [TAVILY_TOOL], LANGUAGE_ASSISTANT_SYSTEM_PROMPT)
language_assistant_node = functools.partial(
    agent_node, agent=language_assistant, name=LANGUAGE_ASSISTANT_NAME
)

visualizer = create_agent(LLM, [generate_image], VISUALIZER_SYSTEM_PROMPT)
visualizer_node = functools.partial(agent_node, agent=visualizer, name=VISUALIZER_NAME)

designer = create_agent(LLM, [markdown_to_pdf_file], DESIGNER_SYSTEM_PROMPT)
designer_node = functools.partial(agent_node, agent=designer, name=DESIGNER_NAME)


workflow = StateGraph(AgentState)
workflow.add_node(TRAVEL_AGENT_NAME, travel_agent_node)
workflow.add_node(LANGUAGE_ASSISTANT_NAME, language_assistant_node)
workflow.add_node(VISUALIZER_NAME, visualizer_node)
workflow.add_node(DESIGNER_NAME, designer_node)
workflow.add_node(TEAM_SUPERVISOR_NAME, team_supervisor_chain)


for member in MEMBERS:
    workflow.add_edge(member, TEAM_SUPERVISOR_NAME)

workflow.add_edge(DESIGNER_NAME, END)

conditional_map = {name: name for name in MEMBERS}
conditional_map["FINISH"] = DESIGNER_NAME
workflow.add_conditional_edges(
    TEAM_SUPERVISOR_NAME, lambda state: state["next"], conditional_map
)

workflow.set_entry_point(TEAM_SUPERVISOR_NAME)

travel_agent_graph = workflow.compile()


for chunk in travel_agent_graph.stream(
    {"messages": [HumanMessage(content="I want to go to Paris for three days")]}
):
    if "__end__" not in chunk:
        print(chunk)
        print(f"{Fore.GREEN}#############################{Style.RESET_ALL}")
