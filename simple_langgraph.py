import operator
from typing import Annotated, TypedDict, Union

from colorama import Fore, Style
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.runnables.base import Runnable
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor

from setup_environment import set_environment_variables
from tools import generate_image, get_weather


set_environment_variables("LangGraph Basics")

LLM = ChatOpenAI(model="gpt-3.5-turbo-0125", streaming=True)
TOOLS = [get_weather, generate_image]
PROMPT = hub.pull("hwchase17/openai-functions-agent")


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


runnable_agent: Runnable = create_openai_functions_agent(LLM, TOOLS, PROMPT)


def agent_node(input: AgentState):
    agent_outcome: AgentActionMessageLog = runnable_agent.invoke(input)
    return {"agent_outcome": agent_outcome}


tool_executor = ToolExecutor(TOOLS)


def tool_executor_node(input: AgentState):
    agent_action = input["agent_outcome"]
    output = tool_executor.invoke(agent_action)
    print(f"Executed {agent_action} with output: {output}")
    return {"intermediate_steps": [(agent_action, output)]}


def continue_or_end_test(data: AgentState):
    if isinstance(data["agent_outcome"], AgentFinish):
        return "END"
    else:
        return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tool_executor", tool_executor_node)

workflow.set_entry_point("agent")

workflow.add_edge("tool_executor", "agent")

workflow.add_conditional_edges(
    "agent", continue_or_end_test, {"continue": "tool_executor", "END": END}
)

weather_app = workflow.compile()


def call_weather_app(query: str):
    inputs = {"input": query, "chat_history": []}
    output = weather_app.invoke(inputs)
    result = output.get("agent_outcome").return_values["output"]  # type: ignore
    steps = output.get("intermediate_steps")

    print(f"{Fore.BLUE}Result: {result}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Steps: {steps}{Style.RESET_ALL}")

    return result


# call_weather_app("What is the weather in New York?")

call_weather_app(
    "Give me a visual image displaying the current weather in Seoul, South Korea."
)
