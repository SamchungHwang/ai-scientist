import getpass
import os

from dotenv import load_dotenv
from langchain_teddynote import logging

load_dotenv()
logging.langsmith("reseacher_charter")


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph, START


# agent chain을 생성한다.
def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


# 사용할 tools를 정의한다.
from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

tavily_tool = TavilySearchResults(max_results=5)

# Warning: This executes code locally, which can be unsafe when not sandboxed

repl = PythonREPL()


@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


# create the individual agents below and tell them how to talk to each other using LangGraph.

# define the state of the graph. This will just a list of messages, along with a key to track the most recent sender
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# Python의 내장 모듈로, 함수형 프로그래밍 도구들을 제공하고 **functools.partial**을 사용하여 함수의 일부 인자를 고정한다.
import functools

# LangChain에서 사용하는 메시지 포맷 중 하나로, AI가 생성한 메시지를 나타낸다.
from langchain_core.messages import AIMessage


# 특정 에이전트가 주어진 **상태(state)**를 바탕으로 작업을 수행하고, 그 결과를 반환한다.
# state: 에이전트가 처리할 상태 데이터입니다.
# agent: 작업을 수행할 에이전트 객체입니다.
# name: 에이전트의 이름을 지정합니다.
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # result가 ToolMessage 타입인 경우 별도의 작업 없이 넘어갑니다(pass).
    # 그렇지 않은 경우, result는 AIMessage 타입으로 변환되며, 이때 result.dict(exclude={"type", "name"})로
    # type과 name 속성을 제외한 나머지 데이터를 변환하여 새로운 AIMessage를 생성합니다.
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }


llm = ChatOpenAI(model="gpt-4o")

# Research agent and node
research_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You should provide accurate data for the chart_generator to use.",
)
# functools.partial: agent_node 함수의 일부 인자를 미리 고정한 새로운 함수를 생성합니다.
# 여기서는 agent와 name 인자를 미리 지정하고, state만 나중에 전달받도록 설정합니다
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# chart_generator
chart_agent = create_agent(
    llm,
    [python_repl],
    system_message="Any charts you display will be visible by the user.",
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

# tool node를 정의한다.
# LangGraph는 언어 모델 기반 워크플로우와 에이전트 시스템을 구성할 때 사용하는 라이브러리인데,
# ToolNode는 여러 도구를 통합하여 사용하는 작업을 처리할 수 있는 노드를 제공합니다.
from langgraph.prebuilt import ToolNode

tools = [tavily_tool, python_repl]
# **ToolNode(tools)**를 통해 tavily_tool과 python_repl을 하나의 노드로 묶어서,
# 이 노드를 사용하는 워크플로우에서 다양한 도구 기반 작업을 수행할 수 있게 합니다.
# ToolNode는 다양한 도구를 활용하여 복잡한 작업을 자동화하거나 처리할 수 있는 유연한 구조를 제공합니다.
tool_node = ToolNode(tools)

# define some of the edge logic that is needed to decide what to do based on results of the agents
# Either agent can decide to end
from typing import Literal


# 에이전트 기반 시스템에서 작업의 상태를 관리하고, 상태에 따라 다음 단계로 무엇을 할지 결정하는 라우터 함수를 정의하고 있습니다.
# 주로 **상태(state)**의 메시지를 기반으로, 작업을 계속할지, 도구를 호출할지, 아니면 작업을 종료할지를 결정하는 역할을 합니다.
# Literal: Python의 타입 힌트로, 함수가 반환할 수 있는 고정된 문자열 값들을 명시합니다.
# 이 경우 함수가 반환할 값은 "call_tool", "__end__", 또는 "continue" 중 하나로 제한됩니다.
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]

    # last_message가 도구(tool)를 호출했는지를 확인합니다. 만약 호출했다면, **"call_tool"**을 반환합니다.
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"


# put it all together and define the graph!
# StateGraph: 상태 기반 워크플로우를 구성하는 그래프입니다. 각 노드가 서로 연결되어 있으며, 조건에 따라 흐름이 결정됩니다.
# AgentState: 상태를 관리하는 딕셔너리의 구조로, 에이전트가 생성한 메시지, 보낸 사람(sender) 등을 포함하는 상태 관리 객체입니다.
#  이 상태를 기반으로 워크플로우가 진행됩니다.
workflow = StateGraph(AgentState)

# 워크플로우에 노드를 추가하는 메서드입니다. 각 노드는 특정 작업을 수행하거나 상태를 처리하는 단위입니다
# 연구 에이전트를 담당하는 노드로, research_node라는 함수 또는 작업을 처리합니다.
workflow.add_node("Researcher", research_node)
# 차트 생성 에이전트를 담당하는 노드로, chart_node라는 함수 또는 작업을 수행합니다.
workflow.add_node("chart_generator", chart_node)
# "call_tool": 도구를 호출하는 작업을 담당하는 노드로, tool_node라는 도구 호출 작업을 수행합니다.
workflow.add_node("call_tool", tool_node)

# 조건에 따라 노드 간의 연결(엣지)을 설정하는 메서드입니다. 여기서는 "Researcher" 노드에서 작업이 완료된 후 어떤 노드로 이동할지를 조건부로 결정합니다.
#  이전에 정의된 라우터 함수로, 상태를 분석하여 "continue", "call_tool", "__end__" 중 하나를 반환합니다
workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
)

# "call_tool" 노드에서의 조건부 이동입니다.
# lambda x: x["sender"]: 도구 호출 후, 원래 도구를 요청한 에이전트로 다시 돌아가는 로직입니다.
# "Researcher": 원래 연구 에이전트가 도구를 호출한 경우 **"Researcher"**로 돌아갑니다.
# "chart_generator": 차트 생성 에이전트가 도구를 호출한 경우 **"chart_generator"**로 돌아갑니다.
workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "chart_generator": "chart_generator",
    },
)

# 워크플로우의 시작점을 설정합니다. **START**에서 "Researcher" 노드로 연결됩니다.
# 즉, 워크플로우가 시작되면 먼저 연구 에이전트가 작업을 수행하게 됩니다.
workflow.add_edge(START, "Researcher")


# 모든 노드와 엣지를 설정한 후, 워크플로우 그래프를 컴파일하여 실행할 준비를 마칩니다.
# 이로써 상태 기반의 에이전트 워크플로우가 완성됩니다.
graph = workflow.compile()
# graph.stream(): LangGraph의 그래프 실행 함수입니다. 입력 데이터를 기반으로 에이전트가 어떤 작업을 수행할지를 결정하고,
# 각 단계에서 생성된 이벤트를 스트리밍 방식으로 반환합니다.
# 첫 번째 인자로 **상태(state)**를 받습니다. 여기서는 HumanMessage 객체가 포함된 메시지 리스트를 전달하여 작업을 시작합니다.
# 두 번째 인자로 옵션을 전달합니다. 여기서는 최대 150 단계까지 그래프의 작업을 재귀적으로 진행할 수 있도록 설정하는
# recursion_limit을 설정했습니다. 이 옵션은 그래프가 과도하게 재귀하는 것을 방지하는 역할을 합니다.
# messages: 주어진 메시지는 작업의 입력 데이터로 사용됩니다.
# 여기서는 사람이 보낸 메시지로
# "Fetch the Korea's GDP over the past 3 years, then draw a line graph of it. Once you code it up, finish."라는 명령을 포함하고 있습니다.
# HumanMessage: 사람이 주는 명령 또는 요청을 나타내는 메시지 타입입니다.
# LangChain에서는 에이전트와 인간 간의 상호작용을 처리하기 위해 HumanMessage 객체를 사용합니다.
events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Fetch the Korea's GDP over the past 3 years,"
                " then draw a line graph of it."
                " Once you code it up, finish."
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 20},
)

# graph.stream()이 반환하는 **이벤트(event)**를 순차적으로 처리하는 반복문입니다.
# 각 이벤트는 그래프에서 진행된 각 단계를 나타내며, 이를 통해 에이전트가 무엇을 수행하고 있는지를 확인할 수 있습니다.
# **s**는 그래프에서 발생한 각각의 이벤트를 나타냅니다. 각 이벤트는 에이전트의 작업 결과, 도구 호출, 작업 완료 여부 등을 포함할 수 있습니다.
# 초기 recursion 값을 0으로 설정
recursion_depth = 0

for s in events:
    # 재귀 깊이 1씩 증가
    recursion_depth += 1

    # 이벤트와 현재 재귀 깊이를 출력
    print(f"Recursion Depth: {recursion_depth}")
    print(s)
    print("----")

from PIL import Image
from io import BytesIO

# 그래프 이미지를 파일로 저장하는 코드
try:
    # Mermaid PNG 데이터를 메모리에 저장합니다.
    png_data = graph.get_graph(xray=True).draw_mermaid_png()

    # PNG 데이터를 파일로 저장합니다.
    image_path = "output/graph_output.png"

    # BytesIO를 사용하여 메모리의 데이터를 읽고 Pillow를 통해 이미지를 저장합니다.
    with open(image_path, "wb") as f:
        f.write(png_data)

    # 저장된 이미지 파일이 존재하는지 확인하고 시스템에서 엽니다.
    if os.path.exists(image_path):
        print(f"Graph image saved at {image_path}.")
        # 시스템의 기본 이미지 뷰어로 이미지를 엽니다 (Windows, Mac, Linux 모두 지원).
        os.system(f"open {image_path}" if os.name == "posix" else f"start {image_path}")
    else:
        print("Graph image could not be saved.")
except Exception as e:
    print(f"An error occurred: {e}")
