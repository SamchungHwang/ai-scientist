import getpass
import os

from dotenv import load_dotenv
from langchain_teddynote import logging

load_dotenv()
logging.langsmith("agent_supervisor")


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import initialize_agent, AgentType

tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()

from langchain_core.messages import HumanMessage

# 에이전트가 처리한 작업의 결과를 사람의 메시지처럼 포맷하여 반환하는 기능을 수행합니다. 주요 역할은 다음과 같습니다:


# 에이전트 작업 수행: 에이전트가 주어진 상태 정보를 바탕으로 작업을 처리합니다.
# 결과 메시지 반환: 에이전트의 작업 결과를 마지막 메시지로 가져와, 이를 HumanMessage로 변환하여 반환합니다.
# 발신자 정보 포함: 메시지를 생성할 때 발신자 정보를 포함시켜, 누가 해당 작업을 수행했는지를 나타냅니다.
# 반환된 메시지는 사람이 보낸 것처럼 포맷되지만, 실제로는 에이전트가 처리한 작업의 결과입니다.
def agent_node(state, agent_instance, name):
    # 상태에서 마지막 메시지의 내용을 추출합니다.
    if "messages" in state and state["messages"]:
        prompt = state["messages"][-1].content
    else:
        prompt = ""

    # 에이전트를 실행합니다.
    result = agent_instance.run(prompt)

    # 결과를 HumanMessage로 변환하여 메시지 리스트에 추가합니다.
    new_message = HumanMessage(content=result, name=name)
    state["messages"].append(new_message)

    # 상태를 반환합니다.
    return {"messages": state["messages"]}


# async def agent_node_async(state, agent, name):
#    try:
#        # 에이전트의 비동기 호출 (invoke_async 함수 사용)
#        result = await agent.invoke_async(state)

# 메시지 목록에서 마지막 메시지의 내용을 추출
#        messages = result["messages"]
#        content = messages[-1].content

# 마지막 메시지를 HumanMessage로 변환하여 반환
#       return {"messages": [HumanMessage(content=content, name=name)]}
#   except Exception as e:
# 에러가 발생한 경우, 에러 메시지를 HumanMessage로 반환
#       print(f"Error during async agent invocation: {e}")
#       return {
#           "messages": [
#               HumanMessage(
#                   content="Error occurred during agent invocation", name=name
#               )
#           ]
#       }


# Create Agent Supervisor
# It will use function calling to choose the next worker node OR finish processing.
# LangChain 프레임워크를 사용하여 **작업 흐름을 관리하는 에이전트(supervisor)**를 정의
# **supervisor(관리자)**는 대형 언어 모델(LLM)을 기반으로 동작하며, 여러 작업자(에이전트) 간의 작업을 관리하고,
# 다음 작업자가 누구인지, 혹은 작업이 끝났는지를 결정하는 역할을 수행합니다.
# MessagesPlaceholder: 사용자로부터의 입력 메시지를 받을 수 있는 자리표시자입니다. 이 자리에 대화 내용이 동적으로 채워집니다.
# BaseModel: Pydantic의 모델 클래스입니다. 이 클래스는 데이터 구조의 검증 및 타입 관리를 담당합니다.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal

# 작업에 참여하는 **작업자(에이전트)**의 목록입니다
members = ["Researcher", "Coder"]
# **관리자(supervisor)**가 사용하는 시스템 프롬프트입니다. 이 프롬프트는 작업자 간의 대화를 관리하는데 사용됩니다.
# 관리자에게 작업을 지시하는 내용으로, 작업자들 중에 다음에 작업할 사람을 선택하고, 작업이 완료되었을 때는 **"FINISH"**라고 응답해야 한다는 규칙을 설명합니다.
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# 작업자들과 "FINISH" 명령을 합쳐서, 작업을 진행하거나 종료할 수 있는 옵션 리스트를 만듭니다.
# 여기서는 ["FINISH", "Researcher", "Coder"]가 됩니다.
options = ["FINISH"] + members


# Pydantic 모델을 사용해 응답 형식을 정의합니다. next 필드는 작업자가 선택될 수 있는 옵션만을 허용합니다.
# Literal[*options]: 이 필드는 반드시 options에 있는 값들(즉, "FINISH", "Researcher", "Coder") 중 하나여야 합니다.
class routeResponse(BaseModel):
    next: Literal[*options]


# prompt: 이 프롬프트는 대화형 템플릿으로 사용되며, 두 가지 중요한 부분을 포함합니다:
# 시스템 프롬프트: 작업자 간의 대화를 관리할 수 있도록 시스템이 할 역할을 설명합니다.
# MessagesPlaceholder: 사용자와의 대화를 저장할 자리표시자입니다. 이 부분에는 실제로 사용자와 작업자들 간의 메시지가 채워집니다.
# partial(): 프롬프트에 대해 동적으로 값을 할당합니다. 여기서는 **options**와 members 값을 채워줍니다.
# options=str(options): 작업자와 "FINISH" 옵션을 템플릿에 동적으로 삽입합니다.
# members=", ".join(members): 작업자 목록을 콤마로 구분된 문자열로 변환하여 템플릿에 추가합니다.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 관리자가 상태(state)를 기반으로 작업자를 선택하는 워크플로우를 관리합니다.
# supervisor_chain: 이 변수는 프롬프트와 **대형 언어 모델(LLM)**을 연결하는 체인입니다.
# prompt: 작업자 선택을 관리하는 프롬프트입니다.
# llm.with_structured_output(routeResponse): OpenAI의 GPT-4 모델이 routeResponse 모델에 따라 구조화된 응답을 반환합니다.
# 즉, 결과가 routeResponse 형식에 맞게 반환됩니다.
# invoke(state): 주어진 상태를 바탕으로 체인을 실행하여, 다음 작업자가 누구인지 또는 작업이 끝났는지를 판단합니다.
def supervisor_agent(state):
    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    result = supervisor_chain.invoke(state)
    return {"next": result.next}


# LangChain과 LangGraph 프레임워크를 사용하여 에이전트(Agent) 간의 워크플로우를 관리하는 그래프 구조를 만든다.
# 주로 여러 에이전트가 상태를 기반으로 서로 상호작용하고, 상태에 따라 다음 작업으로 이동하는 프로세스를 정의합니다.
# 함수를 다루는 유틸리티 모듈로, 여기서는 **partial**을 사용하여 함수의 인자 일부를 고정합니다.
import functools

#: 파이썬 연산을 다루는 모듈로, 여기서는 **operator.add**를 사용해 메시지를 추가하는 방법을 정의합니다.
import operator
from typing import Sequence

# Python 타입 힌트로, 딕셔너리의 구조를 정의할 때 사용됩니다.
from typing_extensions import TypedDict

# LangChain에서 메시지를 주고받을 때 사용하는 기본 메시지 클래스입니다.
from langchain_core.messages import BaseMessage

# StateGraph, START, END 등은 상태 기반 워크플로우를 처리하는 그래프 구조를 만드는 데 사용됩니다.
# **create_react_agent**는 LLM 기반 에이전트를 만드는 함수입니다.
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent


# 각 에이전트의 상태를 관리하는 딕셔너리 구조입니다
class AgentState(TypedDict):
    # 에이전트 간 주고받는 메시지의 리스트입니다. **Annotated[Sequence[BaseMessage], operator.add]**는
    # 새로운 메시지가 기존 상태의 메시지 리스트에 덧붙여진다는 의미입니다.
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # 다음으로 어느 노드로 이동할지를 나타냅니다. 이 값은 에이전트가 작업을 완료한 후 어디로 이동해야 하는지를 결정하는 역할을 합니다.
    next: str


# **create_react_agent**를 사용해 **연구 에이전트(Researcher)**를 생성합니다. 이 에이전트는 **tavily_tool**이라는 도구를 사용하여 작업을 처리합니다.
research_agent = initialize_agent(
    tools=[tavily_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
# **research_node**는 연구 에이전트와 관련된 작업을 정의하는 함수입니다. functools.partial을 사용해 agent_node 함수의 일부 인자(agent와 name)를 미리 고정하여 에이전트에 특화된 노드를 생성합니다.
research_node = functools.partial(
    agent_node, agent_instance=research_agent, name="Researcher"
)

# **Python REPL(코드 실행 도구)**을 사용하는 **코딩 에이전트(Coder)**를 생성합니다. 이 에이전트는 임의의 Python 코드를 실행할 수 있습니다.
# 주의가 필요한 부분은, 이 코드는 임의의 코드 실행이 가능하므로 보안 위험이 있을 수 있다는 점입니다.
code_agent = initialize_agent(
    tools=[python_repl_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
# 코딩 에이전트의 노드를 생성합니다. 연구 에이전트와 마찬가지로 functools.partial을 사용해 일부 인자를 고정합니다.
code_node = functools.partial(agent_node, agent_instance=code_agent, name="Coder")

# LangGraph의 **StateGraph**를 생성하여 상태 기반 워크플로우를 관리합니다.
#  AgentState는 이 워크플로우에 참여하는 모든 에이전트의 상태를 정의하는 구조입니다.
workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_agent)

# LangGraph에서 여러 에이전트(작업자)의 작업 흐름을 관리하는 **상태 기반 워크플로우(StateGraph)**를 정의하고,
# 마지막으로 컴파일하여 실행 가능한 그래프를 생성하는 과정입니다. 각 작업자는 **관리자(supervisor)**에게 작업을 보고하고,
# 관리자는 작업이 완료되었는지 여부에 따라 다음 작업자를 선택하는 방식으로 작업이 진행됩니다

# 작업자가 작업을 완료한 후 관리자로 이동, **members = ["Researcher", "Coder"]**
for member in members:
    # 각 작업자(member)가 작업을 완료한 후 **관리자(supervisor)**에게 **보고(report)**하도록 **엣지(edge)**를 추가합니다.
    # 즉, 작업자가 작업을 완료하면 항상 관리자로 이동하는 흐름을 정의합니다.
    workflow.add_edge(member, "supervisor")

# 관리자(supervisor)가 다음 노드를 결정
# 관리자의 역할: 관리자는 작업이 완료된 후 어떤 작업자가 다음으로 작업을 수행할지 또는 작업이 끝났는지를 결정합니다.
# 관리자는 상태(state)에서 next 필드를 업데이트하여, 작업의 다음 흐름을 결정합니다. 이 필드는 어느 노드로 이동할지, 또는 작업이 끝났는지를 나타냅니다.

# **conditional_map**은 각 작업자를 자신의 이름으로 매핑하는 딕셔너리입니다. 예를 들어, {"Researcher": "Researcher", "Coder": "Coder"}와 같이 각 작업자를 해당 노드로 매핑합니다.
# 이 맵은 관리자가 다음 작업자를 선택할 때 사용됩니다.
conditional_map = {k: k for k in members}
# **"FINISH"**라는 값은 **END**로 매핑됩니다. 즉, 관리자가 작업이 끝났다고 판단하면 워크플로우가 종료됩니다.
conditional_map["FINISH"] = END
# add_conditional_edges: 조건부 엣지를 추가하는 메서드입니다. 조건에 따라 노드 간의 연결을 동적으로 결정할 수 있습니다.
# 출발점: "supervisor" 노드에서 작업을 마친 후, 관리자가 결정하는 조건에 따라 다른 노드로 이동합니다.
# lambda x: x["next"]: 관리자는 상태의 next 필드를 바탕으로 다음 작업자를 결정합니다.
# **x["next"]**는 관리자가 작업 상태에서 선택한 다음 노드 이름을 반환합니다.
# conditional_map: 위에서 정의한 작업자별 매핑을 바탕으로, 작업자 노드 또는 END로 이동합니다.
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# 워크플로우의 시작점을 설정합니다. 처음 시작할 때 START 노드에서 관리자(supervisor) 노드로 연결됩니다.
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

# LangGraph에서 생성된 **그래프(graph)**를 실행하고, 그 **이벤트 스트림(event stream)**을 처리하는 예제입니다. 이 코드는 특정 작업(여기서는
# "Hello World"를 코드로 작성하고 터미널에 출력하는 작업)을 수행하며, 작업이 종료되기 전까지 각 단계에서 발생한 이벤트를 출력합니다.
recursion_depth = 0
for s in graph.stream(
    {
        "messages": [
            HumanMessage(content="Code hello world and print it to the terminal")
            # HumanMessage(
            #     content="지난 3년간의 전 세계 CO₂ 배출량 데이터를 검색하고, 이를 바탕으로 연도별 CO₂ 배출량의 변화를 시각화하는 코드를 작성해라."
            # )
        ]
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 10},
):
    if "__end__" not in s:
        recursion_depth += 1
        print(s)
        print("----")
        print(f"Recursion Depth: {recursion_depth}")

from utils import save_graph

save_graph(graph=graph, image_path="output/agent_supervisor.png")
