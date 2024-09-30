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


def agent_node(state, agent_instance, name):
    # 상태에서 마지막 메시지의 내용을 추출합니다.
    if "messages" in state and state["messages"]:
        prompt = state["messages"][-1].content
    else:
        prompt = ""

    try:
        # 에이전트를 실행합니다.
        result = agent_instance.run(prompt)

        # 결과를 HumanMessage로 변환하여 메시지 리스트에 추가합니다.
        new_message = HumanMessage(content=result, name=name)
        state["messages"].append(new_message)
    except Exception as e:
        # 예외가 발생한 경우, 에러 메시지를 추가합니다.
        new_message = HumanMessage(content=f"An error occurred: {e}", name=name)
        state["messages"].append(new_message)

    # 상태를 반환합니다.
    return {"messages": state["messages"]}


from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import Literal

# 작업에 참여하는 **작업자(에이전트)**의 목록입니다
members = ["Researcher", "Coder"]
# **관리자(supervisor)**가 사용하는 시스템 프롬프트입니다.
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following conversation history,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# 작업자들과 "FINISH" 명령을 합쳐서, 작업을 진행하거나 종료할 수 있는 옵션 리스트를 만듭니다.
options = ["FINISH"] + members


# Pydantic 모델을 사용해 응답 형식을 정의합니다.
class RouteResponse(BaseModel):
    next: Literal[tuple(options)]


prompt_template = PromptTemplate(
    input_variables=[
        "members",
        "conversation",
        "options",
        "system_prompt",
        "format_instructions",
    ],
    template=(
        "{system_prompt}\n\n"
        "Conversation History:\n{conversation}\n\n"
        "Given the conversation above, who should act next?"
        " Or should we FINISH? Select one of: {options}\n\n"
        "{format_instructions}"
    ),
)

# LLM 초기화 (유효한 모델 이름 사용)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# 상태 기반으로 에이전트를 실행하고 다음 작업자가 누구인지를 결정하는 **"관리자 에이전트"**의 역할을 수행합니다.
# 이 에이전트는 특정 상태를 받아들여 그 상태에 따라 다음 단계를 선택하는 데 중점을 둡니다
def supervisor_agent(state):
    # **RouteResponse**라는 Pydantic 모델을 기반으로 출력 값을 파싱하는 데 사용됩니다.
    # 이 모델은 다음에 선택될 작업자를 나타내는 구조를 가지고 있을 것입니다.
    output_parser = PydanticOutputParser(pydantic_object=RouteResponse)
    # LLM(대형 언어 모델)이 결과를 어떻게 반환해야 하는지에 대한 규칙을 제공하는 역할을 합니다.
    # 이를 통해 결과가 올바른 형식으로 생성되도록 유도합니다.
    format_instructions = output_parser.get_format_instructions()
    # LLM과 프롬프트 템플릿을 사용하여 대화 흐름을 관리하고, LLM에 입력을 전달하여 출력을 생성합니다.
    supervisor_chain = LLMChain(llm=llm, prompt=prompt_template)
    # 메시지 리스트에서 대화 내용을 구성합니다.
    conversation = "\n".join(
        [f"{msg.name}: {msg.content}" for msg in state["messages"]]
    )
    # LMChain에 전달할 입력 데이터를 준비합니다. 이 데이터는 프롬프트 템플릿에 전달될 변수 값들입니다
    chain_input = {
        "members": ", ".join(members),
        "conversation": conversation,
        "options": str(options),
        "system_prompt": system_prompt,
        "format_instructions": format_instructions,
    }
    result = supervisor_chain.run(chain_input)
    # 결과를 파싱하여 next 값을 반환합니다.
    parsed_result = output_parser.parse(result)
    return {"next": parsed_result.next}


import functools
import operator
from typing import Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, START


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# 에이전트 생성: initialize_agent 사용
research_agent = initialize_agent(
    tools=[tavily_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

code_agent = initialize_agent(
    tools=[python_repl_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# 에이전트 노드 생성
research_node = functools.partial(
    agent_node, agent_instance=research_agent, name="Researcher"
)

code_node = functools.partial(agent_node, agent_instance=code_agent, name="Coder")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_agent)

for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

recursion_depth = 0
for s in graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Code hello world and print it to the terminal",  # name="User"
                # content="지난 3년간의 전 세계 CO₂ 배출량 데이터를 검색하고, 이를 바탕으로 연도별 CO₂ 배출량의 변화를 시각화하는 코드를 작성해라.",
                # name="User",
            )
        ]
    },
    {"recursion_limit": 10},
):
    if "__end__" not in s:
        recursion_depth += 1
        print(f"Recursion Depth: {recursion_depth}")
        print(s)
        print("----")

from utils import save_graph

save_graph(graph=graph, image_path="output/agent_supervisor.png")
