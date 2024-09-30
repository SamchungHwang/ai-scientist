import getpass
import os

from dotenv import load_dotenv
from langchain_teddynote import logging
from utils import save_graph, save_graph2

load_dotenv()
logging.langsmith("hierarchical_agent_teams")


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

# ResearchTeam tools
# research team can use a search engine and url scraper to find information on the web.
from typing import Annotated, List

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

llm_model = "gpt-4o-mini"  # "gpt-4o"  # gpt-3.5-turbo"  # "gpt-4o-mini"

tavily_tool = TavilySearchResults(max_results=5)


@tool
# 주어진 URL 리스트에서 웹페이지 내용을 스크레이핑하는 역할을 합니다.
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join([f"\n{doc.page_content}\n" for doc in docs])


from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional

from langchain_experimental.utilities import PythonREPL
from typing_extensions import TypedDict

_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)


# LangChain의 tool 데코레이터: 이 데코레이터는 함수를 LangChain 에이전트가 쉽게 사용할 수 있도록 도구로 등록합니다
@tool
# 주어진 **주요 항목 리스트(points)**를 사용하여 **아웃라인(outline)**을 작성하고 파일에 저장합니다.
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
# 지정된 파일을 읽어와 특정 라인 범위(start~end)로 내용을 반환합니다.
def read_document(
    file_name: Annotated[str, "File path to save the document."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])


@tool
# 주어진 **텍스트 내용(content)**을 파일에 저장합니다.
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
# 주어진 문서 파일에 특정 라인에 문자열을 삽입하는 방식으로 문서를 수정합니다.
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""

    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())

    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"


# Warning: This executes code locally, which can be unsafe when not sandboxed

repl = PythonREPL()


@tool
# 사용자가 제공한 Python 코드를 실행하고, 그 결과를 반환합니다.
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"


# LangChain과 LangGraph를 사용하여 팀 관리 에이전트를 구현한 것입니다.
# 이 에이전트는 대화 내용을 분석한 후, 다음 작업을 수행할 팀 멤버를 선택하거나, 작업이 완료되었는지를 결정하는 역할을 합니다.
# LangChain: LLM(대형 언어 모델)을 기반으로 도구를 사용하거나, 특정 작업을 처리하는 워크플로우를 구성할 수 있는 프레임워크입니다.
# LangGraph: 상태 기반 워크플로우 그래프를 관리하고, 이를 통해 에이전트를 실행하거나 순차적으로 작업을 진행할 수 있습니다.
# ChatOpenAI: OpenAI의 GPT 모델을 기반으로 한 대화형 LLM을 사용합니다.
# 에이전트: 팀 내에서 각 작업자의 역할을 분배하고, 다음에 작업할 사람을 결정하는 역할을 합니다.
from typing import List, Optional
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph, START
from langchain_core.messages import HumanMessage, trim_messages

llm = ChatOpenAI(model=llm_model)
# 메시지의 길이를 제한하는 역할을 합니다. 설정된 **최대 토큰 수(max_tokens=100000)**를 넘지 않도록 메시지를 트리밍합니다.
trimmer = trim_messages(
    max_tokens=100000,
    strategy="last",
    token_counter=llm,
    include_system=True,
)


# 팀의 작업을 관리하고, 누가 다음 작업을 해야 하는지 결정하는 팀 관리 에이전트를 생성합니다.
# LLM을 기반으로 작동하며, 특정 역할을 할 멤버를 선택하거나 작업을 종료할지를 판단합니다.
# llm: OpenAI의 GPT 모델을 사용해 대화를 분석하고 추론합니다.
# system_prompt: 시스템 프롬프트로, 이 에이전트가 작업의 목적과 규칙을 이해하도록 합니다.
# members: 팀 멤버 리스트입니다. 이 리스트에서 다음 작업을 수행할 사람을 선택하게 됩니다.
# LLM은 상태를 보고 next 값을 반환해야 합니다.
# **function_def**는 next 값이 어떤 형식이어야 하는지 정의할 뿐, LLM이 추론을 통해 그 값을 생성하여 반환합니다.
from langchain.chains import LLMChain


def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [{"enum": options}],
                },
            },
            "required": ["next"],
        },
    }

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
    ).partial(options=str(options), team_members=", ".join(members))

    # LLM의 결과를 가공하여 messages와 next를 반환하는 로직
    def parse_supervisor_output(result):
        next_action = result["next"]
        # next_action을 기반으로 messages 필드를 생성
        return {
            "messages": [HumanMessage(content=f"Next action: {next_action}")],
            "next": next_action,
        }

    return (
        prompt
        | trimmer
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()  # JSON 형식으로 파싱
        | parse_supervisor_output  # 결과를 가공하여 messages와 next를 반환
    )


# Define Agent Teams
# research team will have a search agent and a web scraping "research_agent" as the two worker nodes
import functools
import operator

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent


# **연구 팀(ResearchTeam)**의 상태를 관리하고, 각 팀 멤버의 작업을 순차적으로 처리하는 에이전트를 만듭니다.
# 주요 기능은 **검색(Search)**과 웹 스크래핑(WebScraping)  작업을 수행하는 두 에이전트를 관리하고,
#  **슈퍼바이저(Supervisor)**가 작업을 제어하는 것입니다.


# ResearchTeam의 상태를 정의하는 데이터 구조입니다.
# TypedDict는 Python의 타입 힌트를 사용해 딕셔너리 형식의 데이터 구조를 정의하는 방식입니다.
class ResearchTeamState(TypedDict):
    # 팀 멤버들이 작업을 완료할 때마다 메시지가 추가됩니다. **operator.add**는 새로운 메시지를 기존 메시지 리스트에 추가하는 방식으로 처리됩니다.
    messages: Annotated[List[BaseMessage], operator.add]
    # 팀에 속한 멤버들의 리스트로, 각 멤버는 자신의 스킬셋과 다른 멤버의 스킬셋을 알고 있어야 합니다.
    team_members: List[str]
    # 다음 작업을 할 팀 멤버를 나타냅니다. 슈퍼바이저는 이 값을 갱신하여 다음 작업자를 선택합니다.
    next: str


# 대화형 언어 모델을 사용하여 팀 멤버들이 작업을 수행하고, 다음 작업을 누가 해야 할지 결정합니다.
llm = ChatOpenAI(model=llm_model)


# 특정 에이전트를 실행하고, 그 결과로 나온 메시지를 **HumanMessage**로 변환하여 반환합니다.
# 작업 결과에서 마지막 메시지의 내용을 가져와서, 이를 **HumanMessage**로 변환하여 반환합니다. 이 메시지는 대화의 연속성을 유지하는 데 사용됩니다.
# Existing agent_node function
def agent_node(state, agent, name):
    result = agent.invoke(state)
    new_state = {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)],
    }
    if "next" in result:
        new_state["next"] = result["next"]
    return new_state


updated_system_prompt = (
    "You are a supervisor tasked with managing a conversation between the "
    "following workers: Search, WebScraper. Your goal is to fulfill the user's "
    "request by coordinating the workers. Review the conversation so far and "
    "decide who should act next based on the tasks that need to be completed. "
    "Only respond with the name of the worker who should act next or FINISH if "
    "all tasks are completed. Remember to make use of all available workers to "
    "satisfy the user's request."
)
# 슈퍼바이저(Supervisor) 역할을 하는 에이전트를 생성합니다. 이 에이전트는 팀 멤버들이 순차적으로 작업을 수행하도록 관리합니다.
# 슈퍼바이저는 각 팀 멤버가 작업을 완료할 때마다 이를 기록하고, 다음 작업을 수행할 팀 멤버를 선택합니다.
# 프롬프트: 슈퍼바이저는 프롬프트를 통해 Search와 WebScraper라는 팀 멤버들 간의 작업 순서를 관리하고, 작업이 완료되면 FINISH를 반환합니다.
supervisor_chain = create_team_supervisor(
    llm,
    updated_system_prompt,
    ["Search", "WebScraper"],
)


# Define the SupervisorAgent class
class SupervisorAgent:
    def __init__(self, supervisor_chain):
        self.supervisor_chain = supervisor_chain

    def invoke(self, state):
        result = self.supervisor_chain.invoke(state)
        print("Supervisor Chain Result:", result)
        next_action = result.get("next")
        if not next_action:
            # Handle the case where 'next' is None or missing
            next_action = "FINISH"
        print(f"Supervisor decided next action: {next_action}")  # Debugging
        # Include messages along with 'next'
        return {
            "messages": [HumanMessage(content=f"Next action: {next_action}")],
            "next": next_action,
        }


# Initialize supervisor_agent as an instance of SupervisorAgent
supervisor_agent = SupervisorAgent(supervisor_chain)

# Modify the supervisor_node to use the updated supervisor_agent
supervisor_node = functools.partial(
    agent_node, agent=supervisor_agent, name="Supervisor"
)
# Prepare initial state
test_state = {
    "messages": [
        HumanMessage(
            content="Please research when Taylor Swift's next tour is and gather detailed information about it."
        )
    ],
}

# Invoke the supervisor chain directly
# supervisor_result = supervisor_chain.invoke(test_state)
# print("Supervisor Test Result:", supervisor_result)

# **tavily_tool**이라는 도구를 사용해 검색(Search) 작업을 수행하는 LLM기반 에이전트를 만듭니다
search_agent = create_react_agent(llm, tools=[tavily_tool])
# **functools.partial**을 사용하여 agent_node 함수에 검색 에이전트와 **에이전트 이름(Search)**을 전달합니다. 이를 통해 검색 노드가 생성되며, 이 노드는 검색 작업을 담당합니다.
search_node = functools.partial(agent_node, agent=search_agent, name="Search")

# *scrape_webpages**라는 웹 스크래핑 도구를 사용해 웹페이지에서 데이터를 수집하는 작업을 수행하는 llm기반 에이전트입니다.
research_agent = create_react_agent(llm, tools=[scrape_webpages])
# 웹 스크래퍼(WebScraper) 역할을 하는 연구 노드로 설정됩니다. 웹 스크래핑 작업을 담당합니다.
research_node = functools.partial(agent_node, agent=research_agent, name="WebScraper")

# LangGraph의 상태 기반 워크플로우를 정의하고, 여러 에이전트를 관리하는 **연구 그래프(Research Graph)**를 구축합니다.
# 각 에이전트는 특정 작업을 수행하며, **슈퍼바이저(Supervisor)**가 그 작업을 관리하고, 작업의 흐름을 제어합니다.
# StateGraph를 통해 각 작업의 순서를 정의하고, 이를 기반으로 작업이 진행됩니다.

# StateGraph: LangGraph에서 상태 기반 워크플로우를 관리하는 그래프 객체입니다. 이 그래프는 각 작업 노드를 관리하며, 상태에 따라 각 작업의 흐름을 제어합니다.
# ResearchTeamState: 연구 팀의 상태를 관리하는 데이터 구조(딕셔너리)입니다. 각 작업이 완료될 때마다 상태가 업데이트되며, 다음 작업이 결정됩니다.
research_graph = StateGraph(ResearchTeamState)
research_graph.add_node(
    "Search", search_node
)  # 검색 작업을 담당하는 검색 에이전트입니다.
research_graph.add_node(
    "WebScraper", research_node
)  # 웹 스크래핑 작업을 담당하는 웹 스크래퍼 에이전트입니다.
research_graph.add_node(  # 작업의 흐름을 관리하고, 다음 작업자를 결정하는 슈퍼바이저 에이전트입니다.
    "supervisor", supervisor_node
)


# 작업 흐름(엣지, edge): 각 작업이 끝나면 슈퍼바이저로 제어가 넘어갑니다
research_graph.add_edge("Search", "supervisor")
research_graph.add_edge("WebScraper", "supervisor")
# 슈퍼바이저가 각 팀 멤버의 작업을 관리하며, 작업이 끝나면 다음 작업자를 결정합니다.
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x[  # 슈퍼바이저는 **현재 상태(state)**에서 next 값을 확인하여, 다음 작업을 결정합니다.
        "next"
    ],
    {"Search": "Search", "WebScraper": "WebScraper", "FINISH": END},
)

# 워크플로우는 슈퍼바이저 노드에서 시작합니다. 그래프의 시작 노드는 **supervisor**로 설정됩니다.
research_graph.add_edge(START, "supervisor")
# 컴파일된 그래프를 **체인(chain)**으로 저장합니다. 이 체인은 실제로 작업을 실행할 때 사용됩니다.
research_chain = research_graph.compile()
save_graph2(graph=research_chain, image_path="output/research_chain.png")


# 상위 그래프와 연구 그래프의 상태 연결:
# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
# enter_chain() 함수는 연구 그래프를 실행하기 전에 메시지를 입력으로 받아 상태를 초기화하는 역할을 합니다.
# message: 연구 작업의 시작 메시지를 전달합니다. 이는 HumanMessage 객체로 변환됩니다.


def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


# **enter_chain**과 컴파일된 **그래프 체인(chain)**을 **연결(파이프라인)**하여, 그래프 실행 전에 상태를 설정합니다.
research_chain = enter_chain | research_chain

"""
initial_message = "Please research when Taylor Swift's next tour is and gather detailed information about it."
recursion_depth = 0
for s in research_chain.stream(initial_message, {"recursion_limit": 10}):
    if "__end__" not in s:
        recursion_depth += 1
        print(f"Recursion Depth: {recursion_depth}")
        print(s)
        print("---")
"""
# Document Writing Team
import operator
from pathlib import Path

# **문서 작성 팀(DocWritingTeam)**의 작업을 관리하는 LangChain 기반의 LangGraph 워크플로우입니다.
# 여러 작업자(에이전트)들이 문서 작성, 노트 작성, 차트 생성 작업을 수행하고, 슈퍼바이저 에이전트가 작업의 흐름을 관리합니다.
# 이 구조에서는 작업 상태와 현재 파일 시스템 상태를 기반으로 각 작업자가 작업을 이어갈 수 있도록 설정되어 있습니다.


class DocWritingState(TypedDict):
    # 팀 멤버들 간의 대화 및 작업 기록을 관리합니다.
    messages: Annotated[List[BaseMessage], operator.add]
    # 팀 멤버들의 스킬셋 정보를 제공하는 문자열입니다.
    team_members: str
    # 슈퍼바이저가 다음 작업자를 선택하기 위해 사용하는 필드입니다.
    next: str
    # 현재 작업 중인 디렉터리의 파일 목록을 저장합니다
    current_files: str


# 각 작업자가 작업을 시작하기 전에 현재 파일 상태를 업데이트합니다
# WORKING_DIRECTORY: 작업 파일이 저장되는 디렉터리입니다. 디렉터리 안에 작성된 파일 목록을 조회하여 상태에 반영합니다.
# 파일 목록을 업데이트: 작업 전에 디렉터리 내의 파일 목록을 확인하고, 이를 current_files 필드에 저장합니다.
# 디렉터리가 비어있다면: 작성된 파일이 없을 경우 "No files written" 메시지를 반환합니다.
def prelude(state):
    written_files = []
    if not WORKING_DIRECTORY.exists():
        WORKING_DIRECTORY.mkdir()
    try:
        written_files = [
            f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")
        ]
    except Exception:
        pass
    if not written_files:
        return {**state, "current_files": "No files written."}
    return {
        **state,
        "current_files": "\nBelow are files your team has written to the directory:\n"
        + "\n".join([f" - {f}" for f in written_files]),
    }


llm = ChatOpenAI(model=llm_model)
# 문서 작성 에이전트는 문서를 작성, 편집, 읽기 도구를 사용하여 문서 작업을 수행합니다.
doc_writer_agent = create_react_agent(
    llm, tools=[write_document, edit_document, read_document]
)
# **prelude**로 감싸져 있어, 작업을 시작하기 전에 현재 디렉터리 상태를 먼저 업데이트한 후 문서 작업을 수행합니다.
context_aware_doc_writer_agent = prelude | doc_writer_agent
doc_writing_node = functools.partial(
    agent_node, agent=context_aware_doc_writer_agent, name="DocWriter"
)

# 노트 작성 에이전트는 아웃라인 작성과 문서 읽기 도구를 사용하여 노트 작업을 수행합니다.
note_taking_agent = create_react_agent(llm, tools=[create_outline, read_document])
# 이 에이전트도 **prelude**로 감싸져, 현재 파일 상태를 업데이트한 후 작업을 수행합니다.
context_aware_note_taking_agent = prelude | note_taking_agent
note_taking_node = functools.partial(
    agent_node, agent=context_aware_note_taking_agent, name="NoteTaker"
)

# 차트 생성 에이전트는 문서 읽기와 Python REPL 도구를 사용하여 차트를 생성합니다.
chart_generating_agent = create_react_agent(llm, tools=[read_document, python_repl])
# 이 에이전트도 파일 상태 업데이트를 거친 후 작업을 수행합니다.
context_aware_chart_generating_agent = prelude | chart_generating_agent
chart_generating_node = functools.partial(
    agent_node, agent=context_aware_chart_generating_agent, name="ChartGenerator"
)

# 슈퍼바이저 에이전트는 팀의 작업 흐름을 관리하며, 각 팀 멤버가 작업을 완료할 때마다 다음 작업자를 선택하거나 작업을 완료하는 결정을 내립니다.
# **create_team_supervisor**는 팀 멤버의 작업 상태와 대화 내용을 기반으로 다음 작업자를 결정합니다.
# 여기서는 DocWriter, NoteTaker, ChartGenerator라는 세 명의 작업자 중 하나를 선택할 수 있습니다.

doc_writing_supervisor = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {team_members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["DocWriter", "NoteTaker", "ChartGenerator"],
)

"""
doc_writing_supervisor = create_team_supervisor(
    llm,
    f"You are a supervisor tasked with managing a conversation between the "
    f"following workers: {', '.join(['DocWriter', 'NoteTaker', 'ChartGenerator'])}. "
    "Given the following user request, respond with the worker to act next. "
    "Each worker will perform a task and respond with their results and status. "
    "When finished, respond with FINISH.",
    ["DocWriter", "NoteTaker", "ChartGenerator"],
)
"""
# Initialize supervisor_agent as an instance of SupervisorAgent
doc_writing_supervisor_agent = SupervisorAgent(doc_writing_supervisor)

# Modify the supervisor_node to use the updated supervisor_agent
doc_writing_supervisor_node = functools.partial(
    agent_node, agent=doc_writing_supervisor_agent, name="DocWritingSupervisor"
)


# Create the graph here:
# Note that we have unrolled the loop for the sake of this doc
authoring_graph = StateGraph(DocWritingState)
authoring_graph.add_node("DocWriter", doc_writing_node)
authoring_graph.add_node("NoteTaker", note_taking_node)
authoring_graph.add_node("ChartGenerator", chart_generating_node)
authoring_graph.add_node("DocWritingSupervisor", doc_writing_supervisor_node)

# Add the edges that always occur
authoring_graph.add_edge("DocWriter", "DocWritingSupervisor")
authoring_graph.add_edge("NoteTaker", "DocWritingSupervisor")
authoring_graph.add_edge("ChartGenerator", "DocWritingSupervisor")

# Add the edges where routing applies
authoring_graph.add_conditional_edges(
    "DocWritingSupervisor",
    lambda x: x["next"],
    {
        "DocWriter": "DocWriter",
        "NoteTaker": "NoteTaker",
        "ChartGenerator": "ChartGenerator",
        "FINISH": END,
    },
)

authoring_graph.add_edge(START, "DocWritingSupervisor")
chain = authoring_graph.compile()


# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str, members: List[str]):
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
    }
    return results


members = list(authoring_graph.nodes.keys())
# We reuse the enter/exit functions to wrap the graph
authoring_chain = authoring_graph.compile()
save_graph2(graph=authoring_chain, image_path="output/authoring_chain.png")
authoring_chain = functools.partial(enter_chain, members=members) | authoring_chain

"""
for s in authoring_chain.stream(
    "Write an outline for poem and then write the poem to disk.",
    {"recursion_limit": 10},
):
    if "__end__" not in s:
        print(s)
        print("---")
"""
# In this design, we are enforcing a top-down planning policy. We've created two graphs already, but we have to decide how to route work between the two.
# We'll create a third graph to orchestrate the previous two, and add some connectors to define how this top-level state is shared between the different graphs.

from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI

# 팀 작업을 관리하는 슈퍼바이저가 팀 간 대화를 관리할 때 사용됩니다.
llm = ChatOpenAI(model=llm_model)
# 슈퍼바이저는 팀 간의 작업 흐름을 관리하고, 다음에 작업할 팀을 결정하는 역할을 합니다.
supervisor_node = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following teams: {team_members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["ResearchTeam", "PaperWritingTeam"],
)


# 팀 간 작업 흐름을 관리하는 **그래프 구조(super_graph)**를 정의하는 예제입니다. 여기서 ResearchTeam과 PaperWritingTeam이
# 각각의 작업을 수행하고, **슈퍼바이저(supervisor)**가 그 작업을 조정하는 역할을 합니다. StateGraph는 상태 기반
# 워크플로우를 정의하고, 작업자들 간의 상호작용 및 작업 흐름을 관리하는 데 사용됩니다.
# Top-level graph state
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def get_last_message(state: State) -> str:
    return state["messages"][-1].content


# 팀 작업 후의 응답을 처리하여, 해당 작업의 마지막 메시지를 가져옵니다.
# 각 작업자 팀의 결과를 전체 그래프 상태에 반영하는 역할을 합니다.
def join_graph(response: dict):
    return {"messages": [response["messages"][-1]]}


# Define the graph.
super_graph = StateGraph(State)
# First add the nodes, which will do the work
super_graph.add_node("ResearchTeam", get_last_message | research_chain | join_graph)
super_graph.add_node(
    "PaperWritingTeam", get_last_message | authoring_chain | join_graph
)
super_graph.add_node("supervisor", supervisor_node)

# Define the graph connections, which controls how the logic
# propagates through the program
super_graph.add_edge("ResearchTeam", "supervisor")
super_graph.add_edge("PaperWritingTeam", "supervisor")
super_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "PaperWritingTeam": "PaperWritingTeam",
        "ResearchTeam": "ResearchTeam",
        "FINISH": END,
    },
)
super_graph.add_edge(START, "supervisor")
super_graph_chain = super_graph.compile()
# save_graph2(super_graph_chain, "output/super_graph")
# save_graph(super_graph, "output/super_graph")
recursion_depth = 0
for s in super_graph_chain.stream(
    {
        "messages": [
            HumanMessage(
                content="Write a brief research report on the North American sturgeon. Include a chart."
            )
        ],
    },
    {"recursion_limit": 20},
):
    if "__end__" not in s:
        recursion_depth += 1
        print(f"Recursion Depth: {recursion_depth}, llm_model={llm_model}")
        print(s)
        print("---")
