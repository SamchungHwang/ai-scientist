import asyncio
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from typing import Sequence


# 에이전트 상태를 나타내는 클래스
class AgentState:
    messages: Sequence[HumanMessage]
    next: str


# 비동기 에이전트 노드 정의
async def agent_node_async(state, agent, name):
    try:
        # 비동기 에이전트 호출
        result = await agent.invoke_async(state)

        # 메시지 처리
        messages = result["messages"]
        content = messages[-1].content if messages else "No valid content available"

        # HumanMessage 반환
        return {"messages": [HumanMessage(content=content, name=name)]}
    except Exception as e:
        # 에러 처리
        print(f"Error during async agent invocation: {e}")
        return {
            "messages": [
                HumanMessage(
                    content="Error occurred during agent invocation", name=name
                )
            ]
        }


# 비동기 메인 함수 설정
async def main():
    # 상태 초기화
    state = {"messages": [HumanMessage(content="Fetch data and process it")]}

    # LLM 및 에이전트 설정 (OpenAI GPT-4 예시)
    llm = ChatOpenAI(model="gpt-4")
    research_agent = create_react_agent(llm, tools=[])

    # 비동기 에이전트 노드 호출
    result = await agent_node_async(state, research_agent, "Researcher")
    print(result)

    # StateGraph 설정 및 시작 노드 추가
    workflow = StateGraph(AgentState)
    workflow.add_node(
        "Researcher",
        lambda s: asyncio.run(agent_node_async(s, research_agent, "Researcher")),
    )

    # 그래프 시작 지점 설정
    workflow.add_edge(START, "Researcher")
    workflow.add_edge("Researcher", END)

    # 그래프 컴파일 및 실행
    graph = workflow.compile()

    # 그래프 실행
    for s in graph.stream(state, {"recursion_limit": 10}):
        print(s)


# 비동기 메인 함수 실행
asyncio.run(main())
