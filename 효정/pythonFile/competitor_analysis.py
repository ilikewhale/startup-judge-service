from langchain_teddynote.tools.tavily import TavilySearch

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

from langchain_teddynote.evaluator import GroundednessChecker
from langchain_teddynote.messages import messages_to_history

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, random_uuid
from langgraph.graph import StateGraph, START, END

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
import os

class WebSearchState(TypedDict):  
    question: Annotated[str, "Question"] # 질문
    context: Annotated[str, "Context"]    
    answer: Annotated[str, "Answer"]        # 답변  
    messages: Annotated[list[BaseMessage], "Messages"] # 메시지(누적되는 list)  
    relevance: Annotated[str, "Relevance"]  # 관련성  

def relevance_check(state: WebSearchState) -> WebSearchState:
    # LLM을 직접 사용하여 관련성 체크
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 관련성 체크 프롬프트
    prompt = f"""
    당신은 질문과 검색 결과의 관련성을 평가하는 평가자입니다.
    
    질문: {state["question"]}
    
    검색 결과: {state["context"]}
    
    위 질문과 검색 결과가 서로 관련이 있는지 판단해주세요. 
    관련성이 있다면 "yes"를, 없다면 "no"를 응답해주세요.
    오직 "yes" 또는 "no"로만 대답하세요.
    """
    
    # LLM 실행
    response = llm.invoke(prompt)
    result = response.content.strip().lower()
    
    # "yes"면 True, "no"면 False 반환
    is_relevant = result == "yes"
    return WebSearchState(relevance=is_relevant)

def is_relevant(state: WebSearchState) -> str:
    """조건부 라우팅을 위한 함수"""
    if state["relevance"]:
        return "yes"
    else:
        return "no"
    
def llm_answer(state: WebSearchState) -> WebSearchState:
    latest_question = state["question"]
    context = state["context"]  # 웹 검색 결과로 받은 경쟁사 분석 정보

    # 경쟁사 분석 결과를 요약하고 핵심 인사이트를 추출하는 프롬프트
    report_prompt = f"""
        다음은 {latest_question}에 대한 경쟁사 분석 정보입니다. 이 내용을 기반으로 요약 보고서를 작성해 주세요.

        형식:
        1. 국내외 경쟁사 리스트 정리
        2. 핵심 인사이트 정리
        3. 경쟁사 대비 전략적 시사점
        4. 투자측면에서 이점

        경쟁사 분석 정보: {context}
        """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(report_prompt)
    
    # 생성된 답변과 (유저의 질문, 답변) 메시지를 상태에 저장
    return WebSearchState(
        answer=response, 
        messages=[("user", latest_question), ("assistant", response)]
    )

def web_search(state: WebSearchState) -> WebSearchState:
    
    tavily_tool = TavilySearch()
    company_name = state["question"]  # 'question' 키에서 기업명 추출
    search_query = f"{company_name} 경쟁사 비교 분석"
    
    # 최근 한 달의 결과 5개 가져오기
    search_result = tavily_tool.search(
        query=search_query,
        topic="general",    # 'general'로 변경 (유효한 옵션)
        days=30,            # 최근 한 달 데이터로 확장
        max_results=3,      # 결과 수 증가
        format_output=True, # 결과 포맷팅
    )

    if search_result:
        # 결과 헤더 추가하여 컨텍스트 명확히 하기
        formatted_result = f"## {company_name} 경쟁사 분석 정보\n\n" + "\n\n".join(search_result)
        return WebSearchState(context=formatted_result)
    else:
        print("No search results found.")
        return WebSearchState(context=f"{company_name}에 대한 경쟁사 비교 정보를 찾을 수 없습니다.")

def makeGraph():
    # 워크 플로우   
    workflow = StateGraph(WebSearchState)

    workflow.add_node("web_search", web_search)
    workflow.add_node("relevance_check", relevance_check)
    workflow.add_node("llm_answer", llm_answer)

    workflow.add_edge(START, "web_search")
    workflow.add_edge("web_search", "relevance_check")  
    workflow.add_edge("llm_answer", END)  

    workflow.add_conditional_edges(
        "relevance_check",  
        is_relevant,
        {
            "yes": "llm_answer", 
            "no": "web_search",  
        },
    )

    workflow.set_entry_point("web_search")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

async def competitor_analysis(company : str):

    app = makeGraph()
    config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})
    inputs = WebSearchState(
        question=company,
        context="",  # context 비워두기
        answer="",   # 초기값 비워두기
        messages=[], # 빈 메시지 목록
        relevance="",  # 초기값 비워두기
        usage_metadata={}  # usage_metadata를 빈 딕셔너리로 추가
    )

    result = await app.ainvoke(inputs, config)
    return result['answer'].content