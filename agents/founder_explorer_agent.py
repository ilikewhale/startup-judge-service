import os
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, random_uuid
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_teddynote.tools.tavily import TavilySearch

# 그래프 상태 정의
class FounderState(TypedDict):
    company_name: Annotated[str, "기업 이름"]
    domain: Annotated[str, "기업 도메인"]
    founder_name: Annotated[str, "창업자 이름"]
    founder_role: Annotated[str, "창업자 역할"]
    profile_info: Annotated[str, "창업자 프로필 정보"]
    reputation_info: Annotated[str, "창업자 평판 정보"]
    sentiment_analysis: Annotated[str, "평판 긍정/부정 분석"]
    final_summary: Annotated[str, "최종 요약"]
    messages: Annotated[list[BaseMessage], "메시지"]
    relevance: Annotated[bool, "관련성"]
    retry_count: Annotated[int, "재시도 횟수"]  # 추가된 필드

# 1. 창업자 식별 에이전트
def founder_wedsearch(state: FounderState) -> FounderState:
    
    tavily = TavilySearch()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 재시도 횟수 증가
    current_retry = state['retry_count'] + 1

    search_query = f"{state['company_name']} {state['domain']} 스타트업 창업자 CEO 대표 설립자"
    
    search_results = tavily.search(
        query=search_query,
        topic="general",
        days=60,
        max_results=5,
        format_output=True,
    )
    
    # 검색 결과를 LLM에 전달하여 창업자 이름과 역할 추출
    extraction_prompt = f"""
    다음은 {state['company_name']} 기업에 관한 정보입니다:
    
    {search_results}
    
    위 정보를 바탕으로 이 기업의 창업자(설립자) 또는 현재 CEO의 이름과 역할을 추출해주세요.
    여러 명이 있다면, 핵심 인물 한 명만 선택해주세요.
    
    다음 형식으로 응답해주세요:
    창업자 이름: [이름]
    창업자 역할: [역할 (예: CEO, 공동창업자, CTO 등)]

    IMPORTANT: 반드시 정확한 이 기업의 창업자(설립자) 또는 현재 CEO의 이름과 역할을 추출해주세요.

    IMPORTANT:
    기업의 창업자(설립자) 또는 현재 CEO를 못찾는 경우 반드시 아래와 같이 출력하세요
    창업자 이름: "정보 없음"
    창업자 역할: "정보 없음" 
    """
    
    extraction_response = llm.invoke(extraction_prompt)
    extraction_content = extraction_response.content
    
    # 응답에서 창업자 이름과 역할 추출
    founder_name = ""
    founder_role = ""
    
    for line in extraction_content.split('\n'):
        if "창업자 이름:" in line or "이름:" in line:
            founder_name = line.split(':')[1].strip()
        elif "창업자 역할:" in line or "역할:" in line:
            founder_role = line.split(':')[1].strip()
    
    print(f"✓ 식별된 창업자: {founder_name} ({founder_role})")
    
    return FounderState(
        founder_name=founder_name,
        founder_role=founder_role,
        retry_count=current_retry  # 재시도 횟수 업데이트
    )

# 2. 창업자 정보 수집 에이전트
def profile_collector(state: FounderState) -> FounderState:
    
    tavily = TavilySearch()
    search_query = f"{state['founder_name']} {state['company_name']} {state['founder_role']} 경력 학력 성과 이력"
    
    search_results = tavily.search(
        query=search_query,
        topic="general",
        days=60,
        max_results=5,
        format_output=True,
    )
    
    # 검색 결과 포맷팅
    profile_info = f"## {state['founder_name']} ({state['founder_role']}) 프로필 정보\n\n"
    profile_info += "\n\n".join(search_results)
    
    return FounderState(profile_info=profile_info)

# 3. 평판 분석 에이전트
def reputation_analyzer(state: FounderState) -> FounderState:
    
    tavily = TavilySearch()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    search_query = f"{state['founder_name']} {state['company_name']} 평판 인터뷰 SNS 미디어 리뷰"
    
    search_results = tavily.search(
        query=search_query,
        topic="news",
        days=180,  # 더 넓은 기간 설정
        max_results=5,
        format_output=True,
    )
    
    # 검색 결과 포맷팅
    reputation_info = f"## {state['founder_name']} ({state['company_name']}) 미디어 및 SNS 평판\n\n"
    reputation_info += "\n\n".join(search_results)
    
    # 감성 분석 수행
    sentiment_prompt = f"""
    다음은 {state['company_name']}의 {state['founder_name']} {state['founder_role']}에 관한 미디어 및 SNS 정보입니다. 
    이 내용을 분석하여 긍정적/부정적 평판을 판단해주세요.
    
    정보:
    {reputation_info}
    
    다음 형식으로 분석해주세요:
    1. 감성 점수: (0-100 사이, 0이 매우 부정적, 100이 매우 긍정적)
    2. 주요 긍정적 언급:
    3. 주요 부정적 언급:
    4. 전반적인 평판 판단:
    5. 투자 관점에서의 시사점:

    
    IMPORTANT: 창업자 이름: "정보 없음", 창업자 역할: "정보 없음" 인 경우에는 반드시 아래와 같이 출력해주세요
    다음 기업에 대한 창업자 정보를 충분히 찾지 못했습니다:
    기업명: {state['company_name']}
    도메인: {state['domain']}

    """
    
    sentiment_response = llm.invoke(sentiment_prompt)
    sentiment_analysis = sentiment_response.content
    
    return FounderState(
        reputation_info=reputation_info,
        sentiment_analysis=sentiment_analysis
    )

# 4. 요약 에이전트
def summary_generator(state: FounderState) -> FounderState:
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    summary_prompt = f"""
    다음은 {state['company_name']} ({state['domain']}) 기업의 창업자/대표 {state['founder_name']} ({state['founder_role']})에 관한 정보입니다. 
    이 정보를 바탕으로 AI 스타트업 투자 가능성 평가를 위한 요약 보고서를 작성해주세요.
    
    ## 기본 프로필 정보
    {state['profile_info']}
    
    ## 평판 정보
    {state['reputation_info']}
    
    ## 감성 분석 결과
    {state['sentiment_analysis']}
    
    다음 형식으로 요약해주세요:
    1. 창업자 기본 정보 (이름, 역할, 기업명)
    2. 창업자 이력 요약 (학력, 경력, 성과, 현재 직책)
    3. 평판 분석 요약 (미디어/SNS에서의 이미지)
    4. 강점 및 약점
    5. 투자 관점에서의 시사점 (창업자 역량이 기업 성장에 미치는 영향)

    IMPORTANT:
    기업의 창업자(설립자) 또는 현재 CEO를 못찾는 경우 반드시 아래와 같이 출력하세요
    "창업자의 정보를 충분히 찾지 못했습니다."
    """
    
    summary_response = llm.invoke(summary_prompt)
    final_summary = summary_response.content
    
    # 메시지 생성
    messages = [
        HumanMessage(content=f"{state['company_name']} ({state['domain']}) 기업의 {state['founder_name']} {state['founder_role']}에 대한 정보 분석 결과입니다."),
        AIMessage(content=final_summary)
    ]
    
    return FounderState(
        final_summary=final_summary,
        messages=messages
    )

def none_summary_generator(state: FounderState) -> FounderState:
    print(f"⚠️ {state['company_name']} 관련 충분한 정보를 찾지 못했습니다. 대체 요약 생성 중...")
    
    # 고정된 메시지 생성
    fallback_summary = f"""다음 기업에 대한 창업자 정보를 충분히 찾지 못했습니다:
기업명: {state['company_name']}
도메인: {state['domain']}

재시도 횟수 한계(10회)에 도달했으나 적절한 정보를 찾지 못했습니다.
직접적인 컨택이나 추가 조사를 통해 더 많은 정보를 수집하는 것을 권장합니다."""
    
    # 메시지 생성
    messages = [
        HumanMessage(content=f"{state['company_name']} ({state['domain']}) 기업 분석 시도 결과"),
        AIMessage(content=fallback_summary)
    ]
    
    return FounderState(
        final_summary=fallback_summary,
        messages=messages
    )

# 창업자 식별 관련성 체크 함수
def founder_relevance_check(state: FounderState) -> FounderState:
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 창업자 정보가 비어있거나 불충분한지 체크
    if not state['founder_name'] or state['founder_name'].strip() == "":
        return FounderState(relevance=False)
    
    # 식별된 창업자와 기업의 관련성 체크
    relevance_prompt = f"""
    질문: {state['founder_name']}이(가) {state['company_name']} ({state['domain']}) 기업의 창업자/CEO/주요 임원인가요?
    
    창업자 이름: {state['founder_name']}
    창업자 역할: {state['founder_role']}
    기업명: {state['company_name']}
    
    이 정보가 일치하는지 판단해주세요. "yes" 또는 "no"로만 대답해주세요.
    """
    
    relevance_response = llm.invoke(relevance_prompt)
    is_relevant = "yes" in relevance_response.content.lower()
    
    return FounderState(relevance=is_relevant)

# 프로필 정보 관련성 체크 함수
def profile_relevance_check(state: FounderState) -> FounderState:
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 프로필 정보와 창업자 이름의 관련성 체크
    relevance_prompt = f"""
    다음은 {state['company_name']} 기업의 {state['founder_name']} ({state['founder_role']})에 관한 검색 결과입니다:
    
    {state['profile_info']}
    
    이 정보가 실제로 해당 인물에 관한 것인지 판단해주세요.
    관련이 있다면 "yes", 없다면 "no"로만 대답해주세요.
    """
    
    relevance_response = llm.invoke(relevance_prompt)
    is_relevant = "yes" in relevance_response.content.lower()
    
    return FounderState(relevance=is_relevant)

# 조건부 라우팅 함수
def is_relevant(state: FounderState) -> str:
    if state["retry_count"] >= 8:
        print("🛑 최대 재시도 횟수(8회) 초과! 대체 요약으로 진행")
        return "none"
    elif state["relevance"]:
        return "yes"
    else:
        return "no"

def makeWorkflow():
    # 워크플로우 그래프 정의
    workflow = StateGraph(FounderState)

    # 노드 추가
    workflow.add_node("founder_wedsearch", founder_wedsearch)
    workflow.add_node("founder_relevance_check", founder_relevance_check)
    workflow.add_node("profile_collector", profile_collector)
    workflow.add_node("profile_relevance_check", profile_relevance_check)
    workflow.add_node("reputation_analyzer", reputation_analyzer)
    workflow.add_node("summary_generator", summary_generator)
    workflow.add_node("none_summary_generator", none_summary_generator)

    # 엣지 추가
    workflow.add_edge(START, "founder_identifier")
    workflow.add_edge("founder_identifier", "founder_relevance_check")
    workflow.add_conditional_edges(
        "founder_relevance_check",
        is_relevant,
        {
            "yes": "profile_collector",
            "no": "founder_identifier",  # 재시도 카운터는 founder_identifier 내부에서 관리
            "none": "none_summary_generator"  # 재시도 횟수 초과 시 대체 요약
        }
    )

    workflow.add_edge("profile_collector", "profile_relevance_check")
    workflow.add_conditional_edges(
        "profile_relevance_check",
        is_relevant,
        {
            "yes": "reputation_analyzer",
            "no": "founder_identifier",  # 프로필 정보 관련성 없으면 다시 창업자 식별부터
            "none": "none_summary_generator"  # 재시도 횟수 초과 시 대체 요약
        }
    )
    workflow.add_edge("reputation_analyzer", "summary_generator")
    workflow.add_edge("summary_generator", END)
    workflow.add_edge("none_summary_generator", END)

    # 그래프 컴파일
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# 실행 함수
async def analyze_startup_founder(company_name: str, domain: str):

    app = makeWorkflow()
    config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})
    
    inputs = FounderState(
        company_name=company_name,
        domain=domain,
        founder_name="",
        founder_role="",
        profile_info="",
        reputation_info="",
        sentiment_analysis="",
        final_summary="",
        messages=[],
        relevance=False,
        retry_count=0  # 초기 retry_count 설정
    )

    final_result = await app.ainvoke(inputs, config)
    return final_result['final_summary']
