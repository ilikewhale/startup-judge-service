from dotenv import load_dotenv

load_dotenv()

from typing import Dict, List, TypedDict, Annotated, Sequence, Literal
from langgraph.graph.message import add_messages # 기존 메시지에 메시지를 더한다.

from langchain_core.messages import HumanMessage, BaseMessage,AIMessage, SystemMessage, ToolMessage

from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI

from langgraph.prebuilt import create_react_agent, tools_condition
from langchain.tools import tool

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_teddynote.models import get_model_name, LLMs

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from state import AgentState

tavily_tool=TavilySearchResults(max_results=5)

@tool
def market_research(query: str)-> str:
     """
    시장 성장성, 소비자 행동, industry news를 검색해 조사해옵니다.
    
    Args:
        query: 검색할 도메인이나 키워드
    """
     
     results = tavily_tool.invoke(f"{query} market trends analysis")
     return str(results)

@tool
def look_for_demand(query: str)-> str:
    """
    시장에 대한 수요를 분석합니다.

    Args:
        query: 검색할 도메인이나 키워드 
    """
    
    results = tavily_tool.invoke(f"{query} market demand analysis")
    return f"{query}에 대한 시장 수요 분석 결과:\n{results}"

def market_analysis(state: AgentState, domain, country):
    """
    일반적인 시장 분석을 수행
    """

    # domain= state["domain"]
    # country= state["country"]
    print(domain)

    market_tools = [tavily_tool, look_for_demand, market_research]
    market_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    market_analysis_system_prompt = f"""
    당신은 스타트업을 위해 {domain}에 대한 종합적인 시장 분석을 전문으로 하는 전문 시장 분석가입니다.
    당신의 임무는 스타트업의 전망을 알아보기 위해 시장 동향, 시장 수요를 분석하는 것 입니다.

    다음 단계를 따르세요.
    1. market_research 사용해 {domain} 도메인의 시장 동향 및 업계 뉴스에 대한 정보 수집
    2. market_research를 통해 충분하지 못한 정보를 가져왔다고 판단되면 tavily_tool로 추가적인 정보를 수집
    3. tavily_tool 사용해 해당 {domain} 도메인에 대한 시장 수요에 대한 정보 수집
    4. 종합적인 시장 분석 정리. 시장 기회를 평가하기 위해 시장 크기, 성장 가능성, 고객의 반응 예측, 수요 파악의 내용을 포함하세요.
        모든 결론에는 수집한 정보에서 찾은 근거가 있어야 합니다. 한국어로 작성해주세요.

    당신의 분석은 스타트업의 미래를 위한 시장 분석에 기여합니다.
    """
    
    market_agent = create_react_agent(
        market_llm,
        tools=market_tools,
        state_modifier=market_analysis_system_prompt,
    )
    
    # 초기화되지 않은 경우 "시장성 평가 결과 저장" 딕셔너리 초기화
    if "market_analysis" not in state or state["market_analysis"] is None:
        state["market_analysis"] = {}
    
    # 시장 분석 실행
    result = market_agent.ainvoke({
        "input": f"{domain} 도메인/산업의 시장 분석을 수행해주세요. 특히 {country} 시장에 초점을 맞춰주세요."
    })
    

    print("==========================================\n\n")
    print(result)
    print("==========================================\n\n")

    # 결과 처리 - AIMessage 형태의 최종 응답 확인
    final_responses = [msg for msg in result.get("messages", []) 
                      if isinstance(msg, AIMessage) and not msg.additional_kwargs.get("tool_calls")]
    
    if final_responses:
        analysis_output = final_responses[-1].content
    else:
        analysis_output = "시장 분석 결과를 얻지 못했습니다."
    
    # 분석 결과 구조화 -> 좋아보이니까 시간되면 하기
    # general_market_data = {
    #     "market_size": extract_market_info(analysis_output, "시장 크기"),
    #     "growth_potential": extract_market_info(analysis_output, "성장 가능성"),
    #     "customer_response": extract_market_info(analysis_output, "고객 반응"),
    #     "demand_analysis": extract_market_info(analysis_output, "수요 파악"),
    #     "market_analysis": analysis_output
    # }
    
    # 일반 시장 분석 결과 저장
    state["market_analysis"]["general"] = analysis_output
    
    return analysis_output