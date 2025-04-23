from typing import TypedDict, Annotated, Dict, List
import asyncio
import pprint
from dotenv import load_dotenv

from util.imports import *

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_teddynote.tools.tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# AgentState 정의
class AgentState(TypedDict):
    domain: str  # 스타트업 도메인 (예: "AI Chatbot")
    country: str  # 국가 (예: "Global" 또는 "South Korea")
    startup_list: List[str]  # 스타트업 목록
    startup_profiles: Dict  # 스타트업 프로필 정보
    tech_summary: Dict  # 기술 요약 정보
    founder_reputation: Dict  # 창업자 평판 정보
    market_analysis: Dict  # 시장 분석 정보
    legal_risk: Dict  # 법적 리스크 정보
    competitor_info: Dict  # 경쟁사 정보
    investment_decision: Dict  # 투자 결정 정보
    final_report: Dict  # 최종 보고서

# 초기 상태 정의
initial_state: AgentState = {
    "domain": "AI Chatbot",
    "country": "Global",
    "startup_list": ["SK", "Qure.ai"],
    "startup_profiles": {},
    "tech_summary": {},
    "founder_reputation": {},
    "market_analysis": {},
    "legal_risk": {},
    "competitor_info": {},
    "investment_decision": {},
    "final_report": {},
}

# ======================== 기술 분석 에이전트 ========================
async def tech_analysis_agent(state: AgentState) -> AgentState:
    """기업의 기술 요약 정보를 분석하는 에이전트"""
    tech_summary = {}
    
    for company in state["startup_list"]:
        try:
            print(f"Starting tech analysis for {company}")
            # tech_analysis 함수를 호출하여 각 기업의 기술 분석 수행
            from agents.agents_hj.tech_analysis_agent import tech_analysis
            result = await tech_analysis(company)
            print(f"Tech analysis completed for {company}")
            tech_summary[company] = result
        except Exception as e:
            print(f"Error in tech_analysis_agent for {company}: {e}")
            tech_summary[company] = f"기술 분석 중 오류 발생: {str(e)}"
    
    return {"tech_summary": tech_summary}

# ======================== 창업자 탐색 에이전트 ========================
async def founder_explorer_agent(state: AgentState) -> AgentState:
    """창업자 정보를 수집하고 평판을 분석하는 에이전트"""
    founder_reputation = {}
    domain = state["domain"]
    
    for company in state["startup_list"]:
        try:
            print(f"Starting founder exploration for {company}")
            # analyze_startup_founder 함수를 호출하여 창업자 분석 수행
            from agents.agents_hj.founder_explorer_agent import analyze_startup_founder
            result = await analyze_startup_founder(company, domain)
            print(f"Founder exploration completed for {company}")
            founder_reputation[company] = result
        except Exception as e:
            print(f"Error in founder_explorer_agent for {company}: {e}")
            founder_reputation[company] = f"창업자 분석 중 오류 발생: {str(e)}"
    
    return {"founder_reputation": founder_reputation}

# ======================== 시장 분석 에이전트 ========================
async def market_analysis_agent(state: AgentState) -> AgentState:
    """도메인과 국가에 대한 시장 분석을 수행하는 에이전트"""
    domain = state["domain"]
    country = state["country"]
    
    try:
        print("Starting market analysis")
        # market_analysis 함수를 호출하여 시장 분석 수행
        from agents.agents_hj.market_analysis_agent import market_analysis
        result = await market_analysis(state, domain, country)
        print("Market analysis completed")
        return {"market_analysis": {"general": result}}
    except Exception as e:
        print(f"Error in market_analysis_agent: {e}")
        return {"market_analysis": {"general": f"시장 분석 중 오류 발생: {str(e)}"}}

# ======================== 경쟁사 분석 에이전트 ========================
async def competitor_analysis_agent(state: AgentState) -> AgentState:
    """경쟁사 정보를 수집하고 분석하는 에이전트"""
    competitor_info = {}
    
    for company in state["startup_list"]:
        try:
            print(f"Starting competitor analysis for {company}")
            # competitor_analysis 함수를 호출하여 경쟁사 분석 수행
            from agents.agents_hj.competitor_analysis_agent import competitor_analysis
            result = await competitor_analysis(company)
            print(f"Competitor analysis completed for {company}")
            competitor_info[company] = result
        except Exception as e:
            print(f"Error in competitor_analysis_agent for {company}: {e}")
            competitor_info[company] = f"경쟁사 분석 중 오류 발생: {str(e)}"
    
    return {"competitor_info": competitor_info}

# ======================== 법적 리스크 분석 에이전트 ========================
async def legal_risk_analysis_agent(state: AgentState) -> AgentState:
    """법적 리스크를 분석하는 에이전트"""
    legal_risk = {}
    domain = state["domain"]
    country = state["country"]
    
    for company in state["startup_list"]:
        try:
            print(f"Starting legal risk analysis for {company}")
            tech_result = state["tech_summary"].get(company, "")
            # legal_risk_analysis 함수를 호출하여 법적 리스크 분석 수행
            from agents.agents_hj.legal_risk_analysis_agent import legal_risk_analysis
            result = await legal_risk_analysis(company, domain, country, tech_result)
            print(f"Legal risk analysis completed for {company}")
            legal_risk[company] = result
        except Exception as e:
            print(f"Error in legal_risk_analysis_agent for {company}: {e}")
            legal_risk[company] = f"법적 리스크 분석 중 오류 발생: {str(e)}"
    
    return {"legal_risk": legal_risk}

# ======================== 투자 결정 에이전트 ========================
async def investment_decision_agent(state: AgentState) -> AgentState:
    """수집된 정보를 바탕으로 투자 결정을 내리는 에이전트"""
    investment_decision = {}
    
    for company in state["startup_list"]:
        try:
            print(f"Starting investment decision for {company}")
            # 각 기업에 대한 분석 데이터 수집
            company_state = {
                "startup_list": [company],
                "domain": state["domain"],
                "country": state["country"],
                "tech_summary": {company: state["tech_summary"].get(company, {})},
                "founder_reputation": {company: state["founder_reputation"].get(company, {})},
                "market_analysis": {company: state["market_analysis"].get("general", {})},
                "legal_risk": {company: state["legal_risk"].get(company, {})},
                "competitor_info": {company: state["competitor_info"].get(company, {})},
            }
            
            # scored_agent 함수를 호출하여 투자 점수 계산
            from agents.agents_hj.scored_agent import scored_agent
            result = scored_agent(company_state)
            investment_decision[company] = result["investment_decision"][company]
            print(f"Investment decision completed for {company}")
        except Exception as e:
            print(f"Error in investment_decision_agent for {company}: {e}")
            investment_decision[company] = {
                "score": 0,
                "decision": "오류",
                "reason": f"투자 결정 중 오류 발생: {str(e)}"
            }
    
    return {"investment_decision": investment_decision}

# ======================== 보고서 생성 에이전트 ========================
async def report_generator_agent(state: AgentState) -> AgentState:
    """최종 투자 보고서를 생성하는 에이전트"""
    try:
        print("Generating final investment report")
        # report_generator 함수를 호출하여 보고서 생성
        from agents.agents_hj.report_generator_agent import report_generator
        report_path = report_generator(state)
        print(f"Final report generated at: {report_path}")
        return {"final_report": {"path": report_path}}
    except Exception as e:
        print(f"Error in report_generator_agent: {e}")
        return {"final_report": {"error": f"보고서 생성 중 오류 발생: {str(e)}"}}

# ======================== LangGraph 워크플로우 구성 ========================
async def run_parallel_analysis(state: AgentState):
    """병렬로 여러 분석 작업을 실행합니다"""
    # 병렬로 실행할 작업들
    tech_task = tech_analysis_agent(state)
    founder_task = founder_explorer_agent(state)
    market_task = market_analysis_agent(state)
    competitor_task = competitor_analysis_agent(state)
    
    # 모든 작업을 동시에 실행
    tech_result, founder_result, market_result, competitor_result = await asyncio.gather(
        tech_task, founder_task, market_task, competitor_task
    )
    
    # 결과를 상태에 병합
    result = {**tech_result, **founder_result, **market_result, **competitor_result}
    return result

# 수정된 워크플로우
def create_workflow():
    workflow = StateGraph(AgentState)
    
    # 병렬 분석을 위한 노드 추가
    workflow.add_node("parallel_analysis_node", run_parallel_analysis)
    workflow.add_node("legal_risk_analysis_node", legal_risk_analysis_agent)
    workflow.add_node("investment_decision_node", investment_decision_agent)
    workflow.add_node("report_generator_node", report_generator_agent)
    
    # 단순화된 워크플로우
    workflow.add_edge(START, "parallel_analysis_node")
    workflow.add_edge("parallel_analysis_node", "legal_risk_analysis_node")
    workflow.add_edge("legal_risk_analysis_node", "investment_decision_node")
    workflow.add_edge("investment_decision_node", "report_generator_node")
    workflow.add_edge("report_generator_node", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ======================== 실행 함수 ========================
async def run_analysis(state: AgentState = None):
    """AI 스타트업 투자 분석 실행"""
    if state is None:
        state = initial_state
    
    print("===== AI 스타트업 투자 분석 시작 =====")
    
    # 워크플로우 생성
    graph = create_workflow()
    
    # 설정
    config = RunnableConfig(
        recursion_limit=10,
        configurable={"thread_id": random_uuid()}
    )
    
    # 그래프 실행
    result = await graph.ainvoke(state, config)
    print("===== AI 스타트업 투자 분석 완료 =====")
    
    # 결과 요약 출력
    print("\n===== 분석 결과 요약 =====")
    for company in result["startup_list"]:
        decision = result["investment_decision"].get(company, {})
        score = decision.get("score", "N/A")
        verdict = decision.get("decision", "N/A")
        print(f"기업: {company}, 점수: {score}, 결정: {verdict}")
    
    # 보고서 위치 출력
    if "final_report" in result and "path" in result["final_report"]:
        print(f"최종 보고서 위치: {result['final_report']['path']}")
    
    return result

# 메인 실행 함수
def main():
    """메인 실행 함수"""
    # 이미 이벤트 루프가 실행 중일 경우
    try:
        if asyncio.get_event_loop().is_running():
            print("Event loop already running. Using the existing loop.")
            asyncio.create_task(run_analysis())  # 기존 이벤트 루프에서 비동기 태스크 실행
        else:
            # 이벤트 루프가 실행 중이지 않으면
            asyncio.run(run_analysis())  # 새 이벤트 루프에서 실행
    except RuntimeError as e:
        print(f"Runtime error: {e}")

if __name__ == "__main__":
    main()