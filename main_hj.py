import asyncio
from util.imports import *
import pprint

load_dotenv()

from agents.agents_hj.competitor_analysis_agent import competitor_analysis
from agents.agents_hj.founder_explorer_agent import analyze_startup_founder
from agents.agents_hj.legal_risk_analysis_agent import legal_risk_analysis
from agents.agents_hj.market_analysis_agent import market_analysis
from agents.agents_hj.tech_analysis_agent import tech_analysis

state: AgentState = {
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

async def tech_analysis_agent(company):
    try:
        print(f"Starting tech analysis for {company}")
        result = await tech_analysis(company)
        # result = "오류남"
        print(f"Tech analysis completed for {company}")
        return result
    except Exception as e:
        print(f"Error in tech_analysis_agent for {company}: {e}")
        return {}

async def founder_explorer_agent(company, domain):
    try:
        print(f"Starting founder exploration for {company}")
        result = await analyze_startup_founder(company, domain)
        print(f"Founder exploration completed for {company}")
        return result
    except Exception as e:
        print(f"Error in founder_explorer_agent for {company}: {e}")
        return {}

async def market_analysis_agent(domain, country):
    try:
        print("Starting market analysis")
        # result = await market_analysis(domain, country)
        result = "오류남"
        print("Market analysis completed")
        return result
    except Exception as e:
        print(f"Error in market_analysis_agent: {e}")
        return {}

async def competitor_analysis_agent(company):
    try:
        print(f"Starting competitor analysis for {company}")
        result = await competitor_analysis(company)
        print(f"Competitor analysis completed for {company}")
        return result
    except Exception as e:
        print(f"Error in competitor_analysis_agent for {company}: {e}")
        return {}

async def legal_risk_analysis_agent(company, domain, country, tech_result):
    try:
        print(f"Starting legal risk analysis for {company}")
        result = await legal_risk_analysis(company, domain, country, tech_result)
        # result = "오류남"
        print(f"Legal risk analysis completed for {company}")
        return result
    except Exception as e:
        print(f"Error in legal_risk_analysis_agent for {company}: {e}")
        return {}

async def analyze_company(company, domain, country):
    print(f"Analyzing company {company}")
    tech_task = asyncio.create_task(tech_analysis_agent(company))
    founder_task = asyncio.create_task(founder_explorer_agent(company, domain))
    competitor_task = asyncio.create_task(competitor_analysis_agent(company))

    tech_result = await tech_task
    legal_result = await legal_risk_analysis_agent(company, domain, country, tech_result)

    founder_result = await founder_task
    competitor_result = await competitor_task

    print(f"Company {company} analysis complete.")
    return {
        "company": company,
        "tech_summary": tech_result,
        "founder_reputation": founder_result,
        "legal_risk": legal_result,
        "competitor_info": competitor_result,
    }

async def analyze_all_companies(state):
    companies = state["startup_list"]
    domain = state["domain"]
    country = state["country"]

    tech_summary = {}
    founder_reputation = {}
    legal_risk = {}
    competitor_info = {}

    print("Starting analysis for all companies")
    
    # 1. 각 기업별 분석 비동기 태스크 준비
    company_tasks = [
        analyze_company(company, domain, country)
        for company in companies
    ]

    # 2. market_analysis는 단 한 번만 실행
    market_analysis_task = asyncio.create_task(market_analysis_agent(domain, country))

    # 3. 모든 작업 병렬 실행
    company_results = await asyncio.gather(*company_tasks)
    market_result = await market_analysis_task

    for i, company in enumerate(companies):
        tech_summary[company] = company_results[i]["tech_summary"]
        founder_reputation[company] = company_results[i]["founder_reputation"]
        legal_risk[company] = company_results[i]["legal_risk"]
        competitor_info[company] = company_results[i]["competitor_info"]

    print("All company analysis completed.")
    return {
        "tech_summary": tech_summary,
        "founder_reputation": founder_reputation,
        "market_analysis": market_result,
        "legal_risk": legal_risk,
        "competitor_info": competitor_info,
    }

async def main():
    print("Starting main analysis process...")
    result = await analyze_all_companies(state)
    print("Main analysis complete.")
    pprint.pprint(result, indent=4)

def run_main():
    try:
        # 이미 이벤트 루프가 실행 중일 경우
        if asyncio.get_event_loop().is_running():
            print("Event loop already running. Using the existing loop.")
            asyncio.create_task(main())  # 기존 이벤트 루프에서 비동기 태스크 실행
        else:
            # 이벤트 루프가 실행 중이지 않으면
            asyncio.run(main())  # 새 이벤트 루프에서 실행
    except RuntimeError as e:
        print(f"Runtime error: {e}")

if __name__ == "__main__":
    run_main()
