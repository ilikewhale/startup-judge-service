import json

from typing import Annotated
from state import AgentState
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(model="gpt-4", temperature=0)

def scored_agent(state: AgentState) -> Annotated[AgentState, "investment_decision"]:
    startup_name = state["startup_list"][0] if state["startup_list"] else "Unknown Startup"
    founder_info = state["founder_reputation"].get(startup_name, {})
    market_info = state["market_analysis"].get(startup_name, {})
    tech_info = state["tech_summary"].get(startup_name, "")
    competitor_info = state["competitor_info"].get(startup_name, {})
    legal_info = state["legal_risk"].get(startup_name, "")

    summary = f"""
    You are an expert early-stage startup investor using the Scorecard Valuation Method.

    You are tasked with evaluating the startup: {startup_name}

    Based on the following information, score the startup on the six categories below. 
    Each score should be from 0 to 10 (10 = excellent, 0 = poor).
    ---

    ## Startup Summary : 

    [Founder Quality]  
    Expertise, leadership, execution capability:  
    {founder_info}

    [Market Opportunity]  
    Size, growth potential, targetability:  
    {market_info}

    [Product/Technology]  
    Innovation, technical feasibility, market fit:  
    {tech_info}

    [Competitive Advantage]  
    Barriers to entry, IP, team edge:  
    {competitor_info}

    [Traction / Results]  
    Revenue, customer acquisition, partnerships:  
    (Information not available — estimate based on context)

    [Legal/Regulatory Risk]  
    Level of risk exposure, compliance potential:  
    {legal_info}

    ---

    ## Evaluation Criteria & Weights (total score = 100) :

    | Category              | Weight | Evaluation Basis                        |
    |----------------------|--------|----------------------------------------|
    | Founders             | 30%    | Execution, team strength, credibility  |
    | Market Opportunity   | 25%    | Size, trend, pain point clarity        |
    | Product/Technology   | 15%    | Innovation, differentiation            |
    | Competitive Advantage| 10%    | Barriers, positioning                  |
    | Traction             | 10%    | Revenue, users, growth evidence        |
    | Legal Risk           | 10%    | Compliance, regulatory risk            |

    ---

    ## Instructions : 

    1. Assign each category a score from 0 to 10.
    2. Multiply each score by its weight to calculate the total score.
    3. Based on the total score:
        - If score ≥ 70 → decision: "Invest"
        - If score < 70 → decision: "Hold"
    4. Please output the results in Korean.

    DO NOT return Markdown or code block syntax (like ```). 
    Return your evaluation in this JSON format:

    {{
    "score": total_score (float),
    "decision": "Invest" or "Hold",
    "reason": "1-2 sentence explanation for your decision"
    }}

    """

    response = llm([HumanMessage(content=summary)])

    try:
        result = json.loads(response.content)
    except Exception as e:
        print("⚠️ 응답 파싱 실패:", e)
        result = {
            "score": 0,
            "decision": "보류",
            "reason": "LLM 응답을 파싱할 수 없습니다."
        }

    return {"investment_decision": {startup_name: result}}
