from typing import TypedDict, List, Dict

class AgentState(TypedDict):
    domain: Annotated[str, "Domain"]
    country: Annotated[str, "Country"]

    startup_list: Annotated[list[str], "Startup_list"]   # 스타트업 탐색 에이전트가 생성하는 주요 기업명 목록
    startup_profiles: Annotated[dict[str, dict], "Startup_profiles"]   # 스타트업별 정보 종합 저장소
    tech_summary: Annotated[dict[str, str], "Tech_summary"]  # 각 스타트업 기술 요약 정보
    founder_reputation: Annotated[dict[str, dict], "Founder_reputation"]  # 창업자 이력 + 평판 정보
    market_analysis: Annotated[dict[str, dict], "Market_analysis"]  # 시장성 분석 결과
    legal_risk: Annotated[dict[str, str], "Legal_risk"]  # 법적/규제 이슈 요약
    competitor_info: Annotated[dict[str, dict], "Competitor_info"]  # 경쟁사 비교 분석
    investment_decision: Annotated[dict[str, str], "Investment_decision"]  # 투자 판단 (투자 / 보류 + 사유)
    final_report: Annotated[str, "Final_report"]  # 보고서 생성 에이전트의 출력물 (PDF or Text)
