# import os
# from typing import Dict
# from state import AgentState
# from datetime import datetime
# from weasyprint import HTML

# # 보고서 제목 설정
# REPORT_TITLE = "SKALA AI 스타트업 투자 보고서"
# REPORT_PATH = "reports/startup_report.pdf"

# def generate_report(state: AgentState) -> str:
#     os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

#     # 1. 기업을 점수 기준으로 정렬
#     sorted_startups = sorted(
#         state["investment_decision"].items(),
#         key=lambda x: x[1]["score"],
#         reverse=True
#     )

#     # 2. 목차 구성
#     toc_entries = "".join(
#         [f"<li>{i+1}위: {name} (점수: {info['score']})</li>" for i, (name, info) in enumerate(sorted_startups)]
#     )

#     # 3. 본문 구성
#     body_sections = ""
#     for idx, (name, decision_data) in enumerate(sorted_startups, start=1):
#         body_sections += f"""
#         <h2>{idx}. {name}</h2>
#         <p><strong>투자 점수:</strong> {decision_data['score']} / <strong>결과:</strong> {decision_data['decision']}</p>
#         <p><strong>사유:</strong> {decision_data['reason']}</p>
#         <p><strong>기술 요약:</strong> {state['tech_summary'].get(name, '정보 없음')}</p>
#         <p><strong>창업자 평판:</strong> {state['founder_reputation'].get(name, '정보 없음')}</p>
#         <p><strong>시장성 분석:</strong> {state['market_analysis'].get(name, '정보 없음')}</p>
#         <p><strong>법적/규제 리스크:</strong> {state['legal_risk'].get(name, '정보 없음')}</p>
#         <p><strong>경쟁사 비교:</strong> {state['competitor_info'].get(name, '정보 없음')}</p>
#         <hr />
#         """

#     # 4. HTML 템플릿 생성
#     now = datetime.now().strftime("%Y-%m-%d")
#     html_content = f"""
#     <html>
#     <head>
#         <meta charset='utf-8'>
#         <style>
#             body {{ font-family: sans-serif; padding: 2em; }}
#             h1 {{ color: #2c3e50; }}
#             h2 {{ color: #34495e; margin-top: 1.5em; }}
#             hr {{ margin-top: 2em; }}
#         </style>
#     </head>
#     <body>
#         <h1>{REPORT_TITLE}</h1>
#         <p><em>작성일: {now}</em></p>
#         <h2>1. 투자 기업 순위</h2>
#         <ul>{toc_entries}</ul>
#         <h2>2. 기업별 분석</h2>
#         {body_sections}
#     </body>
#     </html>
#     """

#     # 5. PDF 저장
#     HTML(string=html_content).write_pdf(REPORT_PATH)

#     return REPORT_PATH
import os
from typing import Dict
from state import AgentState
from datetime import datetime
from weasyprint import HTML

# 보고서 제목 설정
REPORT_TITLE = "SKALA AI 스타트업 투자 보고서"
REPORT_PATH = "reports/startup_report.pdf"

def generate_report(state: AgentState) -> str:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    domain_market = state.get("market_analysis", {}).get("__domain_summary__", f"{state['domain']} 시장 분석 정보 없음")

    # 1. 기업을 점수 기준으로 정렬
    sorted_startups = sorted(
        state["investment_decision"].items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    # 2. 목차 구성
    toc_entries = "".join(
        [f"<li>{i+1}. {name} (점수: {info['score']})</li>" for i, (name, info) in enumerate(sorted_startups)]
    )

    # 3. 본문 구성
    body_sections = ""
    for idx, (name, decision_data) in enumerate(sorted_startups, start=1):
        body_sections += f"""
        <h3>3-{idx}. {name}</h3>
        <p><strong>투자 점수:</strong> {decision_data['score']} / <strong>결과:</strong> {decision_data['decision']}</p>
        <p><strong>사유:</strong> {decision_data['reason']}</p>
        <p><strong>기술 요약:</strong> {state['tech_summary'].get(name, '정보 없음')}</p>
        <p><strong>창업자 평판:</strong> {state['founder_reputation'].get(name, '정보 없음')}</p>
        <p><strong>법적/규제 리스크:</strong> {state['legal_risk'].get(name, '정보 없음')}</p>
        <p><strong>경쟁사 비교:</strong> {state['competitor_info'].get(name, '정보 없음')}</p>
        <hr />
        """

    # 4. HTML 템플릿 생성
    now = datetime.now().strftime("%Y-%m-%d")
    html_content = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <style>
            body {{ font-family: sans-serif; padding: 2em; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 1.5em; }}
            h3 {{ color: #2c3e50; margin-top: 1em; }}
            hr {{ margin-top: 2em; }}
        </style>
    </head>
    <body>
        <h1>{REPORT_TITLE}</h1>
        <p><em>작성일: {now}</em></p>

        <h2>1. 투자 기업 순위</h2>
        <ul>{toc_entries}</ul>

        <h2>2. {state['domain']} 시장성 요약</h2>
        <p>{domain_market}</p>

        <h2>3. 기업별 분석</h2>
        {body_sections}
    </body>
    </html>
    """

    # 5. PDF 저장
    HTML(string=html_content).write_pdf(REPORT_PATH)

    return REPORT_PATH
