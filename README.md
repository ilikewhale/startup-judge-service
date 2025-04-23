# 🧠 AI Startup Investment Evaluation Agent

본 프로젝트는 인공지능 스타트업에 대한 투자 가능성을 자동으로 평가하는</br>
**Agentic RAG 기반의 멀티 에이전트 시스템**을 설계하고 구현한 실습 프로젝트입니다.

---

## 📌 Project Overview

- **Objective**
  
  사용자가 제시한 도메인 내 여러 AI 스타트업을 자동으로 탐색하고,  
  도메인의 시장성 분석과 각 스타트업의 핵심 정보(기술력, 리스크, 창업자, 경쟁사)를 수집하여</br>
  투자 적합성을 종합적으로 평가합니다.

- **Method**
  - **Agentic RAG (Retrieval-Augmented Generation)** 기반의 멀티 에이전트 아키텍처
  - 기술력, 창업자 평판, 시장성, 리스크, 경쟁사 등 주요 요소를 평가하는 전용 Agent 구성
  - 병렬 실행 및 조건 기반 루프(flow control) 설계
  - 자동 판단 및 투자 보고서 생성까지 완전 자동화된 투자 평가 파이프라인

- **Tools**
  - **Tavily Web Search API**: 실시간 웹 탐색을 통한 정보 수집
  - **PDF Retriever**: IR 문서, 시장 보고서 등에서 정보 추출
  - **OpenAI GPT-4o-mini**: 요약 및 판단 생성에 사용되는 LLM

---

## ✅ Features

- 다양한 형식의 외부 정보(PDF, 웹 등)를 자동 수집 및 정제
- 기술력, 창업자 역량, 시장성, 법적 리스크 등 투자 판단 기준별 분석
- 투자 적합성에 따른 **요약 리포트 자동 생성** (예: 유망 / 보류 / 회피)

---

## 🛠️ Tech Stack

| Category   | Details                             |
|------------|-------------------------------------|
| Framework  | LangGraph, LangChain, Python        |
| LLM        | GPT-4o-mini          |
| Retrieval  | Tavily API,  LangChain.PDFRetrievalChain       |

---

## 🤖 Agents

| 에이전트 | 설명 |
|----------|------|
| 🔍 **스타트업 탐색 에이전트** | 관심 도메인의 유망 AI 스타트업을 웹 기반으로 수집합니다. |
| 🔧 **기술 요약 에이전트** | 스타트업의 핵심 기술, 구현 가능성, 기술의 장단점을 정리합니다. |
| 🙋 **창업자 평판 에이전트** | 창업자의 경력과 평판을 평가합니다 (SNS/언론 긍정도 포함). |
| 📊 **시장성 분석 에이전트** | 시장 규모, 수요, 산업 성장 가능성 등을 분석합니다. |
| ⚖️ **법적/규제 리스크 에이전트** | 해당 산업의 법적/정책적 리스크를 확인합니다. |
| 🥊 **경쟁사 비교 에이전트** | 경쟁사의 전략 및 차별성과 비교하여 평가합니다. |
| 🧮 **투자 판단 에이전트** | 종합 판단을 통해 투자 여부를 결정합니다. |
| 📝 **보고서 생성 에이전트** | 각 평가 결과를 기반으로 투자 요약 리포트를 생성합니다. |

---

## 🧩 Architecture  
![image](https://github.com/user-attachments/assets/1efeb2da-6e61-4d10-b085-1de38de29b62)

<details>
<summary>Click to toggle! Agents Workflow</summary>
  <img src="![스타트업 투자 평가 에이전트](https://github.com/user-attachments/assets/cabf7626-d641-4d92-8e03-5fda3396f4c0)"
  <img src="https://github.com/user-attachments/assets/3281c8ab-26f4-404e-8023-8a55e9491132", alt="agents1">
  <img src="https://github.com/user-attachments/assets/45bc23af-db87-4206-98ce-6d05d9f1be4a", alt="agents2">
</details>

---
```
## 📁 Directory Structure
📦 STARTUP-J...
 ┣ 📂 _pycache_
 ┣ 📂 agents
 ┣ 📂 data
 ┣ 📂 reports
 ┣ 📂 founder_reputation_agent
 ┣ 📂 market_analysis_agent
 ┣ 📂 startup_explore_agent
 ┣ 📂 legal_risk_agent
 ┣ 📂 tech_summary_agent
 ┣ 📂 competitor_analysis_agent
 ┣ 📜 .env
 ┣ 📜 .gitignore
 ┣ 📜 main.ipynb
 ┣ 📜 README.md
 ┗ 📜 state.py
```

---

## 💫 Contributors 
<div align="center">

| **김다은** | **김민주** | **손지영** | **이재웅** | **이효정** | **진실** |
| :--------: | :--------: | :--------: | :--------: | :--------: | :------: |
| <img src="https://avatars.githubusercontent.com/u/98153670?v=4" width="100" height="100"> | <img src="https://avatars.githubusercontent.com/u/74577811?v=4" width="100" height="100"> | <img src="https://avatars.githubusercontent.com/u/122194456?v=4" width="100" height="100"> | <img src="https://avatars.githubusercontent.com/u/60501045?v=4" width="100" height="100"> | <img src="https://avatars.githubusercontent.com/u/79013520?v=4" width="100" height="100"> | <img src="https://avatars.githubusercontent.com/u/97718539?v=4" width="100" height="100"> |
| [@ilikewhale](https://github.com/ilikewhale) | [@alswnsp411](https://github.com/alswnsp411) | [@zi0-hand](https://github.com/zi0-hand) | [@ww5702](https://github.com/ww5702) | [@world-dv](https://github.com/world-dv) | [@zinsile](https://github.com/zinsile) |
| 창업자 평판 에이전트 구현, 전체 flow chart 도면 구성 | 시장성 분석 에이전트 구현 | 법적/규제 리스크 분석 에이전트 구현, 비동기 전체 flow 구성 | 스타트업탐색, 투자판단, 보고서 생성 에이전트 구현, 비동기 전체 flow 구성 | 경쟁사 비교 에이전트 구현, 비동기 전체 flow 구성 | 기술 요약 에이전트 구현, 에이전트 별 flow chart 도면 구성 및 리드미 작성 |

</div>
