# 🧠 AI Startup Investment Evaluation Agent

본 프로젝트는 인공지능 스타트업에 대한 투자 가능성을 자동으로 평가하는</br>
**Agentic RAG 기반의 멀티 에이전트 시스템**을 설계하고 구현한 실습 프로젝트입니다.

---

## 📌 Project Overview

- **Objective**
  
  사용자가 제시한 도메인과 지역 내 여러 AI 스타트업을 자동으로 탐색하고,  
  도메인의 시장성 분석과 각 스타트업의 핵심 정보(기술력, 리스크, 창업자, 경쟁사)를 수집하여</br>
  투자 적합성을 종합적으로 평가합니다.

- **Method**
  - **AI Agent + Agentic RAG**</br>
    Retrieval-Augmented Generation(RAG)을 기반으로, 역할별 전문 Agent들이 협업하는 구조로 설계되어 정밀한 투자 분석이 가능합니다.

- **Tools**
  - **Tavily Web Search API**: 실시간 웹 탐색을 통한 정보 수집
  - **PDF Retriever**: IR 문서, 시장 보고서 등에서 정보 추출
  - **OpenAI GPT-4o-mini**: 요약 및 판단 생성에 사용되는 LLM

---

## ✅ Features

- **외부 비정형 정보 자동 수집 및 정제**</br>
  웹 문서, PDF 등 다양한 포맷의 데이터를 자동으로 수집하고, 투자 분석에 필요한 핵심 정보를 정제합니다.

- **투자 판단 기준별 전용 Agent 구성**</br>
  기술력, 창업자 역량, 시장성, 경쟁환경, 법적 리스크 등 주요 평가 항목에 따라 전용 Agent를 구성하여 각 요소를 체계적으로 분석합니다.

- **병렬 실행 및 조건 기반 흐름 제어**</br>
  각 분석 Agent는 병렬로 실행되며, 정보 부족 시 재탐색 등의 조건 기반 루프 로직을 통해 유연한 흐름 제어가 가능합니다.

- **자동 투자 평가 리포트 생성**</br>
  분석 결과를 종합하여 투자 적합성에 따라 '유망', '보류', '회피' 등으로 분류된 요약 보고서를 자동 생성합니다.</br>
  Scorecard Method 로 score table로 투자 적합성 점수 계산합니다.
  
  | 항목 | 비중(%) | 평가 포인트 |
  | --- | --- | --- |
  | 창업자 (Owner) | 30% | 전문성, 커뮤니케이션, 실행력 |
  | 시장성 (Opportunity Size) | 25% | 시장 크기, 성장 가능성  |
  | 제품/기술력 | 15% | 독창성, 구현 가능성 |
  | 경쟁 우위 | 10% | 진입장벽, 특허, 네트워크 효과 |
  | 실적 | 10% | 매출, 계약, 유저수 등 |
  | 투자조건 (Deal Terms) | 10% | Valuation, 지분율 등  |

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
  <img src="https://github.com/user-attachments/assets/3281c8ab-26f4-404e-8023-8a55e9491132", alt="agents1">
  <img src="https://github.com/user-attachments/assets/45bc23af-db87-4206-98ce-6d05d9f1be4a", alt="agents2">
</details>

---

## 📁 Directory Structure
```
📦 STARTUP-JUDGE-SERVICE
 ┣ 📂 _pycache_
 ┣ 📂 agents
 ┣ 📂 data
 ┣ 📂 reports
 ┣ 📜 .env
 ┣ 📜 .gitignore
 ┣ 📜 main.py
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
| 창업자 평판 에이전트 구현, 전체 flow chart 도면 구성 | 시장성 분석 에이전트 구현, 전체 flow chart 도면 구성 | 법적/규제 리스크 분석 에이전트 구현, 비동기 전체 flow 구성 | 스타트업탐색, 투자판단, 보고서 생성 에이전트 구현, 비동기 전체 flow 구성 | 경쟁사 비교 에이전트 구현, 비동기 전체 flow 구성 | 기술 요약 에이전트 구현, 에이전트 별 flow chart 도면 구성 및 리드미 작성 |

</div>
