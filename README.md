# 🧠 AI Startup Investment Evaluation Agent

본 프로젝트는 인공지능 스타트업에 대한 투자 가능성을 자동으로 평가하는</br>
**Agentic RAG 기반의 멀티 에이전트 시스템**을 설계하고 구현한 실습 프로젝트입니다.

---

## 📌 Project Overview

- **Objective**
  
  사용자가 제시한 도메인 내 여러 AI 스타트업을 자동으로 탐색하고,  
  각 스타트업의 핵심 정보를 수집하여 투자 적합성을 종합적으로 평가합니다.

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
| Retrieval  | Tavily API, FAISS         |

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
(그래프 이미지 삽입 예정 - 멀티 에이전트 기반 평가 흐름)

---

## 📁 Directory Structure

## Contributors 
- 김철수 : Prompt Engineering, Agent Design 
- 최영희 : PDF Parsing, Retrieval Agent 
