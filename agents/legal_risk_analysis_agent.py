from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_opentutorial.rag.pdf import PDFRetrievalChain
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from typing import Annotated, Sequence, TypedDict, Literal, Dict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition
from langchain_teddynote.models import get_model_name, LLMs
from langgraph.graph import END, StateGraph, START
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_teddynote.messages import stream_graph
import asyncio
import os

load_dotenv()
logging.langsmith("CH15-Agentic-RAG-Legal")

# 법적 리스크 분석 상태 정의 
class LegalRiskAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    company: str
    domain: str
    tech_summary: str  # 기술 요약 정보
    country: str        # 지역 정보

    legal_assessments: Dict[str, str]


# 모델 이름 설정
MODEL_NAME = get_model_name(LLMs.GPT4)

# PDF 파일로부터 검색 체인 생성
def create_pdf_retriever():
    file_path = ["data/2023 국내외 AI 규제 및 정책 동향.pdf", "data/인공지능(AI) 관련 국내외 법제 동향.pdf"]
    pdf_file = PDFRetrievalChain(file_path).create_chain()
    pdf_retriever = pdf_file.retriever
    
    # PDF 문서를 기반으로 검색 도구 생성
    retriever_tool = create_retriever_tool(
        pdf_retriever,
        "legal_pdf_retriever",
        "Search and return information about AI legal and regulatory frameworks from the PDF files. They contain essential information on AI regulations, policies, and legal trends relevant for AI startups. The documents are focused on both domestic and international AI legal frameworks.",
        document_prompt=PromptTemplate.from_template(
            "<document><context>{page_content}</context><metadata><source>{source}</source><page>{page}</page></metadata></document>"
        ),
    )
    
    return retriever_tool

# 데이터 모델 정의
class grade(BaseModel):
    """A binary score for relevance checks"""
    binary_score: str = Field(
        description="Response 'yes' if the document is relevant to the legal/regulatory question or 'no' if it is not."
    )

# 문서 관련성 평가 함수 (조건부 엣지에서 사용)
def grade_documents(state: LegalRiskAgentState) -> str:
    # LLM 모델 초기화
    model = ChatOpenAI(temperature=0, model=MODEL_NAME, streaming=True)

    # 구조화된 출력을 위한 LLM 설정
    llm_with_tool = model.with_structured_output(grade)

    # 프롬프트 템플릿 정의
    prompt = PromptTemplate(
        template="""You are a legal expert assessing relevance of a retrieved document to an AI startup's legal/regulatory risk question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the question about AI legal/regulatory risks: {question} \n
        If the document contains keyword(s) or semantic meaning related to the legal/regulatory question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # llm + tool 바인딩 체인 생성
    chain = prompt | llm_with_tool

    # 현재 상태에서 메시지 추출
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content

    # 검색된 문서 추출
    retrieved_docs = last_message.content

    # 관련성 평가 실행
    scored_result = chain.invoke({"question": question, "context": retrieved_docs})

    # 관련성 여부 추출
    score = scored_result.binary_score

    # 관련성 여부에 따른 결정
    if score == "yes":
        return "generate"
    else:
        print(score)
        return "rewrite"

# 초기 질의 처리 노드
def initial_query(state: LegalRiskAgentState):
    print("\n🟢 [initial_query] 기업 정보 기반 질문 생성 중...")
    
    company = state['company']
    domain = state['domain']
    tech_summary = state['tech_summary']
    country = state['country']
    
    # 기업, 산업, 기술 요약, 지역을 모두 고려한 질문 생성
    question = f"{company}은(는) {country}에 위치한 {domain} 분야 AI 스타트업으로, 다음 기술을 활용합니다: '{tech_summary}'. 이 기술과 산업 분야를 고려했을 때 해당 지역의 법적/규제 리스크는 무엇인가?"
    
    print(f"➤ 생성된 질문: {question}")
    return {"messages": [HumanMessage(content=question)]}

# 문서 검색 노드 
def pdf_retrieval(state: LegalRiskAgentState):
    print("\n📄 [pdf_retrieval] PDF 기반 법률 문서 검색 시작")
    messages = state["messages"]
    question = messages[-1].content
    retriever_tool = create_pdf_retriever()
    results = retriever_tool.invoke({"query": question})
    print("✅ 검색 완료 - 관련 문서 요약 반환")
    return {"messages": [HumanMessage(content=results)]}

# 질의 재작성 노드
def rewrite(state: LegalRiskAgentState):
    print("\n✍️ [rewrite] 법적 질문이 문서와 관련 없으므로 질의 재작성 시작")
    # 현재 상태에서 메시지 추출
    messages = state["messages"]
    # 원래 질문 추출
    question = messages[0].content
    company = state["company"]
    domain = state["domain"]
    tech_summary = state["tech_summary"]
    country = state["country"]

    # 모든 요소를 고려한 질문 개선을 위한 프롬프트 구성
    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input question about AI legal/regulatory risks for {company} in the {domain} domain in {country} country using technology: '{tech_summary}'.
    Try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question that focuses on specific legal/regulatory frameworks, compliance requirements, or potential legal risks
    for this AI startup, considering their specific technology, domain, and country: """,
        )
    ]

    # LLM 모델로 질문 개선
    model = ChatOpenAI(temperature=0, model=MODEL_NAME, streaming=True)
    # Query-Transform 체인 실행
    response = model.invoke(msg)

    # 재작성된 질문 반환
    print(f"🆕 재작성된 질문: {response.content.strip()[:100]}...")
    return {"messages": [response]}

# Web Search 노드
def web_search(state: LegalRiskAgentState):
    print("\n🌐 [web_search] 웹 기반 보조 법률 정보 검색 시작")
    tavily_tool = TavilySearch()
    
    # 수정된 부분: messages에서 내용 추출
    messages = state["messages"]
    search_query = messages[-1].content
    
    company = state["company"]
    domain = state["domain"]
    tech_summary = state["tech_summary"]
    country = state["country"]
    
    # 검색 쿼리에 기업 정보, 기술 요약, 지역 정보 추가
    enhanced_query = f"{search_query} {company} {domain} {tech_summary} {country} AI 스타트업 법적 규제"

    search_result = tavily_tool.search(
        query=enhanced_query,  # 검색 쿼리
        topic="legal",     # 법률 주제로 변경
        max_results=3,       # 최대 검색 결과
        format_output=True,  # 결과 포맷팅
    )
    print("✅ 웹 검색 완료 - 요약 내용 반환")
    return {"messages": [HumanMessage(content=search_result)]}


# 법적 분석 노드 
def analyze(state: LegalRiskAgentState):
    print("\n🧠 [analyze] 문서 기반 법적 리스크 분석 실행")
    # 현재 상태에서 메시지 추출
    messages = state["messages"]
    # 원래 질문 추출
    question = messages[0].content
    
    # 기업 정보 가져오기
    company = state["company"]
    domain = state["domain"]
    tech_summary = state["tech_summary"]
    country = state["country"]

    # 가장 마지막 메시지 추출 (검색 결과)
    docs = messages[-1].content
    
    # 디버그 메시지 추가
    print(f"기업: {company}")
    print(f"산업: {domain}")
    print(f"기술 요약: {tech_summary}")
    print(f"지역: {country}")
    print(f"질문: {question}")
    print(f"문서 길이: {len(docs) if docs else 0}자")

    # RAG 프롬프트 템플릿 정의 - 기술 요약과 지역 정보 추가
    prompt = PromptTemplate(
        template="""You are a legal expert specialized in AI regulations and policies for startups. 
        Use the following pieces of context to answer the question at the end about {company} in the {domain} domain 
        located in {country} and using the following technology: '{tech_summary}'.
        
        If you don't know the answer, just say you don't know. 
        Don't try to make up an answer.
        
        Always structure your response in the following format: (반드시 한국어로 답변)
        
        1. 법률/규제 분석: 
           - {country} 지역에 특화된 법적/규제 고려사항
           - {domain} 산업에 적용되는 특별 규제 사항
           - {tech_summary} 기술과 관련된 특정 법적 이슈
        
        2. 잠재적 리스크: 
           - 이 AI 스타트업이 직면할 수 있는 구체적인 법적 리스크
           - 기술 특성으로 인한 추가적인 리스크
           - 지역 규제로 인한 특별 고려사항
        
        3. 규정 준수 권장사항: 
           - 규정 준수를 위한 실질적인 단계 제안
           - 기술 개발 및 배포 시 고려해야 할 사항
           - 법적 리스크 최소화를 위한 전략
        
        4. 국제적 고려사항: 
           - 해당되는 경우 관련 국제 프레임워크
           - 국가간 데이터 이전, 사업 확장 시 고려해야 할 점
        
        {context}
        
        Question: {question}
        
        Helpful Answer:""",
        input_variables=["context", "question", "company", "domain", "tech_summary", "country"],
    )

    # LLM 모델 초기화
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0, streaming=True)

    # RAG 체인 구성
    rag_chain = prompt | llm | StrOutputParser()

    try:
        # 답변 생성 실행
        response = rag_chain.invoke({
            "context": docs, 
            "question": question,
            "company": company,
            "domain": domain,
            "tech_summary": tech_summary,
            "country": country
        })
        print("✅ 분석 완료 - 요약 보고 생성")
        return {"messages": [HumanMessage(content=response)]}
    
    except Exception as e:
        print(f"오류 발생: {e}")
        error_msg = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
        return {"messages": [HumanMessage(content=error_msg)]}

# 법적/규제 리스크 분석 결과 처리 노드
def analyze_legal_risks(state: LegalRiskAgentState):
    print("\n📊 [analyze_legal_risks] 최종 평가 내용 저장")
    company = state["company"]
    messages = state["messages"]
    legal_assessment = messages[-1].content
    
    print(f"📝 저장된 평가: {legal_assessment[:100]}...")
    return {
        "legal_assessments": {company: legal_assessment}
    }

# 테크놀로지 리스크 분석 노드 (추가)
def tech_risk_analysis(state: LegalRiskAgentState):
    print("\n🔍 [tech_risk_analysis] 기술 특화 법적 리스크 분석")
    
    # 기업 정보 가져오기
    company = state["company"]
    domain = state["domain"]
    tech_summary = state["tech_summary"]
    country = state["country"]
    
    # 이전 메시지에서 법적 분석 결과 가져오기
    messages = state["messages"]
    legal_analysis = messages[-1].content
    
    # 기술 특화 프롬프트 구성
    msg = [
        HumanMessage(
            content=f"""당신은 AI 기술 및 규제 전문가입니다. 다음 정보를 바탕으로 {company}의 기술이 초래할 수 있는 구체적인 법적/규제 리스크를 분석해주세요:

기업: {company}
산업: {domain}
지역: {country}
기술 요약: {tech_summary}

이전 법적 분석:
{legal_analysis}

이제 기술의 특성에 초점을 맞추어 다음 사항을 분석해주세요:

1. 이 특정 기술이 현재 규제 환경에서 어떤 고유한 법적 문제를 야기할 수 있는지 
2. 이 기술이 해당 산업 내에서 규제 적용을 받을 때의 특별한 고려사항
3. 이 기술이 향후 직면할 수 있는 잠재적인 규제 변화
4. 이 기술의 도입 및 활용에 따른 구체적인 준수 전략

답변은 반드시 한국어로 해주세요."""
        )
    ]
    
    # LLM 모델로 기술 리스크 분석
    model = ChatOpenAI(temperature=0, model=MODEL_NAME, streaming=True)
    response = model.invoke(msg)
    
    print("✅ 기술 특화 리스크 분석 완료")
    return {"messages": [response]}

# 종합 분석 및 권장사항 노드 (추가)
def comprehensive_analysis(state: LegalRiskAgentState):
    print("\n📑 [comprehensive_analysis] 종합 분석 및 권장사항 작성")
    
    # 기업 정보 가져오기
    company = state["company"]
    domain = state["domain"]
    tech_summary = state["tech_summary"]
    country = state["country"]
    
    # 이전 분석 결과 가져오기
    messages = state["messages"]
    legal_analysis = messages[-2].content  # 법적 분석 결과
    tech_analysis = messages[-1].content   # 기술 리스크 분석 결과
    
    # 종합 분석 프롬프트 구성
    msg = [
        HumanMessage(
            content=f"""당신은 AI 스타트업을 위한 법률 및 규제 컨설턴트입니다. {company}에 대한 다음 분석을 바탕으로 종합적인 법적/규제 권장사항을 작성해주세요:

기업: {company}
산업: {domain}
지역: {country}
기술 요약: {tech_summary}

법적 분석:
{legal_analysis}

기술 리스크 분석:
{tech_analysis}

다음 형식으로 최종 권장사항을 작성해주세요 (반드시 한국어로):

## {company} 법적/규제 리스크 종합 평가

### 핵심 리스크 요약
(산업, 기술, 지역을 모두 고려한 3-5가지 핵심 리스크 요약)

### 단기 조치사항 (0-6개월)
(즉각적으로 취해야 할 법적/규제 준수 조치)

### 중기 대응 전략 (6-18개월)
(규제 변화에 대응하기 위한 중기 전략)

### 장기적 고려사항 (18개월 이상)
(장기적인 법적/규제 환경 변화에 대비하기 위한 전략)

### 맞춤형 리스크 관리 프레임워크
(이 기업과 기술에 특화된 리스크 관리 접근법)"""
        )
    ]
    
    # LLM 모델로 종합 분석 실행
    model = ChatOpenAI(temperature=0, model=MODEL_NAME, streaming=True)
    response = model.invoke(msg)
    
    print("✅ 종합 분석 및 권장사항 완료")
    return {"messages": [response]}

# Agentic RAG를 사용한 법적/규제 리스크 분석 그래프 생성
def create_legal_risk_graph():
    """법적 리스크 분석을 위한 LangGraph 워크플로우 생성 함수"""
    
    # 그래프 정의
    workflow = StateGraph(LegalRiskAgentState)

    # 노드 정의
    workflow.add_node("initial_query", initial_query)  # 초기 질의 처리
    workflow.add_node("pdf_retrieval", pdf_retrieval)  # PDF 문서 검색
    workflow.add_node("rewrite", rewrite)             # 질의 재작성
    workflow.add_node("web_search", web_search)       # 웹 검색
    workflow.add_node("analyze", analyze)             # 법적 분석
    workflow.add_node("tech_risk_analysis", tech_risk_analysis)  # 기술 특화 리스크 분석 (추가)
    workflow.add_node("comprehensive_analysis", comprehensive_analysis)  # 종합 분석 및 권장사항 (추가)
    workflow.add_node("legal_risks", analyze_legal_risks)  # 분석 결과 처리

    # 엣지 정의
    workflow.add_edge(START, "initial_query")
    workflow.add_edge("initial_query", "pdf_retrieval")

    # 조건부 엣지: 검색 결과 관련성에 따라 분기
    workflow.add_conditional_edges(
        "pdf_retrieval",
        grade_documents,
        {
            "generate": "analyze",      # 관련성 있으면 분석
            "rewrite": "rewrite"        # 관련성 없으면 재작성
        }
    )

    # 재작성 후 웹 검색 진행
    workflow.add_edge("rewrite", "web_search")
    
    # 웹 검색 후 분석 진행
    workflow.add_edge("web_search", "analyze")
    
    # 분석 후 기술 특화 리스크 분석 진행 (추가)
    workflow.add_edge("analyze", "tech_risk_analysis")
    
    # 기술 리스크 분석 후 종합 분석 및 권장사항 작성 (추가)
    workflow.add_edge("tech_risk_analysis", "comprehensive_analysis")
    
    # 종합 분석 후 최종 결과 처리 (수정)
    workflow.add_edge("comprehensive_analysis", "legal_risks")
    
    # 분석 결과 처리 및 종료
    workflow.add_edge("legal_risks", END)

    # 그래프 컴파일 및 반환
    return workflow.compile()

async def legal_risk_analysis_agent(company: str, domain: str, tech_summary: str, country: str):
    """법적 리스크 분석 에이전트 실행 함수"""
    
    # 법적/규제 리스크 그래프 생성
    legal_graph = create_legal_risk_graph()
    
    # 초기 상태 설정 (legal_assessments 필드 추가)
    initial_state: LegalRiskAgentState = {
        "messages": [HumanMessage(content=f"{company}의 법적 규제 분석")],
        "company": company,
        "domain": domain,
        "tech_summary": tech_summary,
        "country": country,
        "legal_assessments": {}  # 빈 딕셔너리로 초기화
    }
    
    # 그래프 실행
    result = await legal_graph.ainvoke(initial_state)
    
    # 결과 반환 (상위 시스템에서 사용할 수 있도록)
    if "legal_assessments" in result and company in result["legal_assessments"]:
        return result["legal_assessments"][company]
    else:
        return "법적 평가를 완료하지 못했습니다."