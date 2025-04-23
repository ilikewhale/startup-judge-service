
from dotenv import load_dotenv

load_dotenv()

from langchain_teddynote.tools.tavily import TavilySearch

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI
from langchain_teddynote.evaluator import GroundednessChecker
from langchain_teddynote.messages import messages_to_history

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, random_uuid
from langgraph.graph import StateGraph, START, END

# Replace this deprecated import
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI  # Updated import

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
import os

from langchain_openai import ChatOpenAI
from langchain_teddynote.evaluator import GroundednessChecker
from langchain_teddynote.messages import messages_to_history
from langchain_teddynote.tools.tavily import TavilySearch

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

# 그래프 상태
class TechSummaryAgent(TypedDict):
    question: Annotated[str, "Question"] # 질문
    context: Annotated[str, "Context"]
    messages: Annotated[list[BaseMessage], "Messages"] # 메시지(누적되는 list)
    relevance: Annotated[str, "Relevance"]  # 관련성

def llm_answer(state: TechSummaryAgent) -> TechSummaryAgent:
    latest_question = state["question"]
    context = state["context"]  # 웹 검색 결과로 받은 경쟁사 분석 정보

    # 경쟁사 분석 결과를 요약하고 핵심 인사이트를 추출하는 프롬프트
    report_prompt = f"""
    당신은 {latest_question}의 기술력을 분석하는 전문가입니다.
    주어진 정보를 바탕으로 다음 구조로 요약해주세요:

    1. 핵심 기술 개요: 기업의 주요 기술과 특징을 간략하게 요약
    2. 기술적 장점: 경쟁사 대비 우수한 점
    3. 기술적 단점 또는 과제: 개선이 필요한 부분
    4. 기술 경쟁력 평가: 전반적인 기술 경쟁력에 대한 평가

    객관적인 사실에 기반하여 작성하고, 정보가 부족한 부분은 '정보 없음'으로 표시하세요.
    기술 분석 정보 : {context}
    """

    response = llm.invoke(report_prompt)

    # 생성된 답변과 (유저의 질문, 답변) 메시지를 상태에 저장
    return TechSummaryAgent(
        answer=response,
        messages=[("user", latest_question), ("assistant", response)]
    )

# 관련성 체크 노드
def relevance_check(state: TechSummaryAgent) -> TechSummaryAgent:
    # 관련성 평가기 생성
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0), target="question-retrieval"
    ).create()

    # 관련성 체크를 실행("yes" or "no")
    response = question_answer_relevant.invoke(
        {"question": state["question"], "context": state["context"]}
        # {"question": state["question"][-1].content, "context": state["context"]}
    )
    # print("==== [RELEVANCE CHECK] ====")  
    # print(response.score)  
    return TechSummaryAgent(relevance=response.score)


# 관련성 체크하는 함수(router)
def is_relevant(state: TechSummaryAgent) -> TechSummaryAgent:
    return state["relevance"]

# Web Search 노드
def web_search(state: TechSummaryAgent) -> TechSummaryAgent:
    tavily_tool = TavilySearch()

    search_query = state["question"]
    # search_query = state["question"][-1].content

    search_result = tavily_tool.search(
        query=search_query,  # 검색 쿼리
        topic="general",     # 일반 주제
        max_results=3,       # 최대 검색 결과
        format_output=True,  # 결과 포맷팅
    )

    return TechSummaryAgent(context="\n".join(search_result))

# Query Rewrite 노드
def query_rewrite(state: TechSummaryAgent) -> TechSummaryAgent:
    """
    Rewrites a question to make it more effective for retrieving information about 
    a startup's core technologies and their strengths and weaknesses.
    
    Args:
        state: TechSummaryAgent containing the original question
        
    Returns:
        TechSummaryAgent with the rewritten question
    """
    # Query Rewrite 프롬프트 정의
    re_write_prompt = PromptTemplate(
        template="""You are an expert in query optimization for startup technology evaluation. Reformulate the given question to make it more effective for retrieving information about a startup's core technologies and their strengths and weaknesses.

        - Identify the startup name and focus the reformulated question on retrieving detailed technical information.
        - Ensure the rewritten query enables searching through sources like company websites, technical blogs, academic papers, and product documentation.
        - Emphasize keywords related to technology overview, technical strengths, weaknesses, differentiation, and practical use cases.

        # Output Format

        - Provide a single, rewritten question.
        - Do not include any explanatory or introductory text—output only the question.

        # Examples

        **Input**:
        "What is the technology behind MayAI?"

        **Output**:
        "What are the core technologies developed by MayAI, and what are their key advantages and limitations?"

        **Input**:
        "How good is Upstage's tech?"

        **Output**:
        "What AI technologies has Upstage developed, and how do their strengths and limitations compare to competitors?"

        # Notes

        - The rewritten question must retain the original intent (evaluating technical capabilities).
        - Avoid generic or overly simplified phrasing.
        - Ensure the reformulated question is concise, technically focused, and suitable for information retrieval tasks.

        # Original Question:
        {question}
        """,
        input_variables=["question"],
    )

    question_rewriter = (
        re_write_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()
    )

    latest_question = state["question"]
    question_rewritten = question_rewriter.invoke({"question": latest_question})

    return TechSummaryAgent(question=question_rewritten)

# 기술 요약 정리 그래프 생성
def create_tech_summary_graph():
    workflow = StateGraph(TechSummaryAgent)

    workflow.add_node("relevance_check", relevance_check)
    workflow.add_node("llm_answer", llm_answer)
    workflow.add_node("web_search", web_search)
    workflow.add_node("query_rewrite", query_rewrite)  # Query Rewrite 노드 추가

    workflow.add_edge("query_rewrite", "web_search")      # 쿼리 재작성 -> 검색
    workflow.add_edge("web_search", "relevance_check")    # 검색 -> 관련성 체크


    workflow.add_conditional_edges(
        "relevance_check",
        is_relevant,
        {
            "yes": "llm_answer",
            "no": "query_rewrite",
        },
    )

    workflow.add_edge("llm_answer", END)

    workflow.set_entry_point("query_rewrite")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

async def tech_analysis(company: str):
    app = create_tech_summary_graph()
    config = RunnableConfig(recursion_limit=10, configurable={"thread_id": random_uuid()})
    inputs = TechSummaryAgent(question=company)
    
    # 그래프 실행
    result = await app.ainvoke(inputs, config)
    
    # 결과에서 messages 리스트 가져오기
    messages = result.get("messages", [])
    
    # messages가 비어있지 않다면 마지막 메시지의 AIMessage content 반환
    if messages and messages[-1]:
        last_message = messages[-1]
        
        # 튜플 형태이고 두 번째 요소가 AIMessage 객체인 경우
        if isinstance(last_message, tuple) and len(last_message) > 1 and hasattr(last_message[1], 'content'):
            return last_message[1].content

# result = await tech_analysis("메이아이")
# print(result)