import os
import json
import ast
import asyncio

from dotenv import load_dotenv
from typing import Annotated, TypedDict, Sequence, Literal, Dict

from pydantic import BaseModel, Field

# LangChain imports (정리 및 최신화)
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.chat_models import ChatOpenAI 

# from langchain_openai import ChatOpenAI as DeprecatedChatOpenAI
# from langchain.chat_models import ChatOpenAI  # (필요 시 제거 가능)
from langchain.schema import HumanMessage  # 이미 위에서 import됨 (중복 제거 가능)

# LangGraph 관련
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition

# LangChain Teddynote & Opentutorial 관련
from langchain_teddynote import logging
from langchain_teddynote.messages import stream_graph, random_uuid, messages_to_history
from langchain_teddynote.models import get_model_name, LLMs
from langchain_teddynote.evaluator import GroundednessChecker
from langchain_teddynote.tools.tavily import TavilySearch

from langchain_opentutorial.rag.pdf import PDFRetrievalChain
from langchain_community.tools.tavily_search.tool import TavilySearchResults

from state import AgentState
import random
