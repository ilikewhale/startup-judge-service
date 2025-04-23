import uuid
from dotenv import load_dotenv
from langchain_teddynote import logging
import asyncio
from market_analysis import run_market_analysis

def main():
    load_dotenv()
    
    logging.langsmith("HAPPY-6-AGENT")

    # user = input("스타트업 분석을 위한 도메인과 나라를 입력해주세요.")
    
    result= run_market_analysis("스포츠")

    print(result)

if __name__ == "__main__":
    asyncio.run(main())