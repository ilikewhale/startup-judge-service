import asyncio
from competitor_analysis import competitor_analysis
# from agents.competitor_analysis_agent import competitor_analysis

async def main():
    result = await competitor_analysis("SK")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())