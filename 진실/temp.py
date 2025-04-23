import asyncio
from tech_analysis import tech_analysis

async def main():
    result = await tech_analysis("메이아이")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())