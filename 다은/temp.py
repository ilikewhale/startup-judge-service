import asyncio
from founder_explorer import analyze_startup_founder

async def main():
    company_name = "올거나이즈"
    domain = "AI"

    result = await analyze_startup_founder(company_name, domain)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())