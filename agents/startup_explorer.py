# agents/startup_explorer.py
from util.imports import *
from state import AgentState  # 공통 state 타입

llm = ChatOpenAI(model="gpt-4", temperature=0)
search_tool = TavilySearchResults()

def startup_explorer(state: AgentState) -> Annotated[AgentState, "startup_list"]:
    country = state["country"]
    domain = state["domain"]

    search_query = f"{country} {domain} AI startups Seed Series A"
    web_content = search_tool.invoke({"query": search_query})

    prompt_template = PromptTemplate.from_template("""
    You are a professional AI startup analyst.

    From the following web search results, extract **only the names** of 5 early-stage (Seed or Series A) AI startups in the {domain} domain from {country}.

    The startups must be:
    - Recently founded
    - AI-focused and technically innovative
    - Funded at Seed oxzr Series A stage

    Return a valid Python list of strings only.  
    No explanation. Use only the following content as your knowledge
    Example format:
    ["Startup A", "Startup B", "Startup C", "Startup D", "Startup E"]

    source:
    {web_content}
    """)

    final_prompt = prompt_template.format(country=country, domain=domain, web_content=web_content)
    response = llm([HumanMessage(content=final_prompt)])

    try:
        startup_names = ast.literal_eval(response.content)
        if not isinstance(startup_names, list):
            raise ValueError("응답이 리스트가 아님")
    except Exception as e:
        print("⚠️ 응답 파싱 실패:", e)
        startup_names = []

    return {"startup_list": startup_names}
