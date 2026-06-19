import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def web_search(query: str) -> str:
    """
    Search the web using Tavily and return a combined string of results.
    Used as fallback when retrieved PDF chunks are not relevant.
    """
    response = client.search(query=query, max_results=3)
    
    # Combine all result snippets into one string
    results = []
    for r in response.get("results", []):
        results.append(f"Source: {r['url']}\n{r['content']}")
    
    return "\n\n---\n\n".join(results)