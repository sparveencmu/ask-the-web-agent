# ---------------------------------------------------------
# Ask-the-Web agent: search, read the best pages, answer with sources
# ---------------------------------------------------------

import os
import shutil

from ddgs import DDGS
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

SYSTEM_PROMPT = """You are a research assistant that answers questions using the web.

Tools:
- search_web: find relevant pages (returns titles, URLs, and short snippets).
- fetch: read the full content of a URL.

How to work:
1. For questions that need current or specific factual information, call search_web first.
2. Pick the 1-2 most relevant results and call fetch on their URLs to read the details.
   Snippets alone are often incomplete or out of date.
3. Answer from what you read. Cite sources inline as [1], [2] and list their URLs at the end.

Answer greetings and general knowledge directly, without tools.
If the pages you read don't answer the question, say so instead of guessing.
"""


def make_llm():
    """Local Ollama model by default; set LLM_PROVIDER=groq to use a hosted model."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "groq":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0,
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(model=os.getenv("LLM_MODEL", "llama3.2:3b"), temperature=0)


@tool
def search_web(query: str) -> str:
    """Search the web and return the top results with title, URL, and a short snippet."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"Search failed ({type(e).__name__}). Try a different query."
    if not results:
        return "No results found."
    return "\n\n".join(
        f"[{i}] {r['title']}\nURL: {r['href']}\n{r['body']}" for i, r in enumerate(results, 1)
    )


def path_without_node():
    # mcp-server-fetch extracts articles with readabilipy, which runs Readability.js
    # through Node when it finds it. readabilipy installs the newest jsdom, which fails
    # to load on Node < 22 (ERR_REQUIRE_ESM), so every fetch errors out. Hiding Node
    # makes readabilipy use its pure-Python extractor, which works everywhere.
    dirs = os.environ.get("PATH", "").split(os.pathsep)
    return os.pathsep.join(d for d in dirs if not os.path.exists(os.path.join(d, "node")))


async def load_mcp_tools():
    """Load the fetch tool from the MCP fetch server (mcp-server-fetch)."""
    client = MultiServerMCPClient(
        {
            "fetch": {
                "command": shutil.which("mcp-server-fetch") or "mcp-server-fetch",
                "args": [],
                "transport": "stdio",
                "env": {**os.environ, "PATH": path_without_node()},
            }
        }
    )
    tools = await client.get_tools()

    # Small models sometimes pass null for optional params, which fails the
    # server's JSON-schema validation. Strip None values before the call.
    for t in tools:
        original = t.coroutine

        async def without_nulls(_original=original, **kwargs):
            return await _original(**{k: v for k, v in kwargs.items() if v is not None})

        t.coroutine = without_nulls
        # Some pages fail to fetch or parse. Return the error to the model
        # so it can try another source instead of ending the run.
        t.handle_tool_error = True
    return tools


async def build_agent():
    tools = [search_web] + await load_mcp_tools()
    return create_agent(make_llm(), tools, system_prompt=SYSTEM_PROMPT)
