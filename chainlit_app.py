# ---------------------------------------------------------
# Chainlit Web Search Agent
# ---------------------------------------------------------

import chainlit as cl
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from ddgs import DDGS

MODEL = "llama3.2:3b"
# MODEL = "gemma4:e2b"
# MODEL = "gemma3:1b"
llm = ChatOllama(model=MODEL, temperature=0)


@tool
def search_web(query: str) -> str:
    """Search the web for a query and return results."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(f"- {r['title']}: {r['href']}")
    return "\n".join(results) if results else "No results found."


system_prompt = """You are a helpful assistant with access to a web search tool.

Rules:
- Only search the web for questions that need current, real-time, or factual information you are unsure about.
- For general knowledge, greetings, or simple questions, answer directly without searching.
- If a search does not return useful results, answer based on your own knowledge.
- Give a clear, concise final answer.
"""

agent = create_agent(llm, [search_web], system_prompt=system_prompt)


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="Welcome! I'm a web search agent. Ask me anything and I'll search the web to find answers."
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history", [])
    history.append(HumanMessage(content=message.content))

    final_answer = ""

    async for step in agent.astream({"messages": history}):
        for node_name, node_output in step.items():
            if "messages" not in node_output:
                continue

            for msg in node_output["messages"]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        async with cl.Step(name=f"Calling tool: {tc['name']}") as s:
                            s.output = f"```\n{tc['args']}\n```"

                elif isinstance(msg, ToolMessage):
                    content = msg.content
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    async with cl.Step(name="Tool results") as s:
                        s.output = f"```\n{content}\n```"

                elif isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content

    if final_answer:
        history.append(AIMessage(content=final_answer))
        await cl.Message(content=final_answer).send()
    else:
        await cl.Message(content="I couldn't generate a response. Please try again.").send()

    cl.user_session.set("history", history)
