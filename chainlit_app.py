# ---------------------------------------------------------
# Chainlit Web Search Agent
# ---------------------------------------------------------

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent import build_agent


@cl.on_chat_start
async def start():
    # Built per session: loading the MCP fetch tool is async
    cl.user_session.set("agent", await build_agent())
    cl.user_session.set("history", [])
    await cl.Message(
        content="Welcome! I'm a web research agent. Ask me anything and I'll search the web, read the most relevant pages, and answer with sources."
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    agent = cl.user_session.get("agent")
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
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    async with cl.Step(name=f"Tool results: {msg.name}") as s:
                        s.output = f"```\n{content}\n```"

                elif isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content

    if final_answer:
        history.append(AIMessage(content=final_answer))
        await cl.Message(content=final_answer).send()
    else:
        await cl.Message(content="I couldn't generate a response. Please try again.").send()

    cl.user_session.set("history", history)
