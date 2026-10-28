# ---------------------------------------------------------
# Command-line runner: ask one question and print each agent step
# Usage: python ask.py "What changed in the latest Python release?"
# ---------------------------------------------------------

import asyncio
import sys

from langchain_core.messages import AIMessage, ToolMessage

from agent import build_agent


async def main(question: str):
    agent = await build_agent()
    print(f"Question: {question}\n")

    async for step in agent.astream({"messages": [("user", question)]}):
        for node_output in step.values():
            for msg in node_output.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"-> {tc['name']}({tc['args']})")
                elif isinstance(msg, ToolMessage):
                    preview = msg.content if isinstance(msg.content, str) else str(msg.content)
                    print(f"   {msg.name} returned {len(preview)} chars")
                elif isinstance(msg, AIMessage) and msg.content:
                    print(f"\nAnswer:\n{msg.content}")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "What are the main features in the latest Python release?"))
