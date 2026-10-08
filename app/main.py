from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient


model = ChatOllama(
    model="qwen3:1.7b",
)


client = MultiServerMCPClient(
    {
        "task_server": {
            "command": sys.executable,
            "args": ["mcp_server/server.py"],
            "transport": "stdio",
        }
    }
)


async def main():
    print("AI Task Assistant")
    print("Type 'exit' to quit.\n")

    tools = await client.get_tools()

    print("MCP tools:", [tool.name for tool in tools])

    model_with_tools = model.bind_tools(tools)

    conversation_history = [
        SystemMessage(
            content=(
                "You are an AI task assistant for software development projects. "
                "Have natural and concise conversations. "
                "Answer the user's question directly. "
                "Do not provide long lists of suggestions unless the user asks for them. "
                "Do not recommend libraries or technologies unless they are relevant to the user's request. "
                "Only ask a follow-up question when necessary to complete the user's request."
            )
        )
    ]

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        conversation_history.append(
            HumanMessage(content=user_input)
        )

        response = model_with_tools.invoke(conversation_history)

        conversation_history.append(response)

        if response.tool_calls:
            tools_by_name = {tool.name: tool for tool in tools}

            for tool_call in response.tool_calls:
                tool = tools_by_name[tool_call["name"]]

                tool_result = await tool.ainvoke(
                    tool_call["args"]
                )

                conversation_history.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

            response = model_with_tools.invoke(conversation_history)

            conversation_history.append(response)

        print(f"AI: {response.content}\n")


if __name__ == "__main__":
    asyncio.run(main())