from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import asyncio
import sys
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv()


model = ChatOllama(
    model="qwen3:1.7b",
)


client = MultiServerMCPClient(
    {
        "task_server": {
            "command": sys.executable,
            "args": ["mcp_server/server.py"],
            "transport": "stdio",
        },

        "github": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "-e",
                "GITHUB_PERSONAL_ACCESS_TOKEN",
                "-e",
                "GITHUB_TOOLSETS",
                "ghcr.io/github/github-mcp-server",
            ],
            "transport": "stdio",
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ[
                    "GITHUB_PERSONAL_ACCESS_TOKEN"
                ],
                "GITHUB_TOOLSETS": "repos,issues",
            },
        },
    }
)


async def main():
    print("AI Task Assistant")
    print("Type 'exit' to quit.\n")

    
    tools = await client.get_tools()

    # Keep only the tools needed for task and issue management.
    allowed_tools = {
        "get_task_status",
        "list_issues",
        "issue_read",
        "issue_write",
    }

    tools = [tool for tool in tools if tool.name in allowed_tools]

    print("Available tools:", [tool.name for tool in tools])

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

        
        max_tool_rounds = 5

        for _ in range(max_tool_rounds):
            response = await model_with_tools.ainvoke(conversation_history)

            print(
                "DEBUG tool calls:",
                [(call["name"], call["args"]) for call in response.tool_calls]
            )

            conversation_history.append(response)

            if not response.tool_calls:
                break

            tools_by_name = {tool.name: tool for tool in tools}

            for tool_call in response.tool_calls:
                tool = tools_by_name.get(tool_call["name"])

                if tool is None:
                    tool_result = f"Error: unknown tool {tool_call['name']}"
                else:
                    
                    try:
                        args = tool_call["args"]

                        if (
                            tool_call["name"] == "issue_write"
                            and args.get("method") != "create"
                        ):
                            print("\nGitHub issue update requested:")
                            print(f"Repository: {args.get('owner')}/{args.get('repo')}")
                            print(f"Issue: {args.get('title', args.get('issue_number', 'unspecified'))}")

                            confirmation = input("Approve this update? Type 'yes' to continue: ")

                            if confirmation.strip().lower() != "yes":
                                tool_result = "Action cancelled by the user."
                            else:
                                tool_result = await tool.ainvoke(args)
                        else:
                            tool_result = await tool.ainvoke(args)

                    except Exception as error:
                        tool_result = f"Tool execution failed: {error}"


                conversation_history.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )
        else:
            print("Maximum tool rounds reached.")

        print(f"AI: {response.content}\n")



if __name__ == "__main__":
    asyncio.run(main())