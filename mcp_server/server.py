from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Task Assistant")


@mcp.tool()
def get_task_status(task_name: str) -> str:
    """Get the status of a task in the project."""

    tasks = {
        "login system": "The login system is not started yet.",
        "chatbot": "The chatbot is currently in development.",
        "github integration": "The GitHub integration is planned.",
    }

    return tasks.get(
        task_name.lower(),
        f"I don't have a status for the task '{task_name}'."
    )


if __name__ == "__main__":
    mcp.run()