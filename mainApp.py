from typing import Dict, Any, Annotated
from typing_extensions import TypedDict
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from tools import get_all_tools
from langchain_core.messages import ToolMessage
from llm import get_llm
from utils import load_config

# from guardrails import Guard, OnFailAction
# from guardrails.hub import ToxicLanguage, UnusualPrompt

load_dotenv()

config = load_config()
llm = get_llm(config["llm"])


class State(TypedDict):
    messages: Annotated[list, add_messages]


def llm_node(state: State):
    """Node that handles LLM invocation."""
    # Get tools and create LLM with tools
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)

    # Invoke the LLM
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tools_node(state: State):
    """Node that handles tool execution."""
    tool_registry = create_tool_registry()

    # Get the last message (should be from LLM with tool calls)
    last_message = state["messages"][-1]

    tool_messages = []
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Execute all tool calls
        for tool_call in last_message.tool_calls:
            result = execute_tool_call(tool_call, tool_registry)
            # Create tool message
            tool_message = ToolMessage(
                content=str(result), tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_message)

    return {"messages": tool_messages}


def should_continue(state: State):
    """Determine whether to continue to tools or end."""
    last_message = state["messages"][-1]

    # If the last message has tool calls, go to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, we're done
    return END


def create_graph():
    """Create and configure the LangGraph workflow."""
    # Create the graph
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tools_node)

    # Set entry point
    graph.set_entry_point("llm")

    # Add conditional edges
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})

    # After tools, always go back to LLM
    graph.add_edge("tools", "llm")

    return graph.compile()


def visualize_graph(graph: StateGraph, save_path: str):
    """Visualize the graph."""
    png = graph.get_graph().draw_mermaid_png()
    with open(save_path, "wb") as f:
        f.write(png)


def create_tool_registry() -> Dict[str, Any]:
    """Create a registry mapping tool names to their functions."""
    tools = get_all_tools()
    return {tool.name: tool for tool in tools}


def execute_tool_call(tool_call: Dict[str, Any], tool_registry: Dict[str, Any]) -> Any:
    """Execute a single tool call and return the result."""
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name in tool_registry:
        tool_function = tool_registry[tool_name]
        result = tool_function.invoke(tool_args)
        print(f"🔧 Tool used: {tool_name} with args {tool_args} → Result: {result}")
        return result
    else:
        print(f"Unknown tool: {tool_name}")
        return f"Error: Tool '{tool_name}' not found"

# def validate_input(input_text: str) -> bool:
#     guard = Guard().use_many(
#         ToxicLanguage(threshold=0.5, on_fail=OnFailAction.EXCEPTION),
#         UnusualPrompt(on_fail=OnFailAction.EXCEPTION),
#     )

#     guard.validate(input_text)

def main():

    print("LangGraph Chatbot with Custom Tools")
    print("Type 'exit' or 'quit' to end the session.")

    # Create the graph
    app = create_graph()

    # Display available tools
    tool_registry = create_tool_registry()
    print(f"Available tools: {', '.join(tool_registry.keys())}\n")

    # System message with dynamic tool information
    tool_descriptions = "\n".join(
        [f"- {name}: {tool.description}" for name, tool in tool_registry.items()]
    )
    system_content = f"""You are a helpful AI assistant. Remember the previous messages in this conversation. 

You have access to the following tools:
{tool_descriptions}

Use these tools when appropriate to help answer questions."""

    # Initialize conversation state
    initial_state = {"messages": [SystemMessage(content=system_content)]}

    try:
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break

            # validate_input(user_input)
            # Add user message to state
            initial_state["messages"].append(HumanMessage(content=user_input))

            # Run the graph
            result = app.invoke(initial_state)

            # Update state with results
            initial_state["messages"] = result["messages"]

            # Display the final response
            last_message = result["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                print(f"Bot: {last_message.content}\n")

    except KeyboardInterrupt:
        print("\n👋 Session terminated.")


if __name__ == "__main__":
    main()