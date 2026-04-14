from typing import List, Dict, Any, Optional

from ..tools.tool1 import Structured_Data_Query_Tool
from ..tools.tool2 import External_API_Simulation_Tool


def render_prompt(
    messages: List[Dict[str, Any]],
    system: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    is_think_set: bool = False,
    think: bool = False,
) -> str:
    output = []
    """
    Renders a prompt for the LLM agent, including system instructions, tool signatures,
    and conversation history formatted for function calling.

    Args:
        messages (List[Dict[str, Any]]): List of conversation messages, each with role and content.
        system (Optional[str]): System prompt or instructions for the agent.
        tools (Optional[List[Dict[str, Any]]]): List of tool/function signatures available to the agent.
        is_think_set (bool): Whether 'think' mode is enabled for the agent.
        think (bool): Current thinking state.

    Returns:
        str: The formatted prompt string for the LLM agent.
    """
    # ---------------------------------------
    # Find last user index
    # ---------------------------------------
    last_user_idx = -1
    for idx, msg in enumerate(messages):
        if msg.get("role") == "user":
            last_user_idx = idx

    # ---------------------------------------
    # System block (if system or tools exist)
    # ---------------------------------------
    if system or tools:
        output.append("<|im_start|>system")

        if system:
            output.append(system)

        if tools:
            output.append("""
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>""")

            for tool in tools:
                output.append(
                    f'{{"type": "function", "function": {tool["function"]}}}'
                )

            output.append("""</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
""")

        output.append("<|im_end|>")

    # ---------------------------------------
    # Messages rendering
    # ---------------------------------------
    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")
        thinking = msg.get("thinking")
        tool_calls = msg.get("tool_calls")

        last = (i == len(messages) - 1)

        # ---------------- USER ----------------
        if role == "user":
            output.append("<|im_start|>user")
            output.append(content)

            if is_think_set and i == last_user_idx:
                output.append(" /think" if think else " /no_think")

            output.append("<|im_end|>")

        # ---------------- ASSISTANT ----------------
        elif role == "assistant":
            output.append("<|im_start|>assistant")

            if (
                is_think_set
                and thinking
                and (last or i > last_user_idx)
            ):
                output.append(f"<think>{thinking}</think>")

            if content:
                output.append(content)
            elif tool_calls:
                output.append("<tool_call>")
                for call in tool_calls:
                    output.append(
                        f'{{"name": "{call["function"]["name"]}", '
                        f'"arguments": {call["function"]["arguments"]}}}'
                    )
                output.append("</tool_call>")

            if not last:
                output.append("<|im_end|>")

        # ---------------- TOOL ----------------
        elif role == "tool":
            output.append("<|im_start|>user")
            output.append("<tool_response>")
            output.append(content)
            output.append("</tool_response>")
            output.append("<|im_end|>")

        # If last message is not assistant, open assistant block
        if role != "assistant" and last:
            output.append("<|im_start|>assistant")
            if is_think_set and not think:
                output.append("<think>\n\n</think>")

    return "\n".join(output)

def tools_prompt(tools):
    """
    Generate a prompt string that describes available tools for function calling.

    The prompt includes tool signatures inside <tools></tools> XML tags,
    followed by instructions on how to call them using <tool_call></tool_call> tags.

    Args:
        tools (list): A list of tool dictionaries. Each tool dictionary must contain
                      a "function" key with the function specification (as expected
                      by the model's function calling API).

    Returns:
        str: A formatted prompt string containing tool definitions and calling instructions.
    """
    output = []

    # System instruction for tools
    output.append("""
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>""")
    
    for tool in tools:
        output.append(
            f'{{"type": "function", "function": {tool["function"]}}}'
        )

    output.append("""</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
""")
    return "\n".join(output)

if __name__ == "__main__":
    question = "What is the SLA for Premium Support?"
    system = (
        "You are an assistant with access to tools. "
        "When a tool provides information, you MUST use the tool result to answer. "
        "The tool result is real and accurate."
    )
    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "External_API_Simulation_Tool",
                "description": External_API_Simulation_Tool.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the external API endpoint."
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method to use.",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                        "default": "GET"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers to include in the request.",
                        "additionalProperties": {
                        "type": "string"
                        }
                    },
                    "payload": {
                        "type": "object",
                        "description": "JSON body payload for POST or PUT requests.",
                        "additionalProperties": True
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Request timeout in seconds.",
                        "default": 5,
                        "minimum": 1
                    },
                    "retries": {
                        "type": "integer",
                        "description": "Number of retry attempts if request fails.",
                        "default": 3,
                        "minimum": 0
                    }
                    },
                    "required": ["url"]
                }
                }
        },
        {
            "type": "function",
            "function": {
                "name": "Structured_Data_Query_Tool",
                "description": Structured_Data_Query_Tool.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "filters": {"type": "object"}
                    },
                    "required": ["table", "filters"]
                }
            }
        }
    ]
    
    # Prompt Template
    #print(render_prompt(messages, tools=tools, system=system, is_think_set=True, think=False))
    print(tools_prompt(tools))