import re
import json
import uuid

def parse_tool_call(output: dict) -> dict:
    """
    Parses tool call blocks from the LLM output, extracts tool call information,
    and restructures the output dictionary for downstream processing.

    Args:
        output (dict): The LLM response dictionary containing a 'response' key
                       with tool call blocks and other metadata.

    Returns:
        dict: The processed output dictionary with reordered keys and a 'message'
              key containing the assistant's response and tool call details.

    Steps:
        - Uses regex to find tool call blocks (<tool_call>, <tools>, <tool>).
        - Parses each block as JSON and constructs a tool call dictionary.
        - Cleans the original response by removing tool call blocks.
        - Adds the assistant message and tool calls to the output.
        - Reorders keys for consistency.
    """
    # Extract tool calls using regex
    matches = re.findall(r"<tool_call>(.*?)</tool_call>|<tools>(.*?)</tools>|<tool>(.*?)</tool>", output["response"], re.DOTALL)
    matches = [block for match in matches for block in match if block.strip()]
    tool_calls = []
    if matches:
        
        # Parse each matched block as JSON
        for index, block in enumerate(matches):
            block = block.strip()
            try:
                tool_dict = json.loads(block)
                tool_dict = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "index": index,
                        "name": tool_dict.get("name"),
                        "arguments": tool_dict.get("arguments", {})
                    }
                }
                tool_calls.append(tool_dict)
            except json.JSONDecodeError as e:
                print("Invalid JSON:", e)

        # Clean the original response by removing the tool call blocks
        result = output["response"]

        for a in matches:
            result = result.replace("<tool_call>" + a + "</tool_call>", "")

        result = " ".join(result.split())

        # Add the final assistant message with the tool calls
        output["message"] = {"role": "assistant", "content": result, "thinking": output.get("thinking", ""), "tool_calls": tool_calls}

        # Remove any keys that are not needed for the assistant message
        output.pop("response", None)
        output.pop("thinking", None)
    else:
        # Add the final assistant message without tool calls
        output["message"] = {"role": "assistant", "content": output.get("response", ""), "thinking": output.get("thinking", "")}

        # Remove any keys that are not needed for the assistant message
        output.pop("response", None)
        output.pop("thinking", None)
    
    # Reorder keys in the output dictionary
    order = [
        "model", "created_at", "message", "done", "done_reason",
        "total_duration", "load_duration", "prompt_eval_count",
        "prompt_eval_duration"
    ]

    # First: ordered keys
    ordered_part = {k: output[k] for k in order if k in output}

    # Second: remaining keys
    remaining_part = {k: v for k, v in output.items() if k not in order}

    # Merge them
    output = {**ordered_part, **remaining_part}
    return output

if __name__ == "__main__":
    # Example usage
    response = {
    "model": "qwen3:1.7b",
    "created_at": "2026-02-25T02:38:58.000591401Z",
    "response": "<tool_call>\n{\"name\": \"Structured_Data_Query_Tool\", \"arguments\": {\"table\": \"sla_lookup\", \"filters\": {\"service_name\": \"Premium Support\"}}}\n</tool_call>\n<tool_call>\n{\"name\": \"Structured_Data_Query_Tool\", \"arguments\": {\"table\": \"accounts\", \"filters\": {\"user_id\": \"1001\"}}}\n</tool_call>",
    "thinking": "<think>\nOkay, let's see. The user is asking two questions here. First, they want to know the SLA for Premium Support. Then, they want the account name for user ID 1001.\n\nLooking at the tools available, there's the Structured_Data_Query_Tool. This tool can execute SELECT queries on the database. The database has a table called sla_lookup, which probably contains the SLA information. The parameters for this tool include the table name and filters. \n\nFor the first question about the SLA for Premium Support, I need to check the sla_lookup table. The filters would likely be something like {\"service_name\": \"Premium Support\"}. The tool will return the relevant rows, and I can extract the response time and resolution time from those.\n\nNext, the user wants the account name for user ID 1001. That would be querying the accounts table. The filters here would be {\"user_id\": \"1001\"}. The tool will return the row with the user's name, which is the account name.\n\nI need to make two separate function calls. First, for the SLA, then for the account name. Each call uses the Structured_Data_Query_Tool with the appropriate table and filters. I should ensure the parameters are correctly formatted as JSON objects. Also, check if the required parameters are present, like table and filters. Both are required, so I need to include them in each call.\n\nWait, the user might be expecting both answers in one go, but since the tools are separate, I have to handle each query individually. So, first call for SLA, then another for the account name. Make sure the JSON arguments are correctly structured with the table name and filters. Also, check the datatypes for the filters, but the tool should handle validation if needed.\n",
    "done": True,
    "done_reason": "stop",
    "total_duration": 62332011403,
    "load_duration": 380566225,
    "prompt_eval_count": 1000,
    "prompt_eval_duration": 23252154986,
    "eval_count": 448,
    "eval_duration": 38191032883
    }
    
    response = parse_tool_call(response)
    print(response)