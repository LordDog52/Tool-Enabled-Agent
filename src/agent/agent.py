import os
from dotenv import load_dotenv
import requests
import json
import logging
from src.agent.prompt_template import render_prompt, tools_prompt
from src.agent.parser import parse_tool_call
from src.tools.tool3 import Guardrail_Evaluation_Tool
from src.agent.manual_decision import manual_decision

# Load environment variables
load_dotenv()

def agent(url, model, messages, tools, system, is_think_set, thinking):
    """
    Sends a prompt to the LLM API, receives the response, and parses tool calls.

    Args:
        url (str): The LLM API endpoint.
        model (str): Model name or identifier.
        messages (list): Conversation history/messages.
        tools (list): List of available tools/functions.
        system (str): System prompt or instructions.
        is_think_set (bool): Whether 'think' mode is enabled.
        thinking (bool): Current thinking state.

    Returns:
        dict: Parsed response from the LLM, including tool calls if present.

    Raises:
        ValueError: If the API response status is not 200.
    """

    endpoint = url.rsplit('/', 1)[-1]
    
    if endpoint == "generate":
        # For generate endpoint, we need to add "raw": True to the request body
        # This tells Ollama to not apply chat template

        # Prompt Template
        prompt = render_prompt(messages, tools=tools, system=system, is_think_set=is_think_set, think=thinking)
        logger.debug("LLM Prompt input:\n"+prompt)

        # API request payload
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                
            },
            "raw": True
        }
    else:
        # For other endpoints, construct the payload differently
        # Prompt Template
        if system:
            system = system + "/n" + tools_prompt(tools)
            messages.append({"role": "system", "content": system})
            logger.debug("LLM System instruction:\n"+system)
        else:
            system = tools_prompt(tools)
            messages.append({"role": "system", "content": system})
            logger.debug("LLM System instruction:\n"+system)

        # API request payload
        data = {
            "model": model,
            "messages": messages,
            "think": thinking,
            "stream": False,
            "options": {
                "temperature": 0,
            }
        }

    # Ollama API request
    try:
        # Send Request to Ollama API
        if url == "https://ollama.com/api/chat":
            headers = {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
            response = requests.post(url, data=json.dumps(data, default=str), headers=headers, timeout=200)
        else:
            response = requests.post(url, data=json.dumps(data, default=str), timeout=200)
        
        # LLM API Response
        result = response.json()
        response.raise_for_status()  # Raises HTTPError for 4xx/5xx
        logger.debug("LLM Raw Response:\n"+str(result))
        if endpoint == "generate":
            return parse_tool_call(result)
        else:
            result["response"] = result["message"]["content"]
            return parse_tool_call(result)
    except requests.Timeout:
        logger.error("Request to LLM API timed out")
        raise RuntimeError("Failed to get response from LLM: Request timed out") from None
    except requests.RequestException as e:
        if response.status_code == 401:
            logger.error("Unauthorized access to LLM API - check your API key")
            raise RuntimeError("Unauthorized access (check API key)") from None
        logger.error(f"Request to LLM API failed: {e}")
        raise Exception(f"Error: {str(e)}")

def agent_loop(max_retries, url, model, messages, tools, system, is_think_set, thinking, available_functions, manual, fword = ["delete","bypass"]):
    """
    Runs the agent in a loop, handling tool calls and guardrail evaluation.

    Args:
        max_retries (int): Maximum number of retries for tool calls.
        url (str): The LLM API endpoint.
        model (str): Model name or identifier.
        messages (list): Conversation history/messages.
        tools (list): List of available tools/functions.
        system (str): System prompt or instructions.
        is_think_set (bool): Whether 'think' mode is enabled.
        thinking (bool): Current thinking state.
        available_functions (dict): Mapping of tool names to callable functions.
        manual (bool): Whether to enable manual decision making.
        fword (list): list of forbidden word in function argument

    Returns:
        str: Final agent response or error message after guardrail evaluation or max retries.
    """
    attempt = 0
    while True:
        # Attempt increment
        attempt += 1

        # Max retry
        if attempt > max_retries:
            logger.warning("Max retries reached. Tool failed repeatedly.")
            return "Max retries reached. Tool failed repeatedly."

        # Print iteration
        print(f"\n=== Attempt {attempt}/{max_retries} ===")
        
        if attempt == 1 and manual == True:

            # Attempt to make a manual decision
            response = manual_decision(messages)


            if response != None: # manual decision is made
                messages.append(response)
            else:
                # Call the agent
                try:    
                    response = agent(url, model, messages, tools, system, is_think_set, thinking)
                except RuntimeError as e:
                    return str(e)
                # add the agent response to the messages
                messages.append(response["message"])
        else:
            # Call the agent 
            try:   
                response = agent(url, model, messages, tools, system, is_think_set, thinking)
            except RuntimeError as e:
                return str(e)
            
            # add the agent response to the messages
            messages.append(response["message"])
        
        # Tool Calls
        if messages[-1].get("tool_calls"): 
            for tc in messages[-1]["tool_calls"]:
                if tc["function"]["name"] in available_functions:

                    # Function call
                    if any(x.lower() in str(tc["function"]["arguments"]).lower() for x in fword):
                        return "I cannot answer that."
                    try:
                        result = available_functions[tc["function"]["name"]](**tc["function"]["arguments"])
                    except (TypeError, ValueError) as e:
                        logger.debug(f"Invalid arguments for {tc['function']['name']}: {e}")
                        result = f"Invalid arguments: {e}"
                    except Exception as e:
                        logger.debug(f"Error calling {tc['function']['name']}: {e}")
                        result = f"Error: {type(e).__name__}: {e}"

                    messages.append({'role': 'tool', "tool_calls_id": tc["id"], 'function_name': tc["function"]["name"],'argument': tc["function"]["arguments"], 'content': str(result)})
                    for msg in messages:
                        logger.info(msg)
                else:
                    messages.append({'role': 'tool', "tool_calls_id": tc["id"], 'function_name': tc["function"]["name"],'argument': tc["function"]["arguments"], 'content': f"Function {tc["function"]["name"]} is not available."})
                    for msg in messages:
                        logger.info(msg)
        
        # Guardrail Evaluation
        else:
            # end the loop when there are no more tool calls
            for msg in messages:
                logger.info(msg)
            print(f"\n=== Guardrails ===")
            try:
                safe_label, category_label, refusal_label = Guardrail_Evaluation_Tool("Qwen/Qwen3Guard-Gen-0.6B", messages)
            except Exception as e:
                logger.error(f"Guardrail evaluation failed: {e}")
                return "Sorry therese problem with guardrails"
            if safe_label == "Controversial" and refusal_label == "No": # escalate to log as warning
                logger.warning("Warning: The content is controversial. Escalating to human review.")
                return response["message"]["content"]
            elif safe_label == "Unsafe" and refusal_label == "No": # escalate to log and refuse to answer
                logger.error(f"Error: The content is unsafe under category {category_label}. LLM not refuse to answer. Need reviewed by human.")
                return "I cannot answer that."
            elif safe_label == "Unsafe" and refusal_label == "Yes":
                return response["message"]["thinking"] + response["message"]["content"]
            else:
                return response["message"]["thinking"] + response["message"]["content"]
    
if __name__ == "__main__":
    logger = logging.getLogger("Agent")
        