from src.agent.agent import agent, agent_loop
from fastapi import FastAPI, Form, Request, Depends
import json
import shutil
from datetime import datetime
import src.tools.tool1
from src.tools.prompt_guard import prompt_guard
from src.tools.tool2 import External_API_Simulation_Tool
from src.tools.tool1 import Structured_Data_Query_Tool
import requests
import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import logging

# Load environment variables
load_dotenv()

# FastAPI
app = FastAPI(docs_url="/")



@app.on_event("startup")
def startup():
    # Clear log file
    open("src/logging/agent.log", "w").close()
    
    # Construct the database URL from environment variables
    DB_URL = (
        f"postgresql://{os.getenv('DB_USER')}:"
        f"{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:"
        f"{os.getenv('DB_PORT')}/"
        f"{os.getenv('DB_NAME')}"
    )
    # Initialize the connection pool
    src.tools.tool1.pool = ConnectionPool(DB_URL)

    # Initialize logging config
    logging.config.fileConfig('src/agent.conf', disable_existing_loggers=False)

    # initialize logger
    src.agent.agent.logger = logging.getLogger("Agent")
    src.tools.tool3.logger = logging.getLogger("Guardrail")

@app.on_event("shutdown")
def shutdown():
    src.tools.tool1.pool.close()
    # Save log file with current date
    log_file = "src/logging/agent.log"
    date_str = datetime.now().strftime("%Y-%m-%d")
    backup_file = f"src/logging/agent_{date_str}.log"
    
    try:
        # Check if log file exists and has content
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            # If backup already exists, append to it
            if os.path.exists(backup_file):
                with open(log_file, 'r') as source, open(backup_file, 'a') as dest:
                    dest.write(source.read())
            else:
                # Copy to new dated file
                shutil.copy2(log_file, backup_file)
            print(f"Log saved to: {backup_file}")
    except Exception as e:
        print(f"Error saving log: {e}")


@app.post("/ollama/generate")

async def ollama(
    model: str = Form("qwen3:1.7b"),
    question: str = Form(...),
    thinking: bool = Form(False),
    manual: bool = Form(True)
):
    # Prompt Guard - Detect prompt attack like Prompt Injections and Jailbreaks
    try:
        if prompt_guard(question) == "LABEL_1":
            return "Sorry i cannot do that."
    except Exception as e:
        return "Sorry there problem with prompt checking."

    try:
        # Check if model is available
        list_model = requests.get("http://localhost:11434/api/tags").json()["models"]
        list_model = [m["name"] for m in list_model]
        
        if model not in list_model:
            return f"Model '{model}' not avaliable. Available models: {', '.join(list_model)}"
        
        # model capability check
        model_cabilities = requests.post("http://localhost:11434/api/show", json={"model": model}).json()["capabilities"]
        
        # set think flag based on capability
        if "thinking" not in model_cabilities:
            if thinking:
                return f"Model '{model}' does not support thinking. Model capabilities: {', '.join(model_cabilities)}"
            is_think_set = False
        elif "tools" not in model_cabilities:
            return f"Model '{model}' does not support tool calling. Model capabilities: {', '.join(model_cabilities)}"
        else:
            is_think_set = True
    except requests.exceptions.ConnectionError as e:
        return f"Cannot connect to Ollama server at localhost:11434. Please ensure Ollama is running. Error: {e}"
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama server: {e}"
    
    # Ollama generate endpoint
    url = "http://localhost:11434/api/generate"
    
    # Parameters for agent loop
    max_retries = 4

    # Prepare messages and tools for the prompt
    system = (
        "If the question contains harmful, biased, or inappropriate content; refuse to answer"
        "When a tool provides information, you MUST use the tool result to answer. "
        "Execute tools sequentially. If a tool needs data from another tool, call the first tool first. Stop generating text after the tool call. Do not call the second tool until you receive the first tool's result."
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

    available_functions = {
    'Structured_Data_Query_Tool': Structured_Data_Query_Tool,
    'External_API_Simulation_Tool': External_API_Simulation_Tool,
    }

    #fword = ["delete","bypass"]
    # Agent loop to handle tool calls until completion
    try:
        print(f"\n*** Agentic AI Started ***")
        result = agent_loop(max_retries, url, model, messages, tools, system, is_think_set, thinking, available_functions, manual)
        return result
    except Exception as e:
        return f"Sorry i incapable answering that question, Error Messages :{e}"

@app.post("/ollama/chat")

async def ollama(
    model: str = Form("qwen3:1.7b"),
    question: str = Form(...),
    thinking: bool = Form(False),
    ollama_api_key: str = Form(None),
    manual: bool = Form(True)
):
    
    if model.split(':', 1)[-1] == "cloud":
        # Set the environment variable
        os.environ['OLLAMA_API_KEY'] = ollama_api_key
        # Ollama generate endpoint
        url = "https://ollama.com/api/chat"
    else:
        url = "http://localhost:11434/api/chat"

    try:
        # Check if model is available
        list_model = requests.get("http://localhost:11434/api/tags").json()["models"]
        list_model = [m["name"] for m in list_model]
        
        if model not in list_model:
            return f"Model '{model}' not avaliable. Available models: {', '.join(list_model)}"
        
        # model capability check
        model_cabilities = requests.post("http://localhost:11434/api/show", json={"model": model}).json()["capabilities"]
        
        # set think flag based on capability
        if "thinking" not in model_cabilities:
            if thinking:
                return f"Model '{model}' does not support thinking. Model capabilities: {', '.join(model_cabilities)}"
            is_think_set = False
        elif "tools" not in model_cabilities:
            return f"Model '{model}' does not support tool calling. Model capabilities: {', '.join(model_cabilities)}"
        else:
            is_think_set = True
    except requests.exceptions.ConnectionError as e:
        return f"Cannot connect to Ollama server at localhost:11434. Please ensure Ollama is running. Error: {e}"
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama server: {e}"
    
    # Prompt Guard - Detect prompt attack like Prompt Injections and Jailbreaks
    try:
        if prompt_guard(question) == "LABEL_1":
            return "Sorry i cannot do that."
    except Exception as e:
        return "Sorry there problem with prompt checking."
    
    # Parameters for agent loop
    max_retries = 4

    # Prepare messages and tools for the prompt
    system = (
        "If the question contains harmful, biased, or inappropriate content; refuse to answer"
        "When a tool provides information, you MUST use the tool result to answer. "
        "Execute tools sequentially. If a tool needs data from another tool, call the first tool first. Stop generating text after the tool call. Do not call the second tool until you receive the first tool's result."
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

    available_functions = {
    'Structured_Data_Query_Tool': Structured_Data_Query_Tool,
    'External_API_Simulation_Tool': External_API_Simulation_Tool,
    }

    try:
        print(f"\n*** Agentic AI Started ***")
        result = agent_loop(max_retries, url, model, messages, tools, system, is_think_set, thinking, available_functions, manual)
        return result
    except Exception as e:
        return f"Sorry i incapable answering that question, Error Messages :{e}"