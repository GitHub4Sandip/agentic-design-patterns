from pathlib import Path

import requests
import json
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file (e.g., OPENAI_API_KEY)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

model = "gpt-4.1"

# Initialize the OpenAI client using API key from .env (via load_dotenv)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fetch_temperature(lat: float, lon: float) -> float:
    """Gets the current temperature in Celsius for a given latitude and longitude."""
    logger.info(f"Fetching temperature for coordinates: ({lat}, {lon})")
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current": "temperature_2m"}
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()
    return data["current"]["temperature_2m"]

def retrieve_from_kb(question: str) -> dict:
    """Retrieves information from the Educative knowledge base."""
    """
        Loads the entire knowledge base from disk.
        This is a placeholder — no filtering, no logic.
        Just hands over the full contents.
    """
    logger.info(f"Calling Knowledge Base for question: '{question}'...")
    KB_FILE = (
            Path(__file__).parent
            / "resources"
            / "educative_kb.json"
    )
    with open(KB_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data

# master tools registry that the agent can use to route user queries to the appropriate tool
master_tool_registry = [
    {
        "type": "function",
        "function": {
            "name": "fetch_temperature",
            "description": "Return the current temperature (°C) for a given location by its coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "The latitude of the location."},
                    "lon": {"type": "number", "description": "The longitude of the location."}
                },
                "required": ["lat", "lon"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_from_kb",
            "description": "Answer questions about Educative courses and content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user's question about Educative."}
                },
                "required": ["question"]
            }
        }
    }
]

# A simple dispatcher to execute the correct Python function.
def execute_function_call(name: str, args: dict):
    if name == "fetch_temperature":
        return fetch_temperature(**args)
    elif name == "retrieve_from_kb":
        return retrieve_from_kb(**args)
    else:
        return f"Error: function {name} not found"


# run_agentic_router is the main function that takes a user query,
# prompts the LLM to deduce parameters and choose a tool, executes the tool if chosen, and generates a final response.
def run_agentic_router(user_query: str):
    logger.info(f"User Query: '{user_query}'")

    # We give the LLM a special instruction to encourage it to guess coordinates.
    system_prompt = (
        "You are a helpful assistant with access to tools. "
        "For the 'fetch_temperature' tool, if the user provides a location name "
        "but not coordinates, **use your own general knowledge to determine the latitude and "
        "longitude, then call the function with those deduced values.** "
        "If you are unsure or the location is ambiguous, ask the user for clarification."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    logger.info("Step 1: Asking LLM to deduce parameters and choose a tool...")
    first_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=master_tool_registry,
    )

    response_message = first_response.choices[0].message

    # This 'if' statement is the core of the routing logic.
    if response_message.tool_calls:
        # --- PATH A: The LLM chose a tool ---
        tool_name = response_message.tool_calls[0].function.name
        logger.info(f"Step 2: Model decided to use '{tool_name}' and deduced the arguments.")
        function_args = json.loads(response_message.tool_calls[0].function.arguments)
        logger.info(f"   > Deduced Arguments: {function_args}")

        messages.append(response_message)
        tool_output = execute_function_call(name=tool_name, args=function_args)

        messages.append({
            "role": "tool", "tool_call_id": response_message.tool_calls[0].id, "content": json.dumps(tool_output)
        })

        logger.info("Step 3: Generating a final response...")
        second_response = client.chat.completions.create(model=model, messages=messages)
        final_answer = second_response.choices[0].message.content
        logger.info(f"✅ Final Assistant Response: {final_answer}\n")

    else:
        # --- PATH B: The LLM did not choose a tool ---
        logger.info("Step 2: Model decided it could not use a tool.")
        final_answer = response_message.content
        logger.info(f"✅ Final Assistant Response: {final_answer}\n")

if __name__ == "__main__":
    # Example queries to test the routing logic.
    #run_agentic_router("What is the current temperature in New York City?")
    run_agentic_router("What Java Course Educative is providing?")