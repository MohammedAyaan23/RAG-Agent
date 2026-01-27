import asyncio
import json
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.apps.app import App, EventsCompactionConfig
from dotenv import load_dotenv, find_dotenv

# Import your planners
from planner.deterministic_planner import DeterministicPlanner
from planner.pseudo_deterministic_planner import PseudoDeterministicPlanner

load_dotenv(find_dotenv())

deterministic_func = DeterministicPlanner()
pseudo_func = PseudoDeterministicPlanner()

# --- TOOL DEFINITIONS ---

async def deterministic_tool(query: str, tool_context: ToolContext):
    """
    Processes a user query through the deterministic planner. 
    
    Args:
        query: The raw text of the user's request to be routed.
    """
    result = await deterministic_func.run(query)
    # The ADK injects the ToolContext object here automatically
    tool_context.actions.skip_summarization = True
    return result

async def pseudo_deterministic_tool(query: str, tool_context: ToolContext):
    """
    Processes a user query through the pseudo-deterministic planner. 
    
    Args:
        query: The raw text of the user's request to be routed.
    """
    result = await pseudo_func.run(query)
    tool_context.actions.skip_summarization = True
    return result

# Now initialize without the 'declaration' argument
det_tool = FunctionTool(deterministic_tool)
pseudo_tool = FunctionTool(pseudo_deterministic_tool)
# --- AGENT SETUP ---

groq_model = LiteLlm(
    model="groq/llama-3.3-70b-versatile",
    max_tokens=100, # Increased slightly to ensure full tool-call JSON fits
    temperature=0.0
)

main_agent = Agent(
    name="main_agent",
    model=groq_model,
    description="A routing agent that classifies queries.",
    instruction="""
    Identify if the query is a simple request or complex intent.
    - Use 'deterministic_tool' for simple/greetings/short queries.
    - Use 'pseudo_deterministic_tool' for complex queries.
    - Do NOT provide text answers yourself; always use a tool.
    """,
    tools=[det_tool,pseudo_tool],
)

session_service = InMemorySessionService()
router_app = App(
    name="router_agent",
    root_agent=main_agent,
    events_compaction_config=EventsCompactionConfig(compaction_interval=3, overlap_size=1)
)
runner = Runner(app=router_app, session_service=session_service)

# --- ROUTING LOGIC ---

async def route_query(query: str, session_id: str, state=None):
    print("\n" + "=" * 50 + " ROUTER AGENT " + "=" * 50 + "\n")

    print(f"session ID: {session_id}")
    try:
        print(f"Trying to get session {session_id}")
        existing_session = await session_service.get_session(
            app_name="router_agent",
            user_id=session_id,
            session_id=session_id
        )
        print(f"Session {session_id} found")
    except Exception:
        # Depending on ADK version, get_session might raise an error if not found
        print(f"Failed to get session {session_id}")
        existing_session = None

    # 2. Only create the session if it doesn't exist
    if not existing_session:
        print(f"Session {session_id} not found, creating new session")
        state_dict = state if isinstance(state, dict) else {"status": state or "start"}
        await session_service.create_session(
            app_name="router_agent",
            user_id=session_id,
            session_id=session_id,
            state=state_dict
        )
        print(f"Created new session for {session_id}")
    else:
        print(f"Resuming existing session for {session_id}")

    new_user_message = types.Content(
        role="user",
        parts=[types.Part(text=query)]
    )
    
    final_output = None
    print(f"Running the agent...")

    async for event in runner.run_async(
        user_id=session_id,
        session_id=session_id,
        new_message=new_user_message
    ):
        # 1. Capture the actual data from the tool (Dict)
        # Because skip_summarization=True, the data stops here
        tool_responses = event.get_function_responses()
        if tool_responses:
            final_output = tool_responses[0].response
            print(f"Captured tool response: {final_output}")

        # 2. Fallback: If the model refuses to use a tool and just chats
        # if event.is_final_response() and final_output is None:
        #     final_output = event.content.parts[0].text
            
    return final_output