import asyncio
import logging
from typing import Optional

from agents.router_agent import route_query
from agents.state_manager import get_initial_state, next_state
from retriever.vector_retriever import VectorRetriever
from models.answer_generator import AnswerGenerator
from memory.session_manager import (
    get_or_create_session,
    append_event,
    append_state,
    load_last_state,
)


# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -------------------------
# Configuration
# -------------------------

COLLECTION_NAME = "my_documents_personal_project"


# ------------------------
    # chunks
# ------------------------
base_chunks = {
            "factual": 3,
            "explanation": 8,
            "comparison": 12,
            "creative": 6,
        }

# -------------------------
# Main async handler
# -------------------------

async def handle_query(
    user_query: str,
    session_id: Optional[str] = None,
):
    """
    Orchestrates:
    Router → Planner → Retriever → Answer Generator (streaming)
    """
    logger.info(f"Handling query: {user_query}")

    # 1. Session handling
    session_id = get_or_create_session(session_id)
    logger.info(f"Session ID: {session_id}")

    # 2. Load conversation state
    state = get_initial_state()
    logger.info(f"Conversation State: {state}")

    # 3. Persist user message
    append_event(session_id, "user", user_query)

    # 4. Route query (planner selection happens inside)
    plan = await route_query(
        query=user_query,
        session_id=session_id,
        state=state,
    )
    logger.info(f"Router Result: {plan}")

    # The router returns the planner output directly
    # (deterministic planner returns a dict)
    planner_name = plan.get("planner", "unknown")
    append_event(session_id, "planner", planner_name)

    # 5. Update conversation state
    state = next_state(
    current_dict=state,
    intent=plan.get("intent"),
    comparison=plan.get("intent") == "comparison"
    )
    append_state(session_id, state)
    logger.info(f"New Conversation State: {state}")

    # 6. Retrieve context, based on the path or planner
    if planner_name == "deterministic":
        retriever = VectorRetriever(
        collection_name=COLLECTION_NAME
        )

        context_chunks = retriever.retrieve(
        query=plan["query"],
        n_chunks=plan["recommended_chunks"],
        )
    elif planner_name == "pseudo_deterministic":
        context_chunks = []
        retriever = VectorRetriever(
        collection_name=COLLECTION_NAME
        )
        for intent, score in plan["intent_scores"].items():
            sub_query = plan["sub_queries"][f"{intent}_query"]
            chunks = max(3,int(base_chunks.get(intent, 5)*score))
            sub_query_chunks = retriever.retrieve(
            query=sub_query,
            n_chunks=chunks,
            )
            context_chunks.extend(sub_query_chunks)
    else:
        return {
            "error": "Invalid planner name"
        }
    
    
    logger.info(f"Retrieved {len(context_chunks)} chunks for generation.")

    # 7. Generate answer (streaming)
    generator = AnswerGenerator()

    print("\n" + "=" * 20 + " AI RESPONSE " + "=" * 20 + "\n")

    full_response = ""

    async for token in generator.generate(
        query=plan["query"],
        context_chunks=context_chunks,
    ):
        print(token, end="", flush=True)
        full_response += token

    print("\n" + "=" * 56 + "\n")

    # 8. Persist assistant response
    append_event(session_id, "assistant", full_response)
    
    logger.info("Response generated and persisted.")

    return {
        "session_id": session_id,
        "planner": planner_name,
        "state": state.get("state", "start"),
        "response": full_response,
    }


# -------------------------
# CLI entry point
# -------------------------

async def main():
    print("RAG Agent (Deterministic Path)")
    print("=" * 50)

    session_id = None

    while True:
        user_query = input("\nAsk a question (or type 'exit'): ").strip()
        if user_query.lower() == "exit":
            break

        result = await handle_query(
            user_query=user_query,
            session_id=session_id,
        )

        # Reuse session for follow-ups
        logger.info(f"Session ID: {result}")
        session_id = result["session_id"]


# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("\nShutting down gracefully.")

