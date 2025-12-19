from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from RAG_agent.Rag_agent import Rag_agent
import chromadb
from chromadb.config import Settings
from utils.asyn_call import asyn_call
import asyncio

# Configure ChromaDB persistent client ## will be using psql for session storage
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",  # Directory where ChromaDB will persist data
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Create or get collection
collection_name = "rag_agent_sessions"
try:
    collection = chroma_client.get_collection(name=collection_name)
except:
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "RAG agent conversation sessions"}
    )

# Initialize DatabasePersistentSession with ChromaDB
session = DatabaseSessionService(
    collection=collection,
    session_id="default_session"  # You can make this dynamic based on user/conversation
)


Initial_state = {
    "name": Ayaan,
    "query":""
}

async def asyn_main():
    APP_NAME="RAG Agent with ChromaDB Persistent Session"
    USER_ID = "default_user"
    existing_session = await session.get_session(app_name=APP_NAME,user_id=USER_ID)
    if existing_session:
        print("users exists!")
        SESSION_ID = existing_session.sessions[0].id
        print("Session ID:",SESSION_ID)
    else:
        new_session = await session.create_session(app_name=APP_NAME,user_id=USER_ID,state=Initial_state)
        print("new session created!")
        SESSION_ID = new_session.id
        print("Session ID:",SESSION_ID)
    # Create and run the agent with persistent session
    runner = Runner(
        agent=Rag_agent,
        app_name=APP_NAME,
        session_service=session
    ) 

    await asyn_call(runner,APP_NAME,USER_ID,SESSION_ID)
        
        
       


if __name__ == "__main__":
    # Run the agent
    print("RAG Agent with ChromaDB Persistent Session")
    print("=" * 50)
    
    asyncio.run(asyn_main())
