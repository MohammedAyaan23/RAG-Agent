from google.adk.agent import Agent



root_agent = Agent(
    name="rag_agent",
    model="gemini-2.5-flash",
    description="An agent whcih takes the user's query and returns the answer",
    instruction="""You are a helpful agent who can use following tools:
    - google_search tool""",
    tools=[google_search],
)