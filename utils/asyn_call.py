import asyncio
from google.adk.runner import Runner
from google.genai import types




def process_response(event):




    return response



async def asyn_call(session,APP_NAME,USER_ID):
    SESSION_ID = session.get_session(app_name=APP_NAME,user_id=USER_ID).sessions[0].id
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break

        content = types.Content(role = "user", parts = [types.Part(text = user_input)])

        try:
            async for event in runner.async_run(
                user_id = USER_ID,
                session_id = SESSION_ID,
                app_name = APP_NAME,
                content = content
            ):

                response = await process_response(event)            
        # Run the agent with user input
        except Exception as e:
            print(f"Error: {e}")
            break

        print(f"\nAgent: {response}")
