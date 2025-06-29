import os
import chainlit as cl
from typing import cast
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    RunConfig,
    set_default_openai_client,
    AsyncOpenAI,
    set_tracing_disabled,
    set_default_openai_api,
)
from my_secrets import MySecrets

load_dotenv()

secrets = MySecrets()


def setup_config():
    external_client = AsyncOpenAI(
        api_key=secrets.gemini_api_key,
        base_url=secrets.gemini_api_url,
    )

    set_default_openai_client(external_client)
    set_tracing_disabled(True)
    set_default_openai_api("chat_completions")

    # Setup Agents
    spanish_agent = Agent(
        name="spanish_agent",
        instructions="You translate the user's message to Spanish",
        model=secrets.gemini_api_model,
    )

    french_agent = Agent(
        name="french_agent",
        instructions="You translate the user's message to French",
        model=secrets.gemini_api_model,
    )

    italian_agent = Agent(
        name="italian_agent",
        instructions="You translate the user's message to Italian",
        model=secrets.gemini_api_model,
    )

    # triage Agent
    triage_agent = Agent(
        name="triage_agent",
        instructions=(
            "You are a translation agent. You use the tools given to you to translate."
            "If asked for multiple translations, you call the relevant tools in order."
            "You never translate on your own, you always use the provided tools." 
            "Also mention which tool you used for translation."
        ),
        tools=[
            spanish_agent.as_tool(
                tool_name="translate_to_spanish",
                tool_description="Translate the user's message to Spanish",
            ),
            french_agent.as_tool(
                tool_name="translate_to_french",
                tool_description="Translate the user's message to French",
            ),
            italian_agent.as_tool(
                tool_name="translate_to_italian",
                tool_description="Translate the user's message to Italian",
            ),
        ],
        model=secrets.gemini_api_model,
    )

    return triage_agent


@cl.on_chat_start
async def start():
    triage_agent = setup_config()
    cl.user_session.set("triage_agent", triage_agent)
    cl.user_session.set("chat_history", [])
    await cl.Message(content="Welcome to the Panaversity AI Assistant!").send()


@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    # Send a thinking message
    msg = cl.Message(content="")
    await msg.send()

    # Retrieve the AI agent from the session
    triage_agent = cast(Agent, cl.user_session.get("agent"))

    # Get the chat history from the session
    chat_history: list = cl.user_session.get("chat_history") or []

    # Add the current user message to the history
    chat_history.append({"role": "user", "content": message.content})

    # Run the agent synchronously with the chat history as input
    result = Runner.run_streamed(triage_agent, input=chat_history)
    
    async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, 'delta'):
                token = event.data.delta
                await msg.stream_token(token)

    response_content = result.final_output

    chat_history.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("chat_history", chat_history)
    await msg.update()

    # Update the thinking message with the actual response
    # msg.content = response_content
    # await msg.update()

    # history.append({"role": "assistant", "content": response_content})

    # cl.user_session.set("chat_history", history)

    # print(f"History: {history}")