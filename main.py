import chainlit as cl
from my_secrets import MySecrets
from typing import cast
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    set_tracing_disabled,
    function_tool,
    OpenAIChatCompletionsModel,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    input_guardrail,
    output_guardrail,
)
from pydantic import BaseModel

secrets = MySecrets()
set_tracing_disabled(True)

def setup_config():
    external_client = AsyncOpenAI(
        api_key=secrets.gemini_api_key,
        base_url=secrets.gemini_api_url,
    )
    external_client1 = AsyncOpenAI(
        api_key=secrets.deepseek_api_key,
        base_url=secrets.deepseek_api_url,
    ) 
      
    class Validator(BaseModel):  
        is_foul: bool  
        reasoning: str  

    guardrail_agent = Agent(  
        name="Guardrail Check",  
        instructions="Detect if the user's reflection contains foul language.",  
        output_type=Validator,  
        model=OpenAIChatCompletionsModel( 
        model=secrets.gemini_api_model,
        openai_client= external_client,
        ),  
    )  

    @input_guardrail  
    async def reflection_input_guardrail(  
        ctx: RunContextWrapper[None],  
        agent: Agent,  
        user_input: str  
    ) -> GuardrailFunctionOutput:  
        result = await Runner.run(guardrail_agent, user_input, context=ctx.context)  
        return GuardrailFunctionOutput(  
            output_info=result.final_output,  
            tripwire_triggered=result.final_output.is_foul,  
        )  

    summarizer = Agent(  
        name="summarizer",  
        instructions="Summarize the reflection into three concise key takeaways.",
        model=OpenAIChatCompletionsModel( 
        model=secrets.deepseek_api_model,
        openai_client= external_client1
        ),
    )  

    @output_guardrail  
    async def coach_output_guardrail(  
        ctx: RunContextWrapper[None],  
        agent: Agent,  
        coach_output: str  
    ) -> GuardrailFunctionOutput:  
        result = await Runner.run(guardrail_agent, coach_output, context=ctx.context)  
        return GuardrailFunctionOutput(  
            output_info=result.final_output,  
            tripwire_triggered=result.final_output.is_foul,  
        )  

    coach = Agent(  
        name="Coach",  
        instructions="Offer multiple mindfulness tips or exercises based on the summary.",   
        output_guardrails=[coach_output_guardrail],  
        model=OpenAIChatCompletionsModel( 
        model=secrets.gemini_api_model,
        openai_client= external_client,
        ),
    )  

    triage = Agent(  
        name="Triage",  
        instructions=(   
            "Take the user's reflection, use the Summarizer tool to get a summary, "  
            "then hand off to the Coach to provide suggestions based on the summary."  
            "Whenever there is a tool call, provide its name. When there is a handoff, provide the handoff agent name."  
        ),  
        handoffs=[coach],  
        tools=[  
            summarizer.as_tool(  
                tool_name="Summarizer",  
                tool_description="Summarize the reflection into three concise key takeaways.")  
        ],  
        input_guardrails=[reflection_input_guardrail],  
        model=OpenAIChatCompletionsModel( 
        model=secrets.gemini_api_model,
        openai_client= external_client,
        ),  
    )  
    return triage

@cl.on_chat_start
async def start():
    triage_agent = setup_config()
    cl.user_session.set("triage_agent", triage_agent)
    cl.user_session.set("chat_history", [])
    await cl.Message(content="Hi! Welcome to Reflect Coach. Your Daily Reflection Coach!. How was day today?").send()

@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    msg = cl.Message(content="Thinking...")
    await msg.send()
    triage_agent = cast(Agent, cl.user_session.get("triage_agent"))
    user_input = message.content
    try:
        result = Runner.run_streamed(triage_agent, user_input)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                token = event.data.delta
                await msg.stream_token(token)
                # print(event.data.delta, end="", flush=True)
    except Exception as e:
        error_msg = cl.Message(content=f"Your reflection couldn't be processed: {str(e)}")
        await error_msg.send()

if __name__ == "__main__":
    asyncio.run(main())        