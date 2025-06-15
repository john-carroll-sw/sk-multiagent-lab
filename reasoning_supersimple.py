import asyncio
import os
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory

load_dotenv()

async def main():
    # The envionrment variables needed to connect to the gpt-4o model in Azure AI Foundry
    deployment_name = "o3"
    endpoint = os.environ["CHAT_MODEL_ENDPOINT"]
    api_key = os.environ["CHAT_MODEL_API_KEY"]
    service_id = "reasoning"

    # Initialize the OpenAI chat completion with reasoning model
    # Assumes OPENAI_API_KEY and OPENAI_ORG_ID are in your environment variables
    # chat_service = OpenAIChatCompletion(ai_model_id="o3-mini")

    chat_service = AzureChatCompletion(service_id=service_id, deployment_name=deployment_name, endpoint=endpoint, api_key=api_key)

    # Start a chat history with the system prompt/instruction
    chat_history = ChatHistory(system_message="You are a helpful assistant.")
    chat_history.add_user_message("Why is the sky blue in one sentence?")

    # Create settings and set high reasoning effort for a more detailed response
    settings = OpenAIChatPromptExecutionSettings(
        # max_tokens=8000,  # Set a high token limit for detailed responses
        # reasoning_effort="high" # applicable only for o3-mini models
    )

    # Get the model's response
    response = await chat_service.get_chat_message_content(chat_history, settings=settings)
    print(f"{deployment_name} reply:", response)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
