# Copyright (c) Microsoft. All rights reserved.
# https://devblogs.microsoft.com/semantic-kernel/using-openais-o3-mini-reasoning-model-in-semantic-kernel/

import asyncio
import os


from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory

"""
# Reasoning Models Sample

This sample demonstrates an example of how to use reasoning models such as OpenAI’s o1 and o1-mini for inference.
Reasoning models currently have certain limitations, which are outlined below.

1. Requires API version `2024-09-01-preview` or later.
  - `reasoning_effort` and `developer_message` are only supported in API version `2024-12-01-preview` or later.
  - o1-mini is not supported property `developer_message` `reasoning_effort` now.
2. Developer message must be used instead of system message
3. Parallel tool invocation is currently not supported
4. Token limit settings need to consider both reasoning and completion tokens

# Unsupported Properties ⛔

The following parameters are currently not supported:
- temperature
- top_p
- presence_penalty
- frequency_penalty
- logprobs
- top_logprobs
- logit_bias
- max_tokens
- stream
- tool_choice

# .env examples

OpenAI: semantic_kernel/connectors/ai/open_ai/settings/open_ai_settings.py

```.env
OPENAI_API_KEY=*******************
OPENAI_CHAT_MODEL_ID=o1-2024-12-17
```

Azure OpenAI: semantic_kernel/connectors/ai/open_ai/settings/azure_open_ai_settings.py

```.env
AZURE_OPENAI_API_KEY=*******************
AZURE_OPENAI_ENDPOINT=https://*********.openai.azure.com
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=o1-2024-12-17
AZURE_OPENAI_API_VERSION="2024-12-01-preview"
```

Note: Unsupported features may be added in future updates.
"""

load_dotenv()

# The envionrment variables needed to connect to the gpt-4o model in Azure AI Foundry
deployment_name = "o3"
endpoint = os.environ["CHAT_MODEL_ENDPOINT"]
api_key = os.environ["CHAT_MODEL_API_KEY"]
service_id = "reasoning"

# chat_service = OpenAIChatCompletion(service_id="reasoning", deployment_name=deployment_name, endpoint=endpoint, api_key=api_key)
chat_service = AzureChatCompletion(service_id=service_id, deployment_name=deployment_name, endpoint=endpoint, api_key=api_key)
settings = OpenAIChatPromptExecutionSettings(
    max_tokens=8000,  # Set a high token limit for detailed responses
    # reasoning_effort="high" # applicable only for o3-mini models
)

# Create a ChatHistory object
chat_history = ChatHistory()

# This is the system message that gives the chatbot its personality.
developer_message = """
As an assistant supporting the user,
you recognize all user input
as questions or consultations and answer them.
"""
# The developer message was newly introduced for reasoning models such as OpenAI’s o1 and o1-mini.
# `system message` cannot be used with reasoning models.
chat_history.add_developer_message(developer_message)


async def chat() -> bool:
    try:
        user_input = input("User:> ")
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False

    chat_history.add_user_message(user_input)

    # Get the chat message content from the chat completion service.
    response = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=settings
    )
    if response:
        print(f"{deployment_name} reply:", response)

        # Add the chat message to the chat history to keep track of the conversation.
        chat_history.add_message(response)

    return True


async def main() -> None:
    # Start the chat loop. The chat loop will continue until the user types "exit".
    chatting = True
    while chatting:
        chatting = await chat()

    # Sample output:
    # User:> Why is the sky blue in one sentence?
    # Reasoning model:> The sky appears blue because air molecules in the atmosphere scatter shorter-wavelength (blue)
    #           light more efficiently than longer-wavelength (red) light.


if __name__ == "__main__":
    asyncio.run(main())