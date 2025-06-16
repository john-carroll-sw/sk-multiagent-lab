import os
from typing import Any
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import CodeInterpreterTool

load_dotenv() # Load environment variables from .env file

# Get the project connection string and model from environment variables, which are needed to make a call to the LLM
project_connection_string = os.getenv("AIPROJECT_CONNECTION_STRING")
model = os.getenv("CHAT_MODEL")

# Use the connection string to connect to your Foundry project
project_client = AIProjectClient(
    endpoint=project_connection_string, 
    credential=DefaultAzureCredential()
)

with project_client:
    # Create an instance of the CodeInterpreterTool, which is responsible for generating the bar chart
    code_interpreter = CodeInterpreterTool()

    # The CodeInterpreterTool needs to be included in creation of the agent so that it can be used
    agent = project_client.agents.create_agent(
        model=model,
        name="my-agent-barchart",
        instructions="You are a helpful agent that creates clear, well-labeled charts.",
        tools=code_interpreter.definitions,
        tool_resources=code_interpreter.resources,
    )
    print(f"Created agent, agent ID: {agent.id}")

    # Create a thread which is a conversation session between an agent and a user.
    thread = project_client.agents.threads.create()
    print(f"Created thread, thread ID: {thread.id}")

    # Create a prompt with clearer formatting
    prompt = """Could you please create a bar chart using the following health plan data and save it as 'health-plan-comparison.png'?

    Data:
    Provider          | Monthly Premium | Deductible | Out-of-Pocket Limit
    Northwind         | $300           | $1,500     | $6,000
    Aetna             | $350           | $1,000     | $5,500
    United Health     | $250           | $2,000     | $7,000
    Premera           | $200           | $2,200     | $6,500

    Please create a comprehensive chart showing all three metrics (Premium, Deductible, Out-of-Pocket Limit) for easy comparison.
    """
    
    # Create a message, with the prompt being the message content that is sent to the model
    message = project_client.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
    )
    print(f"Created message, message ID: {message.id}")

    # Run the agent to process the message in the thread
    run = project_client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
    print(f"Run finished with status: {run.status}")

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")
    elif run.status == "completed":
        print("Agent successfully completed the task!")

    # Get all messages from the thread
    messages = project_client.agents.messages.list(thread_id=thread.id)
    
    # Convert ItemPaged to list to get count and iterate
    messages_list = list(messages)
    print(f"Number of messages: {len(messages_list)}")
    files_saved = 0
    
    # Iterate through each message to find file attachments
    for message in messages_list:
        print(f"Message ID: {message.id}, Role: {message.role}")
        
        # Save generated image files
        for image_content in message.image_contents:
            file_id = image_content.image_file.file_id
            file_name = f"{file_id}_image_file.png"
            project_client.agents.files.save(file_id=file_id, file_name=file_name)
            print(f"Saved image file: {file_name}")
            files_saved += 1

        # Process file path annotations
        for file_path_annotation in message.file_path_annotations:
            print(f"File Paths:")
            print(f"Type: {file_path_annotation.type}")
            print(f"Text: {file_path_annotation.text}")
            print(f"File ID: {file_path_annotation.file_path.file_id}")
            
        
    print(f"Total files saved: {files_saved}")
    
    # Delete the agent once done
    project_client.agents.delete_agent(agent.id)
    print("Deleted agent")