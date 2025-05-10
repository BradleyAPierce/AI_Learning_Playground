"""
Week 1: How to Build An AI Agent
==========================================

We'll walk you through the fundamentals frameworks you need to build AI agents.

You'll learn how to get the most out of each component - the prompts, models, tools, knowledge, memory and more.

And you'll learn how an agent 'thinks' so you can more easily map out the best agentic system.

By the end of the week, you'll know the key fundamentals and have built a simple Agent that focuses on executing one task very well.

Author: Bradley Pierce
Date Created: May 07, 2025

How to Run:

https://share.streamlit.io/


-----------
1. Command Line Interface:
   python3 Lonely_Octopus_Bootcamp/healthcarequestions.py

2. Web Interface:
   streamlit run Lonely_Octopus_Bootcamp/healthcarequestions.py -- --web
"""

# Standard library imports
import os
import asyncio
import sys
from dotenv import load_dotenv
import streamlit as st

# Third-party imports
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool, Tool

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')

# Load environment variables from .env file (only in local environment)
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    print(f"Loaded .env file from: {env_path}")
else:
    print(f"Warning: .env file not found at: {env_path}")

# Get API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your environment or .env file.")

def create_task_generator_agent():
    """
    Creates an AI agent that helps Sales people create effective questions to their clients.
    
    Returns:
        Agent: The configured Question Generator agent
    """
    # Initialize the language model
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    # Create a real function for the single-input tool
    def task_generator_func(goal_text):
        # Here, you could call another LLM, or just echo for now
        return f"Received goal: {goal_text}"

    # Create a Tool for task generation
    task_generator_tool = Tool(
        name="TaskGenerator",
        func=task_generator_func,
        description="Breaks down a prompt question into qualifing questions to ask the client"
    )
    
    agent_kwargs={
        "system_message": (
            "You are a Healthcare Sales Qualifying Questions Generator that helps sales representatives develop effective qualifying questions. "
            "When given a client's situation or pain point, generate exactly 10 qualifying questions. "
            "For each question, provide a clear explanation of why it's an effective question to ask. "
            "Format your response as follows:\n\n"
            "1. First Question?\n"
            "   - Why this is effective: [explanation]\n\n"
            "2. Second Question?\n"
            "   - Why this is effective: [explanation]\n\n"
            "And so on for all 10 questions.\n\n"
            "Focus on questions that:\n"
            "- Uncover specific pain points\n"
            "- Quantify the impact of current issues\n"
            "- Identify decision-making processes\n"
            "- Understand budget and timeline constraints\n"
            "- Reveal current solution limitations\n"
            "Never answer anything unrelated to healthcare sales qualifying questions."
        )
    }

    # Initialize the agent
    agent = initialize_agent(
        tools=[task_generator_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        ),
        agent_kwargs=agent_kwargs
    )
    
    return agent

async def generate_tasks(goal):
    """
    Generates a list of 10 qualifying questions with explanations for a given healthcare sales situation.
    
    Args:
        goal (str): The client's situation or pain point to generate questions for
        
    Returns:
        str: The generated list of 10 qualifying questions with explanations
    """
    agent = create_task_generator_agent()
    response = agent.run(f"""Based on this healthcare client situation: {goal}
    
    Please provide:
    1. Exactly 10 qualifying questions
    2. For each question, explain why it's an effective question to ask
    3. Format each question and explanation in a clear, bullet-point style
    
    Focus on questions that will help uncover the client's needs, pain points, and decision-making process.""")
    return response

def run_web_interface():
    """
    Runs the Streamlit web interface for the Qualifying Questions Generator.
    """
    st.set_page_config(page_title="Healthcare Sales Qualifying Questions Generator", layout="centered")
    st.title("Healthcare Sales Qualifying Questions Generator")
    st.write("Generate 10 effective qualifying questions based on your client's situation.")

    user_goal = st.text_area("Enter your client's situation or pain point", 
                           placeholder="e.g., Client is struggling with patient data management and compliance")

    if st.button("Generate Qualifying Questions"):
        if user_goal.strip() == "":
            st.warning("Please enter a client situation or pain point.")
        else:
            with st.spinner("Generating your qualifying questions..."):
                tasks = asyncio.run(generate_tasks(user_goal))
                st.success("Here are your 10 qualifying questions with explanations:")
                # Format the output with proper markdown spacing
                formatted_output = tasks.replace("\n\n", "\n")  # Remove extra newlines
                st.markdown(formatted_output, unsafe_allow_html=True)

async def run_cli():
    """
    Runs the command-line interface for the Qualifying Questions Generator.
    """
    # Example usage
    user_goal = "Client is struggling with patient data management and compliance"
    print(f"\nGenerating qualifying questions for: {user_goal}\n")
    tasks = await generate_tasks(user_goal)
    print("\nGenerated Qualifying Questions with Explanations:")
    print(tasks)

if __name__ == "__main__":
    # If running with "python week1.py --cli", use CLI
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        import asyncio
        asyncio.run(run_cli())
    else:
        run_web_interface()

