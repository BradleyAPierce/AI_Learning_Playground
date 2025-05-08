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
-----------
1. Command Line Interface:
   python3 Lonely_Octopus_Bootcamp/week1.py

2. Web Interface:
   streamlit run Lonely_Octopus_Bootcamp/week1.py -- --web
"""

# Standard library imports
import os
import asyncio
import sys
from dotenv import load_dotenv

# Third-party imports
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# Load environment variables from .env file
load_dotenv(override=True)

# Get API key from environment variables
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

def create_task_generator_agent():
    """
    Creates an AI agent that helps Sales people create effective prompts for their AI agents.
    
    Returns:
        Agent: The configured Task Generator agent
    """
    # Initialize the language model
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    # Create a tool for task generation
    task_generator_tool = Tool(
        name="TaskGenerator",
        func=lambda x: "Task generation tool",
        description="Helps break down goals into actionable tasks"
    )
    
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
        agent_kwargs={
            "system_message": """You are a Task Generator agent that helps users break down their goals into small, achievable tasks.
            For any goal, analyze it and create a structured plan with specific actionable steps.
            Each task should be concrete, time-bound when possible, and manageable.
            Organize tasks in a logical sequence with dependencies clearly marked.
            
            For each task, also provide clear success criteria that answer "What does good look like?"
            Success criteria should be specific, measurable, and achievable.
            
            Format your response in two sections:
            1. Task Breakdown: List all tasks with their dependencies
            2. Success Criteria: For each task, list what success looks like
            
            Never answer anything unrelated to AI Agents."""
        }
    )
    
    return agent

async def generate_tasks(goal):
    """
    Generates a list of tasks and success criteria for a given goal using the Task Generator agent.
    
    Args:
        goal (str): The user's goal to break down into tasks
        
    Returns:
        str: The generated list of tasks and success criteria
    """
    agent = create_task_generator_agent()
    response = agent.run(f"""Break down this goal into tasks and success criteria: {goal}
    
    Please provide:
    1. A detailed task breakdown with clear steps
    2. For each task, specify what success looks like (success criteria)
    
    Format the response in two clear sections.""")
    return response

def run_web_interface():
    """
    Runs the Streamlit web interface for the Task Generator.
    """
    import streamlit as st
    
    st.set_page_config(page_title="AI Task Generator", layout="centered")
    st.title("Training Sales Professionals to create Effective Prompting Inputs")
    st.write("Break any goal into a set of actionable tasks with success criteria.")

    user_goal = st.text_area("Enter your prompt sales objective", 
                           placeholder="e.g. Start with a customer you currently do business with")

    if st.button("Generate Success Prompting Steps and learn what Success Criteria is"):
        if user_goal.strip() == "":
            st.warning("Please enter a sales prompt objective.")
        else:
            with st.spinner("Generating your sales prompt steps and success criteria..."):
                tasks = asyncio.run(generate_tasks(user_goal))
                st.success("Here are your prompting steps and success criteria:")
                st.markdown(f"```text\n{tasks}\n```")

async def run_cli():
    """
    Runs the command-line interface for the Task Generator.
    """
    # Example usage
    user_goal = "Learn how to prompt successful sales outcomes"
    print(f"\nBreaking down prompting steps for the goal: {user_goal}\n")
    tasks = await generate_tasks(user_goal)
    print("\nGenerated successful Prompting Techniques and Success Criteria:")
    print(tasks)

def main():
    """
    Main function to run the program.
    Determines whether to run the web interface or CLI based on command line arguments.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        run_web_interface()
    else:
        asyncio.run(run_cli())

if __name__ == "__main__":
    main()

