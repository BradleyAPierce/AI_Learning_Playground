"""
WEEK TEMPLATE: Copy this file for each new week and update as needed.
- Update the week number, title, and description below.
- Customize the agent's system message and logic for each week's focus.
"""

# =========================
# Week X: [Your Week Title]
# =========================
# TODO: Update the week number and title above.

# Author: [Your Name]
# Date Created: [Update Date]

# How to Run:
# -----------
# 1. Command Line Interface:
#    python3 Lonely_Octopus_Bootcamp/weekX.py --cli
# 2. Web Interface:
#    streamlit run Lonely_Octopus_Bootcamp/weekX.py
# 3. Run upload and create app: https://share.streamlit.io/

# =========================
# SECTION: Imports
# =========================
import os
import asyncio
import sys
from dotenv import load_dotenv
import streamlit as st

# Third-party imports
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# =========================
# SECTION: Environment Setup
# =========================
# NOTE: .env is only needed for local development.
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    print(f"Loaded .env file from: {env_path}")
else:
    print(f"Warning: .env file not found at: {env_path}")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your environment or .env file.")

# =========================
# SECTION: Agent Configuration
# =========================
# TODO: Update the system_message for this week's learning objectives.
agent_kwargs = {
    "system_message": (
        "You are a Task Generator agent for Week X. "
        "Update this message with specific instructions for this week's focus."
    )
}

def create_task_generator_agent():
    """
    Creates an AI agent for this week's objectives.
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    def task_generator_func(goal_text):
        # TODO: Update this logic for each week's focus.
        return f"Received goal: {goal_text}"

    task_generator_tool = Tool(
        name="TaskGenerator",
        func=task_generator_func,
        description="Breaks down a goal into actionable tasks and success criteria. Input should be a single string describing the goal."
    )

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
    Generates a list of tasks and success criteria for a given goal using the Task Generator agent.
    """
    agent = create_task_generator_agent()
    response = agent.run(
        f"Break down this goal into tasks and success criteria: {goal}\n"
        "Please provide:\n"
        "1. A detailed task breakdown with clear steps\n"
        "2. For each task, specify what success looks like (success criteria)\n"
        "Format the response in two clear sections."
    )
    return response

# =========================
# SECTION: Streamlit Web Interface
# =========================
def run_web_interface():
    """
    Runs the Streamlit web interface for the Task Generator.
    """
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

# =========================
# SECTION: CLI Interface
# =========================
async def run_cli():
    """
    Runs the command-line interface for the Task Generator.
    """
    user_goal = "Learn how to prompt successful sales outcomes"
    print(f"\nBreaking down prompting steps for the goal: {user_goal}\n")
    tasks = await generate_tasks(user_goal)
    print("\nGenerated successful Prompting Techniques and Success Criteria:")
    print(tasks)

# =========================
# SECTION: Main Entry Point
# =========================
if __name__ == "__main__":
    # If running with "python weekX.py --cli", use CLI
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        asyncio.run(run_cli())
    else:
        run_web_interface()
