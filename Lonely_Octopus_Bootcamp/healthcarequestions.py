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
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

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
        return f"Received goal: {goal_text}"

    # Create a Tool for task generation
    task_generator_tool = Tool(
        name="TaskGenerator",
        func=task_generator_func,
        description="Breaks down a prompt question into qualifying questions to ask the client"
    )
    
    agent_kwargs={
        "system_message": (
            "You are a Healthcare Sales Qualifying Questions Generator that helps sales representatives develop effective qualifying questions. "
            "When given a client's situation or pain point, generate exactly 10 qualifying questions. "
            "For each question, you MUST provide a clear explanation of why it's an effective question to ask. "
            "Format your response as follows:\n\n"
            "• Question 1: [Question]\n"
            "  Explanation: [Detailed explanation]\n\n"
            "• Question 2: [Question]\n"
            "  Explanation: [Detailed explanation]\n\n"
            "Continue this exact format for all 10 questions.\n\n"
            "Focus on questions that:\n"
            "- Uncover specific pain points\n"
            "- Quantify the impact of current issues\n"
            "- Identify decision-making processes\n"
            "- Understand budget and timeline constraints\n"
            "- Reveal current solution limitations"
        )
    }

    # Initialize the agent with error handling
    agent = initialize_agent(
        tools=[task_generator_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,  # Add this to handle parsing errors
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
    try:
        agent = create_task_generator_agent()
        response = agent.invoke({"input": f"""Based on this healthcare client situation: {goal}
        
        You MUST provide:
        1. Exactly 10 qualifying questions
        2. For EACH question, you MUST include an explanation of why it's an effective question to ask
        3. Format each question and explanation EXACTLY as follows:
        
        • Question 1: [Question]
          Explanation: [Detailed explanation]
        
        • Question 2: [Question]
          Explanation: [Detailed explanation]
        
        Continue this exact format for all 10 questions. Each question MUST start with a bullet point (•).
        Focus on questions that will help uncover the client's needs, pain points, and decision-making process.
        Remember to include an explanation for EVERY question."""})
        
        # Extract the response from the agent's output
        if isinstance(response, dict) and "output" in response:
            output = response["output"]
        else:
            output = str(response)
            
        # Ensure proper formatting
        formatted_output = output.replace("1.", "• Question 1:")
        formatted_output = formatted_output.replace("2.", "• Question 2:")
        formatted_output = formatted_output.replace("3.", "• Question 3:")
        formatted_output = formatted_output.replace("4.", "• Question 4:")
        formatted_output = formatted_output.replace("5.", "• Question 5:")
        formatted_output = formatted_output.replace("6.", "• Question 6:")
        formatted_output = formatted_output.replace("7.", "• Question 7:")
        formatted_output = formatted_output.replace("8.", "• Question 8:")
        formatted_output = formatted_output.replace("9.", "• Question 9:")
        formatted_output = formatted_output.replace("10.", "• Question 10:")
        
        return formatted_output
    except Exception as e:
        error_message = f"An error occurred while generating questions: {str(e)}"
        print(error_message)
        return error_message

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
            try:
                with st.spinner("Generating your qualifying questions..."):
                    tasks = asyncio.run(generate_tasks(user_goal))
                    st.success("Here are your 10 qualifying questions with explanations:")
                    # Format the output with proper markdown spacing
                    formatted_output = tasks.replace("\n\n", "\n")  # Remove extra newlines
                    # Ensure proper markdown formatting
                    st.markdown(formatted_output, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try again with a different input or contact support if the issue persists.")

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

