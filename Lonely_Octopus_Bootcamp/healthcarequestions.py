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

# Try to load from .env file only if it exists (suppress warnings)
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

# Get API key from Streamlit secrets first, then fall back to environment variables
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found in Streamlit secrets. Please check your configuration.")
    st.stop()

def create_task_generator_agent():
    """
    Creates an AI agent that helps Sales people create effective questions to their clients.
    
    Returns:
        Agent: The configured Question Generator agent
    """
    # Initialize the language model
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    
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
            "Format your response as a numbered list with explanations."
        )
    }

    # Initialize the agent with error handling
    agent = initialize_agent(
        tools=[task_generator_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
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
        list: A list of tuples containing (question, explanation) pairs
    """
    try:
        agent = create_task_generator_agent()
        response = agent.invoke({"input": f"""Based on this healthcare client situation: {goal}
        
        You MUST provide:
        1. Exactly 10 qualifying questions
        2. For EACH question, you MUST include an explanation of why it's an effective question to ask
        3. Format your response as a numbered list with explanations.
        
        Focus on questions that will help uncover the client's needs, pain points, and decision-making process.
        Remember to include an explanation for EVERY question."""})
        
        # Extract the response from the agent's output
        if isinstance(response, dict) and "output" in response:
            output = response["output"]
        else:
            output = str(response)
            
        # Parse the output into questions and explanations
        questions = []
        current_question = None
        current_explanation = None
        
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a new question (starts with a number)
            if line[0].isdigit() and '. ' in line:
                # Save previous question if it exists
                if current_question and current_explanation:
                    questions.append((current_question, current_explanation))
                
                # Start new question
                current_question = line.split('. ', 1)[1]
                current_explanation = None
            elif line.lower().startswith('explanation:'):
                current_explanation = line.split(':', 1)[1].strip()
        
        # Add the last question
        if current_question and current_explanation:
            questions.append((current_question, current_explanation))
            
        return questions
    except Exception as e:
        error_message = f"An error occurred while generating questions: {str(e)}"
        print(error_message)
        return []

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
                    questions = asyncio.run(generate_tasks(user_goal))
                    if questions:
                        st.success("Here are your 10 qualifying questions with explanations:")
                        for i, (question, explanation) in enumerate(questions, 1):
                            st.markdown(f"**Question {i}:** {question}")
                            st.markdown(f"*Explanation:* {explanation}")
                            st.markdown("---")  # Add a separator between questions
                    else:
                        st.error("No questions were generated. Please try again.")
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
    questions = await generate_tasks(user_goal)
    print("\nGenerated Qualifying Questions with Explanations:")
    for question, explanation in questions:
        print(f"Question: {question}")
        print(f"Explanation: {explanation}")

if __name__ == "__main__":
    # If running with "python week1.py --cli", use CLI
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        import asyncio
        asyncio.run(run_cli())
    else:
        run_web_interface()

