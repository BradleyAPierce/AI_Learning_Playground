"""
Healthcare Sales Qualifying Questions Generator
=============================================

This Streamlit application generates qualifying questions for healthcare sales representatives
based on client situations or pain points. It uses OpenAI's GPT model to generate relevant
questions and explanations.

Author: Bradley Pierce
Date Created: May 07, 2024

How to Run:
-----------
1. Command Line Interface:
   python3 Lonely_Octopus_Bootcamp/healthcarequestions.py --cli

2. Web Interface:
   streamlit run Lonely_Octopus_Bootcamp/healthcarequestions.py

Requirements:
------------
- Python 3.8+
- OpenAI API key (set in Streamlit secrets or .env file)
- Required packages (see requirements.txt)
"""

# Standard library imports
import os
import asyncio
import sys
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import streamlit as st

# Third-party imports
try:
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.memory import ConversationBufferMemory
    from langchain.tools import Tool
except ImportError as e:
    st.error(f"Required packages not installed. Please run: pip install -r requirements.txt\nError: {str(e)}")
    st.stop()

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')

# Try to load from .env file only if it exists (suppress warnings)
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

def get_api_key() -> Optional[str]:
    """
    Get the OpenAI API key from Streamlit secrets or environment variables.
    
    Returns:
        Optional[str]: The API key if found, None otherwise
    """
    # Try Streamlit secrets first
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        # Fall back to environment variable
        return os.getenv('OPENAI_API_KEY')

# Get API key
OPENAI_API_KEY = get_api_key()

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it in Streamlit secrets or .env file.")
    st.info("To set up your API key in Streamlit Cloud:\n"
            "1. Go to your app's settings\n"
            "2. Add a secret with key 'OPENAI_API_KEY'\n"
            "3. Add your OpenAI API key as the value")
    st.stop()

def create_task_generator_agent() -> AgentExecutor:
    """
    Creates an AI agent that helps Sales people create effective questions to their clients.
    
    Returns:
        AgentExecutor: The configured Question Generator agent
    """
    try:
        # Initialize the language model
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
        
        # Create a real function for the single-input tool
        def task_generator_func(goal_text: str) -> str:
            return f"Received goal: {goal_text}"

        # Create a Tool for task generation
        task_generator_tool = Tool(
            name="TaskGenerator",
            func=task_generator_func,
            description="Breaks down a prompt question into qualifying questions to ask the client"
        )
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Healthcare Sales Qualifying Questions Generator that helps sales representatives develop effective qualifying questions. 
            When given a client's situation or pain point, generate exactly 10 qualifying questions. 
            For each question, you MUST provide a clear explanation of why it's an effective question to ask. 
            Format your response as a numbered list with explanations."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the agent
        agent = create_react_agent(llm, [task_generator_tool], prompt)
        
        # Create the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[task_generator_tool],
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            ),
            handle_parsing_errors=True,
            verbose=True
        )
        
        return agent_executor
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None

async def generate_tasks(goal: str) -> List[Tuple[str, str]]:
    """
    Generates a list of 10 qualifying questions with explanations for a given healthcare sales situation.
    
    Args:
        goal (str): The client's situation or pain point to generate questions for
        
    Returns:
        List[Tuple[str, str]]: A list of tuples containing (question, explanation) pairs
    """
    try:
        agent = create_task_generator_agent()
        if not agent:
            return []
            
        response = agent.invoke({
            "input": f"""Based on this healthcare client situation: {goal}
            
            You MUST provide:
            1. Exactly 10 qualifying questions
            2. For EACH question, you MUST include an explanation of why it's an effective question to ask
            3. Format your response as a numbered list with explanations.
            
            Focus on questions that will help uncover the client's needs, pain points, and decision-making process.
            Remember to include an explanation for EVERY question."""
        })
        
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
    st.set_page_config(
        page_title="Healthcare Sales Qualifying Questions Generator",
        page_icon="🏥",
        layout="centered"
    )
    
    st.title("Healthcare Sales Qualifying Questions Generator")
    st.write("Generate 10 effective qualifying questions based on your client's situation.")

    user_goal = st.text_area(
        "Enter your client's situation or pain point",
        placeholder="e.g., Client is struggling with patient data management and compliance"
    )

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
    for i, (question, explanation) in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print(f"Explanation: {explanation}")
        print("-" * 80)

if __name__ == "__main__":
    # If running with "python healthcarequestions.py --cli", use CLI
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        asyncio.run(run_cli())
    else:
        run_web_interface()

