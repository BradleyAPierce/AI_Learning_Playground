"""
Week 0: Kickoff & Laying the AI Agent Foundations
==========================================

This file contains the initial setup and exercises for Week 0 of the Lonely Octopus Bootcamp.

Details
The first week is all about laying the foundations for you to build the best AI agents. We&#39;ll cover what an AI agent really is.

And dive into their true capabilities so you can pick the best projects to build.

Then we&#39;ll help you set up your full agent tech-stack, map your project goals and show you how to get the most out of the bootcamp.

Author: Bradley Pierce
Date Created: May, 2025

How to Run:

https://share.streamlit.io/

-----------
Run this script directly in the terminal using:
python3 Lonely_Octopus_Bootcamp/week0.py
"""

# Standard library imports
import os
from dotenv import load_dotenv

# Third-party imports
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def create_chef_agent(dish_name):
    """
    Creates an AI agent that acts as a World Class Chef and provides recipe information.
    
    Args:
        dish_name (str): The name of the dish to get recipe information for
        
    Returns:
        str: The AI's response with recipe information
    """
    # Initialize the OpenAI client with API key from environment variable
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Specify the model to use
    model = "gpt-4"  # You can change this to gpt-3.5-turbo or others

    # Format the chat messages
    messages = [
        {"role": "assistant", "content": "You are a World Class Chef"},
        {"role": "user", "content": f"Show me the ingredients, recipe and preparation method of this dish: {dish_name}. Organize your answer in clear and concise bullet points."}
    ]

    # Send the request
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    # Return the response
    return response.choices[0].message.content

def main():
    """
    Main function to run the program.
    """
    # Example usage
    dish = "chicken soup"
    recipe = create_chef_agent(dish)
    print(recipe)

if __name__ == "__main__":
    main()
