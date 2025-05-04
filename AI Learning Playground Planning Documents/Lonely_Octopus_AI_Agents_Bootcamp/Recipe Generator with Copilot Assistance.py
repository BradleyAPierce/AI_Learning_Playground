from openai import OpenAI
import os
from google.colab import userdata

# Read the API key from the secure Colab storage
api_key = userdata.get('OPENAI_API_KEY')

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Specify the model you want to use
model = "gpt-4o"

# Your input text
text = "overeasy eggs and bacon"

# Format the chat messages
messages = [
    {"role": "assistant", "content": "You are a World Class Chef"},
    {"role": "user", "content": f"Show me the ingredients, recipe, stove temperatures and preparation method of this dish:\n{text}. Organize your answer in clear and concise bullet points."}
]

# Send the request
response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0
)

# Display the response
response_message = response.choices[0].message.content
print(response_message)