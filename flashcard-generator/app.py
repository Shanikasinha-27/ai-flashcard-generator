from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")

# Create Hugging Face client
client = InferenceClient(
    api_key=hf_token
)

# Select model
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

# Send prompt to the model
prompt = """
Create 5 flashcards about Artificial Intelligence.

For each flashcard, give:
Question:
Answer:

Keep the answers short and simple.
"""

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# Display the response
print(response.choices[0].message.content)