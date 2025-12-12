import os
from groq import Groq           #type: ignore
from dotenv import load_dotenv  #type: ignore

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Hello, are you Llama 3?"}
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)