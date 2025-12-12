import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from huggingface_hub import AsyncInferenceClient

load_dotenv()

# Define the models we want to test (Updated IDs)
MODELS_TO_TEST = {
    "gpt-4o": "gpt-4o",
    #"claude-3-5-sonnet": "claude-3-5-sonnet-latest",
    #"llama-3-70b": "meta-llama/Llama-3.3-70B-Instruct"
}

async def test_model(name, model_id):
    print(f"\n--- Testing {name} ({model_id}) ---")
    try:
        if "gpt" in name:
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "Say 'Hello'"}],
                max_tokens=5
            )
        elif "claude" in name:
            client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            await client.messages.create(
                model=model_id,
                max_tokens=5,
                messages=[{"role": "user", "content": "Say 'Hello'"}]
            )
        elif "llama" in name:
            client = AsyncInferenceClient(token=os.getenv("HF_TOKEN"))
            await client.chat_completion(
                model=model_id,
                messages=[{"role": "user", "content": "Say 'Hello'"}],
                max_tokens=5
            )
        print(f"✅ {name}: Success")
    except Exception as e:
        print(f"❌ {name}: Failed")
        print(f"   Error: {e}")

async def main():
    tasks = [test_model(name, mid) for name, mid in MODELS_TO_TEST.items()]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())