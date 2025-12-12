from dotenv import load_dotenv          # type: ignore
load_dotenv()

import json
import os
import asyncio
from tqdm import tqdm
from openai import AsyncOpenAI          # type: ignore
from anthropic import AsyncAnthropic    # type: ignore
from groq import AsyncGroq              # type: ignore

# Load API Keys (ensure these are in your environment variables)
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# Llama 3 via Hugging Face (requires a Pro account or dedicated endpoint usually, 
# or use a provider like Groq/Together if you have those keys)
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

INPUT_FILE = "agreement_bias_objective_dataset_v2.json"
OUTPUT_FILE = "raw_model_responses.json"

# Model Configurations (As per your paper Section 3.4)
MODELS = {
    # OpenAI: Point to the generic alias to always get the current stable version
    "gpt-4o": "gpt-4o", 
    
    # Anthropic: Use the 'latest' alias to avoid 404s on old date-stamps
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    
    # Llama 3.3 (via Groq): 70B Model
    "llama-3-70b": "llama-3.3-70b-versatile" 
}
async def query_model(model_family, prompt):
    """Generic wrapper to call different model APIs"""
    try:
        if "gpt" in model_family:
            response = await openai_client.chat.completions.create(
                model=MODELS[model_family],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # Low temp as per Sec 3.4
                max_tokens=300
            )
            return response.choices[0].message.content
            
        elif "claude" in model_family:
            response = await anthropic_client.messages.create(
                model=MODELS[model_family],
                max_tokens=300,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
            
        elif "llama" in model_family:
            response = await groq_client.chat.completions.create(
                model=MODELS[model_family],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content

    except Exception as e:
        print(f"Error calling {model_family}: {e}")
        return None

async def main():
    with open(INPUT_FILE, 'r') as f:
        dataset = json.load(f)

    results = []

    print(f"Starting evaluation on {len(dataset)} items across {len(MODELS)} models...")

    for item in tqdm(dataset):
        # We process one item at a time, but run all models/prompts concurrently for speed
        tasks = []
        
        # Structure to hold this item's results
        item_result = item.copy() 
        item_result["responses"] = {}

        for model_key in MODELS.keys():
            # 1. Query Neutral
            tasks.append(query_model(model_key, item["prompts"]["neutral"]))
            # 2. Query Framed
            tasks.append(query_model(model_key, item["prompts"]["framed"]))
        
        # Wait for all API calls for this item
        responses = await asyncio.gather(*tasks)
        
        # Unpack results
        # The order matches how we appended tasks: Model A Neutral, Model A Framed, Model B Neutral...
        idx = 0
        for model_key in MODELS.keys():
            item_result["responses"][model_key] = {
                "neutral_response": responses[idx],
                "framed_response": responses[idx+1]
            }
            idx += 2
            
        results.append(item_result)

        #sleep so I don't hit the llama free tier limit
        #await asyncio.sleep(4)


        # Save intermediate results
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    print(f"Saved raw responses to {OUTPUT_FILE}")



if __name__ == "__main__":
    asyncio.run(main())