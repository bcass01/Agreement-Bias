import json
import os
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv          # type: ignore
from openai import AsyncOpenAI          # type: ignore
from anthropic import AsyncAnthropic    # type: ignore
from groq import AsyncGroq              # type: ignore

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
# Make sure this matches the filename output by your builder script
INPUT_FILE = "agreement_bias_subjective_dataset_triplets.json"
OUTPUT_FILE = "raw_model_responses_triplets.json"

# --- MODEL CONFIGURATIONS ---
MODELS = {
    # OpenAI: GPT-4o
    #"gpt-4o": "gpt-4o", 
    
    # Anthropic: Claude 4.5 Sonnet
    #"claude-4.5-sonnet": "claude-sonnet-4-5-20250929",
    
    # Llama 3.3 (via Groq): 70B Model
    "llama-3-70b": "llama-3.3-70b-versatile" 
}

# --- CLIENT INITIALIZATION ---
# Ensure you have OPENAI_API_KEY, ANTHROPIC_API_KEY, and GROQ_API_KEY in your .env file
try:
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Error initializing clients. Missing keys? {e}")

async def query_model(model_family, prompt):
    """
    Sends a prompt to the specified model family and returns the text response.
    Includes basic error handling to prevent the whole script from crashing.
    """
    if not prompt: 
        return ""
        
    try:
        if "gpt" in model_family:
            response = await openai_client.chat.completions.create(
                model=MODELS[model_family],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # Low temp for reproducibility
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
        print(f"\n[!] Error calling {model_family}: {e}")
        return None

async def main():
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}. Did you run the builder script?")
        return

    with open(INPUT_FILE, 'r') as f:
        dataset = json.load(f)

    results = []
    print(f"Starting evaluation on {len(dataset)} items...")
    print(f"Models: {list(MODELS.keys())}")

    # 2. Processing Loop
    for item in tqdm(dataset):
        item_result = item.copy() 
        item_result["responses"] = {}
        
        # We will collect tasks for all models and all 3 prompt types here
        tasks = []
        model_keys = list(MODELS.keys())
        
        # Create Async Tasks
        for model_key in model_keys:
            prompts = item["prompts"]
            # Append 3 tasks per model (Neutral, Positive, Negative)
            tasks.append(query_model(model_key, prompts["neutral"]))
            tasks.append(query_model(model_key, prompts["framed_positive"]))
            tasks.append(query_model(model_key, prompts["framed_negative"]))
        
        # Run all tasks in parallel for this item
        # Total tasks = (Num Models) * 3
        responses = await asyncio.gather(*tasks)
        
        # 3. Map Results Back
        # We must unpack in the exact same order we appended them
        idx = 0
        for model_key in model_keys:
            item_result["responses"][model_key] = {
                "neutral_response": responses[idx],
                "framed_positive_response": responses[idx+1],
                "framed_negative_response": responses[idx+2]
            }
            idx += 3 # Move index forward by 3 for the next model
            
        results.append(item_result)

        # 4. Save Incrementally (Overwrite file after every item)
        # This prevents total data loss if the script crashes halfway
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nSuccess! Saved responses to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())