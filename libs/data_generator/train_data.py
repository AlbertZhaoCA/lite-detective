import json
import os
import random
from libs.llm_sdk.local_llm import LLM
from utils.templates import build_training_data_prompt

def load_policies(file_path="data/policy.jsonl"):
    """Load a JSONL file and return a list of dictionaries.
    Args:
        file_path: The path to the JSONL file.
    Returns:
        A list of dictionaries.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError:
                continue
    return data


def random_policy():
    """Randomly select a policy from the list of policies.
        
    Returns:
        dict: A randomly selected policy.
    """
    policies = load_policies()
    return random.choice(policies)

def generate_training_data()-> str:
    """
    Generate training data for the model.
    
    Returns:
        str: The generated training data.
    """
    def create_llm():
        llm = LLM(os.getenv("AGENT_MODEL_PATH"))
        return llm
    
    policy = random_policy()
    print(f"Generating training data for policy: {policy['summary']}")
    
    llm = create_llm()
    try:
        res = llm.generate(build_training_data_prompt(policy['summary'], policy['policy']))
        return res
    finally:
        # Clean up the LLM instance
        llm = None

if __name__ == "__main__":
    print("Loading policies...")
    policies = load_policies()
    print(f"Loaded {len(policies)} policies.")
    print("Random policy:")
    print(random_policy())